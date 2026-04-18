"""AnnData-native ``DoubletFinder`` class — high-level wrapper around
``doubletfinder_py.core`` with the same lifecycle as ``milor_py.Milo``
and ``monocle2_py.Monocle``.

Usage
-----
>>> from doubletfinder_py import DoubletFinder
>>> df = DoubletFinder(adata)
>>> df.param_sweep(PCs=10)          # populates df.sweep_stats, df.bcmvn
>>> df.find_pK()                    # chooses optimal pK
>>> df.run(pN=0.25, nExp=200)       # populates adata.obs pANN / classifications
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from . import core as _core
from .preprocessing import preprocess_and_pca


class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def _log(msg: str, level: str = "info") -> None:
    c = {"info": Colors.BLUE, "ok": Colors.GREEN, "warn": Colors.WARNING, "err": Colors.FAIL}[level]
    print(f"{c}{msg}{Colors.ENDC}")


@dataclass
class DoubletFinder:
    """AnnData-native implementation of the DoubletFinder workflow."""

    adata: AnnData
    layer: str | None = None  # which adata layer holds raw counts (None = .X)
    random_state: int = 0

    # populated as the pipeline runs
    sweep_list: list = field(default_factory=list, init=False)
    sweep_stats: pd.DataFrame | None = field(default=None, init=False)
    bcmvn: pd.DataFrame | None = field(default=None, init=False)
    optimal_pK: float | None = field(default=None, init=False)

    # --- internal helpers ---------------------------------------------------
    def _counts_gene_by_cell(self) -> np.ndarray:
        """Return a genes x cells dense counts matrix (Seurat orientation)."""
        mat = self.adata.layers[self.layer] if self.layer is not None else self.adata.X
        if sp.issparse(mat):
            mat = mat.toarray()
        # anndata stores cells x genes; transpose to genes x cells
        return np.asarray(mat).T.astype(np.float64)

    # --- steps --------------------------------------------------------------
    def param_sweep(
        self,
        *,
        PCs: int = 10,
        pN_grid: np.ndarray | None = None,
        pK_grid: np.ndarray | None = None,
        n_top_genes: int = 2000,
        max_cells: int = 10_000,
    ) -> "DoubletFinder":
        """Run the pN/pK parameter sweep.

        For every pN in the grid, synthesize artificial doublets, preprocess,
        run PCA, and compute pANN on the ``pK_grid``. Results are stashed on
        ``self.sweep_list`` as a list of ``SweepEntry`` objects.

        To reproduce an R result exactly: instead of calling this, build the
        ``SweepEntry`` list yourself with PCA embeddings + cell index pairs
        from R, and assign it to ``self.sweep_list``.
        """
        rng = np.random.default_rng(self.random_state)
        counts = self._counts_gene_by_cell()
        n_cells_total = counts.shape[1]

        # Subsample real cells if above the R threshold
        if n_cells_total > max_cells:
            real_idx = rng.choice(n_cells_total, size=max_cells, replace=False)
            real_idx = np.sort(real_idx)
            counts = counts[:, real_idx]
            cell_names = self.adata.obs_names[real_idx].to_numpy()
        else:
            real_idx = np.arange(n_cells_total)
            cell_names = self.adata.obs_names.to_numpy()
        n_real = counts.shape[1]

        pN_grid = _core.DEFAULT_PN_GRID if pN_grid is None else np.asarray(pN_grid)
        pK_grid = _core.DEFAULT_PK_GRID if pK_grid is None else np.asarray(pK_grid)
        pK_grid = _core._filter_pk_grid(pK_grid, n_real)

        pca_embeddings: dict[float, np.ndarray] = {}
        for pN in pN_grid:
            _log(f"[paramSweep] pN={pN:g} — generating artificial doublets & PCA")
            c1, c2, n_doublets = _core.sample_artificial_doublets(
                n_real_cells=n_real, pN=float(pN), rng=rng
            )
            doublets = (counts[:, c1] + counts[:, c2]) / 2.0
            merged = np.concatenate([counts, doublets], axis=1)
            emb = preprocess_and_pca(
                merged,
                n_pcs=PCs,
                n_top_genes=n_top_genes,
                random_state=self.random_state,
            )
            pca_embeddings[float(pN)] = emb

        self.sweep_list = _core.param_sweep(
            pca_embeddings=pca_embeddings,
            real_cell_names=list(cell_names),
            n_real_cells=n_real,
            pN_grid=pN_grid,
            pK_grid=pK_grid,
        )
        return self

    def summarize_sweep(self, *, GT: bool = False, GT_calls=None) -> "DoubletFinder":
        self.sweep_stats = _core.summarize_sweep(
            self.sweep_list, GT=GT, GT_calls=GT_calls, seed=self.random_state
        )
        return self

    def find_pK(self) -> pd.DataFrame:
        """Compute BCmvn, cache the optimal pK, return the BCmvn table."""
        if self.sweep_stats is None:
            self.summarize_sweep()
        self.bcmvn = _core.find_pK(self.sweep_stats)
        best = self.bcmvn.loc[self.bcmvn["BCmetric"].idxmax(), "pK"]
        self.optimal_pK = float(best)
        _log(f"[find.pK] optimal pK = {self.optimal_pK}", level="ok")
        return self.bcmvn

    def run(
        self,
        *,
        pN: float = 0.25,
        pK: float | None = None,
        nExp: int,
        annotations: str | None = None,
        PCs: int = 10,
        n_top_genes: int = 2000,
        reuse_pANN: str | None = None,
    ) -> AnnData:
        """Run the final ``doubletFinder`` scoring + classification step.

        Writes two columns to ``self.adata.obs``:

            * ``pANN_{pN}_{pK}_{nExp}``
            * ``DF.classifications_{pN}_{pK}_{nExp}``
        """
        if pK is None:
            if self.optimal_pK is None:
                raise ValueError("pK not specified and no optimal pK cached — call find_pK() first")
            pK = self.optimal_pK

        if reuse_pANN is not None:
            pann = self.adata.obs[reuse_pANN].astype(float).values
            result = _core.doublet_finder(
                pca_coord=np.zeros((len(pann), 1)),  # unused
                n_real_cells=len(pann),
                pN=pN, pK=pK, nExp=int(nExp),
                reuse_pANN=pann,
            )
        else:
            rng = np.random.default_rng(self.random_state)
            counts = self._counts_gene_by_cell()
            n_real = counts.shape[1]

            c1, c2, _ = _core.sample_artificial_doublets(n_real_cells=n_real, pN=pN, rng=rng)
            doublets = (counts[:, c1] + counts[:, c2]) / 2.0
            merged = np.concatenate([counts, doublets], axis=1)
            emb = preprocess_and_pca(
                merged, n_pcs=PCs, n_top_genes=n_top_genes,
                random_state=self.random_state,
            )

            ann_vec = None
            d1 = d2 = None
            if annotations is not None:
                ann_vec = self.adata.obs[annotations].astype(str).values
                d1 = ann_vec[c1]
                d2 = ann_vec[c2]

            result = _core.doublet_finder(
                pca_coord=emb,
                n_real_cells=n_real,
                pN=pN, pK=pK, nExp=int(nExp),
                annotations=ann_vec,
                doublet_types1=d1, doublet_types2=d2,
            )

        self.adata.obs[result.column_name_pANN] = result.pANN
        self.adata.obs[result.column_name_DF] = result.classifications
        self.adata.uns.setdefault("doubletfinder", {})[result.column_name_pANN] = {
            "pN": pN, "pK": pK, "nExp": int(nExp),
        }
        _log(
            f"[doubletFinder] wrote {result.column_name_pANN} "
            f"and {result.column_name_DF}",
            level="ok",
        )
        return self.adata
