"""AnnData-native ``DoubletFinder`` class — high-level wrapper around
``pydoubletfinder.core`` with the same lifecycle as ``milor_py.Milo``
and ``monocle2_py.Monocle``.

Usage
-----
>>> from pydoubletfinder import DoubletFinder
>>> df = DoubletFinder(adata)
>>> df.param_sweep(PCs=10)          # populates df.sweep_stats, df.bcmvn
>>> df.find_pK()                    # chooses optimal pK
>>> df.run(pN=0.25, nExp=200)       # populates adata.obs pANN / classifications

Random draws follow R. Seurat's ``RunPCA`` calls ``set.seed(42)``, so in R the artificial doublets
of ``paramSweep`` (after its first pN) and of ``doubletFinder`` come from that state; with
``random_state=None`` (default) the same holds here, and the doublets equal R's. An integer
``random_state`` plays the role of ``set.seed(random_state)`` before ``paramSweep``/``doubletFinder``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from . import core as _core
from .rrng import RRandom
from .seurat import preprocess


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
    random_state: int | None = None   # None: R's state after Seurat RunPCA; int: set.seed()

    # populated as the pipeline runs
    sweep_list: list = field(default_factory=list, init=False)
    sweep_stats: pd.DataFrame | None = field(default=None, init=False)
    bcmvn: pd.DataFrame | None = field(default=None, init=False)
    optimal_pK: float | None = field(default=None, init=False)
    _rng: RRandom | None = field(default=None, init=False, repr=False)

    # --- internal helpers ---------------------------------------------------
    def _counts_gene_by_cell(self) -> sp.csc_matrix:
        """Return the genes x cells counts (Seurat orientation), sparse."""
        mat = self.adata.layers[self.layer] if self.layer is not None else self.adata.X
        return sp.csc_matrix(mat, dtype=np.float64).T.tocsc()

    def _stream(self, n_features: int) -> RRandom:
        """The R random stream: continues across calls, starts as R would."""
        if self._rng is None:
            self._rng = RRandom()
            if self.random_state is None:
                self._rng.after_seurat_pca(n_features)
            else:
                self._rng.set_seed(self.random_state)
        return self._rng

    def _merged_pca(self, counts, n_real, pN, PCs, n_top_genes, rng):
        """R parallel_paramSweep / doubletFinder: sample parents, average, Seurat PCA."""
        n_doublets = int(round(n_real / (1.0 - pN) - n_real))
        c1 = rng.sample(n_real, n_doublets, replace=True)
        c2 = rng.sample(n_real, n_doublets, replace=True)
        merged = sp.hstack([counts, (counts[:, c1] + counts[:, c2]) / 2.0]).tocsr()
        emb = preprocess(merged, n_pcs=PCs, n_features=n_top_genes)
        rng.after_seurat_pca(min(n_top_genes, merged.shape[0]))       # RunPCA(seed.use = 42)
        return emb, c1, c2

    # --- steps --------------------------------------------------------------
    def param_sweep(
        self,
        *,
        PCs: int = 10,
        pN_grid: np.ndarray | None = None,
        pK_grid: np.ndarray | None = None,
        n_top_genes: int = 2000,
        max_cells: int = 10_000,
        knn_fn=None,
        knn_backend: str = "auto",
        progress: bool = True,
    ) -> "DoubletFinder":
        """Run the pN/pK parameter sweep.

        For every pN in the grid, synthesize artificial doublets, preprocess,
        run PCA, and compute pANN on the ``pK_grid``. Results are stashed on
        ``self.sweep_list`` as a list of ``SweepEntry`` objects.

        To reproduce an R result exactly: instead of calling this, build the
        ``SweepEntry`` list yourself with PCA embeddings + cell index pairs
        from R, and assign it to ``self.sweep_list``.
        """
        counts = self._counts_gene_by_cell()
        rng = self._stream(n_top_genes)
        n_cells_total = counts.shape[1]

        # Down-sample to max_cells as R does (sample(), order kept)
        if n_cells_total > max_cells:
            real_idx = rng.sample(n_cells_total, max_cells, replace=False)
            counts = counts[:, real_idx]
        else:
            real_idx = np.arange(n_cells_total)
        cell_names = self.adata.obs_names[real_idx].to_numpy()
        n_real = counts.shape[1]

        pN_grid = _core.DEFAULT_PN_GRID if pN_grid is None else np.asarray(pN_grid)
        pK_grid = _core.DEFAULT_PK_GRID if pK_grid is None else np.asarray(pK_grid)
        pK_grid = _core._filter_pk_grid(pK_grid, n_real)

        pca_embeddings: dict[float, np.ndarray] = {}
        for pN in pN_grid:
            _log(f"[paramSweep] pN={pN:g} — generating artificial doublets & PCA")
            pca_embeddings[float(pN)], _, _ = self._merged_pca(counts, n_real, float(pN), PCs, n_top_genes, rng)

        self.sweep_list = _core.param_sweep(
            pca_embeddings=pca_embeddings,
            real_cell_names=list(cell_names),
            n_real_cells=n_real,
            pN_grid=pN_grid,
            pK_grid=pK_grid,
            knn_fn=knn_fn,
            knn_backend=knn_backend,
            progress=progress,
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
        knn_fn=None,
        knn_backend: str = "auto",
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
            counts = self._counts_gene_by_cell()
            n_real = counts.shape[1]
            rng = self._stream(n_top_genes)
            emb, c1, c2 = self._merged_pca(counts, n_real, pN, PCs, n_top_genes, rng)

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
                knn_fn=knn_fn, knn_backend=knn_backend,
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
