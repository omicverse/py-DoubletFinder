"""Core DoubletFinder algorithm — Python port of the R implementation.

Mirrors the R package's computational primitives one-to-one:

    R function              Python function
    --------------------    ---------------------------
    paramSweep              param_sweep
    parallel_paramSweep     _parallel_param_sweep (internal)
    summarizeSweep          summarize_sweep
    find.pK                 find_pK
    doubletFinder           doublet_finder
    modelHomotypic          model_homotypic
    bimodality_coefficient  bimodality_coefficient  (in .bimodality)

The core numerical kernel — pANN = fraction of PC-space k-NN that are
artificial doublets — is bit-for-bit deterministic given:

  * the same artificial-doublet cell pairs (real.cells1, real.cells2)
  * the same PCA embedding of the merged (real + artificial) matrix
  * the same pK / nExp values

That is the surface along which ``tests/test_exact_match.py`` checks
agreement with the R reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .bimodality import bimodality_coefficient
from .kde import approxfun, bkde


# ---------------------------------------------------------------------------
# Public API: model_homotypic
# ---------------------------------------------------------------------------
def model_homotypic(annotations) -> float:
    """Model the homotypic-doublet proportion as sum of squared cluster frequencies.

    Matches R: ``sum((table(annotations)/length(annotations))^2)``.
    """
    ann = pd.Series(np.asarray(annotations))
    freq = ann.value_counts(normalize=True, dropna=False)
    return float((freq.values ** 2).sum())


# ---------------------------------------------------------------------------
# pANN kernel — the single place where neighborhood counts are computed
# ---------------------------------------------------------------------------
def _compute_knn(pca, n_neighbors, knn_fn=None, knn_backend="auto"):
    """Top-``n_neighbors`` nearest neighbours of every row of ``pca`` (self at
    column 0). Returns ``(dist, idx)`` both ``(n, n_neighbors)``, distance-sorted.

    Backends: an injectable ``knn_fn(X, n_neighbors)->(dist, idx)`` (e.g.
    omicverse's chunked-GPU kNN — **exact**, avoids the O(N²) distance matrix);
    ``'auto'`` uses exact sklearn below ~10^5 rows and approximate pynndescent
    (O(N log N)) above, so million-cell atlases stay tractable.
    """
    pca = np.asarray(pca, dtype=np.float32)
    n = pca.shape[0]
    n_neighbors = int(min(n_neighbors, n))
    if knn_fn is not None:
        dist, idx = knn_fn(pca, n_neighbors)
        return np.asarray(dist), np.asarray(idx)
    backend = knn_backend
    if backend == "auto":
        backend = "pynndescent" if n > 100000 else "sklearn"
    if backend == "pynndescent":
        try:
            from pynndescent import NNDescent
            index = NNDescent(pca, n_neighbors=n_neighbors, metric="euclidean",
                              random_state=0, low_memory=True)
            idx, dist = index.neighbor_graph
            o = np.argsort(dist, axis=1)
            return np.take_along_axis(dist, o, 1), np.take_along_axis(idx, o, 1)
        except Exception:
            backend = "sklearn"
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(pca)
    return nn.kneighbors(pca, return_distance=True)


def _ordered_neighbor_matrix(pca_coord: np.ndarray, n_real_cells: int,
                             n_neighbors=None, knn_fn=None,
                             knn_backend: str = "auto") -> np.ndarray:
    """Neighbour-ordering matrix for the first ``n_real_cells`` cells, 1-based
    (row 0 = the cell itself). Shape ``(n_neighbors, n_real_cells)``.

    When ``n_neighbors`` is given, only the top-``n_neighbors`` neighbours are
    computed with a **kNN** (query=real cells, index=all cells) instead of the
    old full ``cdist`` distance matrix + column-wise ``argsort`` — the latter is
    O(N²) memory/time and OOMs at ~10^5 cells. Legacy full-order path is kept
    (``n_neighbors=None``) for small exact runs / reference.
    """
    if n_neighbors is None:
        from scipy.spatial.distance import cdist
        dist = cdist(pca_coord, pca_coord[:n_real_cells], metric="euclidean")
        return np.argsort(dist, axis=0, kind="stable") + 1
    # Self-kNN over all cells; keep only the real-cell rows and lay them out
    # column-per-real-cell (1-based) to match the legacy order-matrix contract.
    _, idx = _compute_knn(pca_coord, n_neighbors, knn_fn=knn_fn,
                          knn_backend=knn_backend)
    return idx[:n_real_cells].T.astype(np.int64) + 1


def _compute_pann_from_order(
    order_mat: np.ndarray,
    k: int,
    n_real_cells: int,
) -> np.ndarray:
    """Given the pre-ordered neighbor matrix, compute pANN for each real cell.

    ``order_mat`` is 1-based (see ``_ordered_neighbor_matrix``). Row 0 is the
    cell itself (self-distance zero), so neighbors span rows 1..k.
    """
    # neighbors: k x n_real_cells, 1-based indices
    neighbors = order_mat[1 : k + 1, :n_real_cells]
    # Artificial doublets occupy the tail of the merged matrix
    is_artificial = neighbors > n_real_cells
    return is_artificial.sum(axis=0).astype(np.float64) / float(k)


# ---------------------------------------------------------------------------
# Default pK / pN sweep grids (match the R defaults exactly)
# ---------------------------------------------------------------------------
DEFAULT_PK_GRID = np.array(
    [0.0005, 0.001, 0.005] + list(np.round(np.arange(0.01, 0.301, 0.01), 10)),
    dtype=np.float64,
)
DEFAULT_PN_GRID = np.round(np.arange(0.05, 0.301, 0.05), 10)


def _filter_pk_grid(pk_grid: np.ndarray, n_real_cells: int) -> np.ndarray:
    """Drop pK values that would yield <1 neighbor at the smallest pN (0.05).

    Matches R: ``min.cells <- round(n/(1-0.05) - n); keep pK[round(pK*min.cells) >= 1]``.
    """
    min_cells = round(n_real_cells / (1.0 - 0.05) - n_real_cells)
    return pk_grid[np.round(pk_grid * min_cells).astype(int) >= 1]


# ---------------------------------------------------------------------------
# Public API: doublet_finder — the core scoring + classification routine
# ---------------------------------------------------------------------------
@dataclass
class DoubletFinderResult:
    """Container for the outputs of ``doublet_finder``.

    Attributes
    ----------
    pANN : 1-D np.ndarray of length ``n_real_cells``
        Proportion of artificial nearest neighbors for every real cell.
    classifications : 1-D np.ndarray of {"Singlet", "Doublet"}
        Top-``nExp`` pANN scores become "Doublet".
    pN, pK, nExp : parameters used (surfaced for reproducibility)
    neighbor_types : pd.DataFrame | None
        Per-cell breakdown of neighbor types (only when ``annotations`` given).
    column_name_pANN, column_name_DF : str
        ``pANN_{pN}_{pK}_{nExp}`` and ``DF.classifications_{pN}_{pK}_{nExp}``,
        matching R's ``seu@meta.data`` column naming.
    """

    pANN: np.ndarray
    classifications: np.ndarray
    pN: float
    pK: float
    nExp: int
    neighbor_types: pd.DataFrame | None = None
    column_name_pANN: str = ""
    column_name_DF: str = ""


def doublet_finder(
    *,
    pca_coord: np.ndarray,
    n_real_cells: int,
    pN: float = 0.25,
    pK: float,
    nExp: int,
    annotations: Sequence[str] | None = None,
    doublet_types1: Sequence | None = None,
    doublet_types2: Sequence | None = None,
    reuse_pANN: np.ndarray | None = None,
    knn_fn=None,
    knn_backend: str = "auto",
) -> DoubletFinderResult:
    """Python port of R ``doubletFinder`` — pANN + classification.

    Parameters
    ----------
    pca_coord : (n_real_cells + n_doublets, n_PCs) array
        PCA coordinates of the *merged* real + artificial matrix, in the same
        row order as R (real cells first, artificial cells after). This lets
        callers reuse the exact embedding R produced (via e.g. Seurat) for
        bit-for-bit parity.
    n_real_cells : int
        Number of real cells (the first ``n_real_cells`` rows of ``pca_coord``).
    pN, pK, nExp : see R docs.
    annotations : per-real-cell cluster labels, optional
        When given, also returns the neighbor-type breakdown per real cell.
    doublet_types1, doublet_types2 : cluster labels of each artificial doublet's
        parent cells (length n_doublets each). Required when ``annotations`` is
        given — these are produced by ``_sample_artificial_doublets``.
    reuse_pANN : pre-computed pANN vector — skips re-computing from PCA and
        just re-thresholds to ``nExp``. Matches R's ``reuse.pANN`` argument.
    """
    if reuse_pANN is not None:
        pANN = np.asarray(reuse_pANN, dtype=np.float64).ravel()
        if pANN.size != n_real_cells:
            raise ValueError(
                f"reuse_pANN length ({pANN.size}) must equal n_real_cells ({n_real_cells})"
            )
        classifications = _classify(pANN, nExp)
        return DoubletFinderResult(
            pANN=pANN,
            classifications=classifications,
            pN=pN,
            pK=pK,
            nExp=nExp,
            column_name_pANN=f"pANN_{pN}_{pK}_{nExp}",
            column_name_DF=f"DF.classifications_{pN}_{pK}_{nExp}",
        )

    if pca_coord.shape[0] < n_real_cells:
        raise ValueError(
            f"pca_coord has {pca_coord.shape[0]} rows but n_real_cells={n_real_cells}"
        )
    nCells = pca_coord.shape[0]
    k = int(round(nCells * pK))
    if k < 1:
        raise ValueError(f"pK={pK} yields k<1 at nCells={nCells}")

    # Only the top (k+1) neighbours per cell are ever needed (self + k). Compute
    # them with a kNN (query=real, index=all) — O(N·k) — instead of the old
    # O(N²) full cdist distance matrix, which OOMs on large atlases.
    order_mat = _ordered_neighbor_matrix(
        pca_coord, n_real_cells, n_neighbors=min(k + 1, nCells),
        knn_fn=knn_fn, knn_backend=knn_backend,
    )
    pANN = _compute_pann_from_order(order_mat, k, n_real_cells)

    neighbor_types = None
    if annotations is not None:
        if doublet_types1 is None or doublet_types2 is None:
            raise ValueError(
                "annotations requires doublet_types1 and doublet_types2"
            )
        assert order_mat is not None
        neighbor_types = _compute_neighbor_types(
            order_mat, k, n_real_cells,
            annotations=annotations,
            doublet_types1=doublet_types1,
            doublet_types2=doublet_types2,
        )

    classifications = _classify(pANN, nExp)
    return DoubletFinderResult(
        pANN=pANN,
        classifications=classifications,
        pN=pN,
        pK=pK,
        nExp=nExp,
        neighbor_types=neighbor_types,
        column_name_pANN=f"pANN_{pN}_{pK}_{nExp}",
        column_name_DF=f"DF.classifications_{pN}_{pK}_{nExp}",
    )


def _classify(pann: np.ndarray, n_exp: int) -> np.ndarray:
    """Top-``n_exp`` pANN scores → "Doublet", rest → "Singlet".

    Tie-breaking matches R's ``order(x, decreasing=TRUE)``: stable, indices in
    the original order for tied values. numpy's argsort is stable too, but
    R's decreasing=TRUE breaks ties in the OPPOSITE direction — we fix that
    by negating the values and using a stable ascending sort.
    """
    n = pann.size
    out = np.array(["Singlet"] * n, dtype=object)
    # Stable ascending argsort on -pann gives R's decreasing=TRUE order exactly
    idx = np.argsort(-pann, kind="stable")[: int(n_exp)]
    out[idx] = "Doublet"
    return out


def _compute_neighbor_types(
    order_mat: np.ndarray,
    k: int,
    n_real_cells: int,
    annotations: Sequence[str],
    doublet_types1: Sequence,
    doublet_types2: Sequence,
) -> pd.DataFrame:
    """Per-real-cell breakdown of the cluster makeup of artificial neighbors.

    Matches the R block inside ``doubletFinder`` when ``annotations != NULL``.
    """
    levels = pd.Categorical(np.asarray(annotations)).categories
    d1 = pd.Categorical(np.asarray(doublet_types1), categories=levels)
    d2 = pd.Categorical(np.asarray(doublet_types2), categories=levels)
    neighbors = order_mat[1 : k + 1, :n_real_cells]  # 1-based indices

    rows = []
    for i in range(n_real_cells):
        col = neighbors[:, i]
        artificial = col[col > n_real_cells] - n_real_cells - 1  # back to 0-based
        if artificial.size == 0:
            rows.append([np.nan] * len(levels))
            continue
        counts = (
            pd.Series(d1[artificial]).value_counts().reindex(levels, fill_value=0)
            + pd.Series(d2[artificial]).value_counts().reindex(levels, fill_value=0)
        )
        total = counts.sum()
        rows.append((counts / total).tolist() if total else [np.nan] * len(levels))
    return pd.DataFrame(rows, columns=list(levels))


# ---------------------------------------------------------------------------
# Artificial-doublet synthesis — the randomness entry point
# ---------------------------------------------------------------------------
def sample_artificial_doublets(
    n_real_cells: int,
    pN: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    real_cells1: Sequence[int] | None = None,
    real_cells2: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample two vectors of parent-cell indices (0-based) with replacement.

    ``n_doublets = round(n_real_cells/(1-pN) - n_real_cells)`` matches the R
    formula. Callers that need to match an R run exactly should pass the
    R-saved ``real.cells1`` / ``real.cells2`` directly.
    """
    n_doublets = int(round(n_real_cells / (1.0 - pN) - n_real_cells))
    if real_cells1 is not None and real_cells2 is not None:
        c1 = np.asarray(real_cells1, dtype=np.int64)
        c2 = np.asarray(real_cells2, dtype=np.int64)
        if c1.size != c2.size:
            raise ValueError("real_cells1 and real_cells2 must have same length")
        return c1, c2, c1.size
    if rng is None:
        rng = np.random.default_rng(seed)
    c1 = rng.integers(0, n_real_cells, size=n_doublets)
    c2 = rng.integers(0, n_real_cells, size=n_doublets)
    return c1, c2, n_doublets


# ---------------------------------------------------------------------------
# Parameter sweep (the `paramSweep` wrapper)
# ---------------------------------------------------------------------------
@dataclass
class SweepEntry:
    """One pN/pK entry in a paramSweep result — Python analogue of the R list."""

    pN: float
    pK: float
    pANN: pd.DataFrame  # 1-column DataFrame, matching R's structure

    @property
    def name(self) -> str:
        return f"pN_{self.pN}_pK_{self.pK}"


def param_sweep(
    *,
    pca_embeddings: dict,
    real_cell_names: Sequence[str],
    n_real_cells: int | None = None,
    pN_grid: np.ndarray | None = None,
    pK_grid: np.ndarray | None = None,
    knn_fn=None,
    knn_backend: str = "auto",
    progress: bool = True,
) -> list[SweepEntry]:
    """Python port of R's ``paramSweep``.

    Unlike R's version, this function expects pre-computed PCA embeddings
    keyed by pN. This is intentional: it lets the Python port (a) reproduce
    R results bit-for-bit when fed R-generated PCAs, and (b) remain
    preprocessing-agnostic so scanpy / omicverse users can plug in their
    own pipeline without dragging Seurat in.

    Parameters
    ----------
    pca_embeddings : dict[float, np.ndarray]
        Mapping from pN value to the PCA embedding of the merged real +
        artificial matrix at that pN. Each array is shape
        ``(n_real_cells + n_doublets_at_pN, n_PCs)``.
    real_cell_names : 1-D array of real cell IDs (used as pANN row names).
    pN_grid, pK_grid : override the default sweep grid if given.

    Returns
    -------
    list[SweepEntry] — one per (pN, pK) pair. Ordered pN-outer, pK-inner,
    matching the R output.
    """
    real_cell_names = list(real_cell_names)
    if n_real_cells is None:
        n_real_cells = len(real_cell_names)
    if pN_grid is None:
        pN_grid = DEFAULT_PN_GRID
    if pK_grid is None:
        pK_grid = DEFAULT_PK_GRID
    pK_grid = _filter_pk_grid(np.asarray(pK_grid, dtype=np.float64), n_real_cells)

    out: list[SweepEntry] = []
    _pn_iter = pN_grid
    if progress:
        try:
            from tqdm.auto import tqdm
            _pn_iter = tqdm(pN_grid, total=len(pN_grid),
                            desc="DoubletFinder: pN sweep", leave=False)
        except Exception:
            pass
    for pN in _pn_iter:
        if float(pN) not in {float(k) for k in pca_embeddings}:
            raise KeyError(
                f"pca_embeddings is missing entry for pN={pN}; expected keys {list(pca_embeddings)}"
            )
        pca_coord = np.asarray(
            pca_embeddings[float(pN)]
            if float(pN) in pca_embeddings
            else pca_embeddings[pN],
            dtype=np.float64,
        )
        nCells = pca_coord.shape[0]
        # Compute the ordering once for the largest k the pK grid needs, then
        # reuse (slice top-k) for every pK — a single kNN instead of the old
        # O(N²) full distance matrix.
        max_k = int(round(nCells * float(np.max(pK_grid))))
        order_mat = _ordered_neighbor_matrix(
            pca_coord, n_real_cells, n_neighbors=min(max_k + 1, nCells),
            knn_fn=knn_fn, knn_backend=knn_backend,
        )
        for pK in pK_grid:
            k = int(round(nCells * pK))
            if k < 1:
                continue
            pann = _compute_pann_from_order(order_mat, k, n_real_cells)
            df = pd.DataFrame({"pANN": pann}, index=real_cell_names)
            out.append(SweepEntry(pN=float(pN), pK=float(pK), pANN=df))
    return out


# ---------------------------------------------------------------------------
# Sweep summary / pK selection
# ---------------------------------------------------------------------------
def summarize_sweep(
    sweep_list: list[SweepEntry] | dict,
    GT: bool = False,
    GT_calls: Sequence[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Python port of R ``summarizeSweep``.

    ``sweep_list`` may be the list returned by :func:`param_sweep` or a dict
    keyed by the R-style name ``pN_{pN}_pK_{pK}``.

    The KDE/bimodality pipeline matches R:

        gkde <- approxfun(bkde(pANN, kernel="normal"))
        x    <- seq(min(pANN), max(pANN), length.out = n_real_cells)
        BC   <- bimodality_coefficient(gkde(x))

    If ``GT=True`` and ``GT_calls`` is a length-``n_real_cells`` vector of
    ground-truth labels ("Singlet"/"Doublet"), also fits a logistic regression
    and returns AUC per (pN, pK), matching the R behavior.
    """
    if isinstance(sweep_list, dict):
        entries = []
        for key, df in sweep_list.items():
            # key like "pN_0.25_pK_0.01"
            parts = key.split("_")
            pN = float(parts[1])
            pK = float(parts[3])
            entries.append(SweepEntry(pN=pN, pK=pK, pANN=df))
        sweep_list = entries

    rows = []
    for entry in sweep_list:
        pann = entry.pANN["pANN"].values if isinstance(entry.pANN, pd.DataFrame) else np.asarray(entry.pANN)
        kde = bkde(pann, kernel="normal")
        gkde = approxfun(kde.x, kde.y, rule=1)
        n = pann.size
        xs = np.linspace(pann.min(), pann.max(), n)
        density = gkde(xs)
        # Drop NaN endpoints if any (rule=1 can NaN the outermost points)
        density = density[np.isfinite(density)]
        bc = bimodality_coefficient(density)
        row = {"pN": entry.pN, "pK": entry.pK, "BCreal": bc}
        if GT:
            row["AUC"] = _auc_logreg(pann, np.asarray(GT_calls), seed=seed)
        rows.append(row)

    cols = ["pN", "pK", "AUC", "BCreal"] if GT else ["pN", "pK", "BCreal"]
    df = pd.DataFrame(rows)[cols]
    df["pN"] = df["pN"].astype("category")
    df["pK"] = df["pK"].astype("category")
    return df


def _auc_logreg(pann: np.ndarray, gt_calls: np.ndarray, seed: int | None) -> float:
    """Logistic-regression AUC on a random 50/50 split — matches R's ROCR path."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = pann.size
    idx = rng.permutation(n)
    train = idx[: n // 2]
    test = idx[n // 2 : 2 * (n // 2)]
    # R levels = c("Doublet","Singlet") — treat "Singlet" as positive class
    # since R uses it as the reference level (factor level 2) in glm binomial.
    y = (gt_calls == "Singlet").astype(int)
    model = LogisticRegression().fit(pann[train].reshape(-1, 1), y[train])
    prob = model.predict_proba(pann[test].reshape(-1, 1))[:, 1]
    return float(roc_auc_score(y[test], prob))


def find_pK(sweep_stats: pd.DataFrame) -> pd.DataFrame:
    """Python port of R ``find.pK`` — compute BCmvn and (optional) MeanAUC per pK.

    Matches R: ``BCmetric = mean(BCreal) / sd(BCreal)^2`` per unique pK,
    where ``sd`` uses the unbiased estimator (ddof=1).
    """
    has_auc = "AUC" in sweep_stats.columns
    pk_values = pd.unique(sweep_stats["pK"])
    # pd.unique preserves insertion order, matching R's unique()
    rows = []
    for pk in pk_values:
        mask = sweep_stats["pK"] == pk
        bc = sweep_stats.loc[mask, "BCreal"].astype(float).values
        mean_bc = float(np.mean(bc))
        var_bc = float(np.var(bc, ddof=1))
        bc_metric = mean_bc / var_bc if var_bc != 0 else np.nan
        row = {
            "ParamID": len(rows) + 1,
            "pK": pk,
            "MeanBC": mean_bc,
            "VarBC": var_bc,
            "BCmetric": bc_metric,
        }
        if has_auc:
            row["MeanAUC"] = float(np.mean(sweep_stats.loc[mask, "AUC"].astype(float)))
        rows.append(row)

    cols = (
        ["ParamID", "pK", "MeanAUC", "MeanBC", "VarBC", "BCmetric"]
        if has_auc
        else ["ParamID", "pK", "MeanBC", "VarBC", "BCmetric"]
    )
    return pd.DataFrame(rows)[cols]
