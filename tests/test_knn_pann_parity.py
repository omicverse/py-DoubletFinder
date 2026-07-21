"""The kNN-based pANN (top-k neighbours) must equal the legacy full-cdist path
exactly. Replacing the O(N^2) distance matrix with a top-k kNN (query=real,
index=all) is a pure performance change — pANN is unchanged."""
import numpy as np

from pydoubletfinder.core import _ordered_neighbor_matrix, _compute_pann_from_order


def _pann(pca, n_real, k, **kw):
    om = _ordered_neighbor_matrix(pca, n_real, **kw)
    return _compute_pann_from_order(om, k, n_real)


def test_knn_pann_matches_full_cdist():
    rng = np.random.default_rng(0)
    n_real, n_art = 800, 300
    n = n_real + n_art
    pca = rng.standard_normal((n, 15)).astype(np.float64)
    for pK in (0.01, 0.05, 0.1):
        k = int(round(n * pK))
        ref = _pann(pca, n_real, k, n_neighbors=None)                  # full cdist
        knn = _pann(pca, n_real, k, n_neighbors=k + 1, knn_backend="sklearn")
        assert np.array_equal(ref, knn), f"pANN mismatch at pK={pK}"


def test_ordered_neighbor_matrix_layout():
    rng = np.random.default_rng(1)
    n_real, n = 50, 70
    pca = rng.standard_normal((n, 8)).astype(np.float64)
    om = _ordered_neighbor_matrix(pca, n_real, n_neighbors=10, knn_backend="sklearn")
    assert om.shape == (10, n_real)          # (n_neighbors, n_real)
    # row 0 is each real cell itself (1-based index == column + 1)
    assert np.array_equal(om[0], np.arange(1, n_real + 1))
