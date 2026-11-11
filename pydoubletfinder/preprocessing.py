"""Backward-compatible entry point for the preprocessing of DoubletFinder's merged matrices.

R's ``paramSweep`` and ``doubletFinder`` re-run Seurat's NormalizeData (LogNormalize),
FindVariableFeatures (vst, 2,000 genes), ScaleData and RunPCA on every merged real + artificial
matrix; the port of those steps is :mod:`pydoubletfinder.seurat`.
"""
from __future__ import annotations

import numpy as np


def preprocess_and_pca(
    counts,
    *,
    n_pcs: int = 10,
    n_top_genes: int = 2000,
    random_state: int = 0,
) -> np.ndarray:
    """Seurat 5 preprocessing of a ``genes x cells`` count matrix; returns ``(n_cells, n_pcs)``.

    ``random_state`` is unused (kept for compatibility): the PCA is exact.
    """
    from .seurat import preprocess

    return preprocess(counts, n_pcs=n_pcs, n_features=n_top_genes)
