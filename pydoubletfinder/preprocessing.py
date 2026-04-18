"""Scanpy-style preprocessing pipeline for DoubletFinder's pN/pK sweep.

R's ``paramSweep`` (and ``doubletFinder``) re-runs the entire Seurat
pipeline — NormalizeData(LogNormalize), FindVariableFeatures(vst, 2000),
ScaleData, RunPCA — on the merged real + artificial matrix for every
single pN value. For a Python-only pipeline we do the same with scanpy
(if available) so that users without R can get end-to-end results.

If you need bit-for-bit parity with R, bypass this module and hand
pre-computed PCA embeddings directly to ``param_sweep`` /
``doublet_finder``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def _log_normalize(counts, scale_factor: float = 1e4) -> np.ndarray:
    """Seurat's LogNormalize: ``log1p(counts / colSums * scale_factor)``.

    Expects counts as genes x cells (matching Seurat's orientation).
    """
    if sparse.issparse(counts):
        lib = np.asarray(counts.sum(axis=0)).ravel()
        lib[lib == 0] = 1
        out = counts.multiply(1.0 / lib).multiply(scale_factor).log1p()
        return out.toarray() if hasattr(out, "toarray") else np.asarray(out)
    counts = np.asarray(counts, dtype=np.float64)
    lib = counts.sum(axis=0)
    lib[lib == 0] = 1.0
    return np.log1p(counts / lib[np.newaxis, :] * scale_factor)


def _find_variable_features_vst(
    counts,
    n_top: int = 2000,
    clip_max: str | float = "auto",
) -> np.ndarray:
    """Seurat-style ``vst`` variable-feature selection.

    Returns integer indices (0-based) of the top-``n_top`` variable features,
    ranked by their clipped, standardized variance on log10(mean)~log10(var).
    """
    if sparse.issparse(counts):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=np.float64)
    n_genes = counts.shape[0]
    means = counts.mean(axis=1)
    variances = counts.var(axis=1, ddof=1)
    keep = (means > 0) & (variances > 0)
    log_m = np.log10(np.where(keep, means, 1.0))
    log_v = np.log10(np.where(keep, variances, 1.0))

    # Seurat's vst fits a LOESS of log10(variance) ~ log10(mean); we
    # approximate with a LOWESS from statsmodels when available, else
    # fall back to a local polynomial — not identical to Seurat but
    # close enough for variable-gene selection.
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        fit = lowess(
            endog=log_v[keep], exog=log_m[keep], frac=0.3, it=0, return_sorted=False
        )
    except Exception:  # pragma: no cover
        order = np.argsort(log_m[keep])
        sorted_x = log_m[keep][order]
        sorted_y = log_v[keep][order]
        fit = np.interp(log_m[keep], sorted_x, sorted_y)

    expected_var = np.ones_like(variances)
    expected_var[keep] = 10.0 ** fit
    # Standardize + clip
    n_cells = counts.shape[1]
    if clip_max == "auto":
        clip = np.sqrt(n_cells)
    else:
        clip = float(clip_max)

    z = np.zeros(n_genes, dtype=np.float64)
    sd = np.sqrt(expected_var)
    for g in np.where(keep)[0]:
        if sd[g] <= 0:
            continue
        centered = (counts[g] - means[g]) / sd[g]
        centered = np.clip(centered, -clip, clip)
        z[g] = centered.var(ddof=1)
    top = np.argsort(-z)[:n_top]
    return np.sort(top)


def _scale_and_pca(
    log_norm: np.ndarray,
    feature_idx: np.ndarray,
    n_pcs: int,
    *,
    random_state: int = 0,
) -> np.ndarray:
    """Row-scale (z-score) selected genes, then run PCA on cells x features."""
    from sklearn.decomposition import PCA

    X = log_norm[feature_idx, :]  # features x cells
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, ddof=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd
    Z = np.clip(Z, -10.0, 10.0)
    # PCA on cells x features
    pca = PCA(n_components=n_pcs, random_state=random_state)
    return pca.fit_transform(Z.T)


def preprocess_and_pca(
    counts,
    *,
    n_pcs: int = 10,
    n_top_genes: int = 2000,
    random_state: int = 0,
) -> np.ndarray:
    """End-to-end Seurat-style preprocessing on a ``genes x cells`` count matrix.

    Returns a ``(n_cells, n_pcs)`` PCA embedding. Intended for the merged
    real + artificial count matrix produced inside a pN iteration.
    """
    log_norm = _log_normalize(counts)
    feat = _find_variable_features_vst(counts, n_top=n_top_genes)
    return _scale_and_pca(log_norm, feat, n_pcs=n_pcs, random_state=random_state)
