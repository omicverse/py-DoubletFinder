"""Smoke tests: import surface + basic functional checks that don't
require an R install. Exact R-parity tests live in test_exact_match.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_import_surface():
    import pydoubletfinder as df

    for sym in (
        "DoubletFinder", "param_sweep", "summarize_sweep", "find_pK",
        "doublet_finder", "model_homotypic", "bimodality_coefficient",
        "bkde", "approxfun", "skewness", "kurtosis",
    ):
        assert hasattr(df, sym), sym
    assert df.__version__


def test_model_homotypic_matches_r_formula():
    from pydoubletfinder import model_homotypic

    # sum of squared frequencies: {A:3, B:1} -> (3/4)^2 + (1/4)^2 = 0.625
    assert model_homotypic(["A", "A", "A", "B"]) == pytest.approx(0.625)


def test_skewness_kurtosis_against_handcomputed():
    """Check skewness/kurtosis on a small vector against an R-equivalent
    direct computation."""
    from pydoubletfinder import kurtosis, skewness

    x = np.array([1.0, 2.0, 2.5, 3.0, 5.0])
    # Replicate the R formula by hand:
    n = x.size
    m = x.mean()
    d = x - m
    m2 = np.mean(d ** 2); m3 = np.mean(d ** 3); m4 = np.mean(d ** 4)
    s_ref = m3 / m2 ** 1.5 * np.sqrt(n * (n - 1)) / (n - 2)
    k_ref = m4 / m2 ** 2 - 3
    k_ref = ((n - 1) * ((n + 1) * k_ref - 3 * (n - 1)) / ((n - 2) * (n - 3))) + 3
    assert skewness(x) == pytest.approx(s_ref)
    assert kurtosis(x) == pytest.approx(k_ref)


def test_bimodality_on_bimodal_sample_is_high():
    from pydoubletfinder import bimodality_coefficient
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(-3, 0.5, 400), rng.normal(3, 0.5, 400)])
    uniform = rng.uniform(-3, 3, 800)
    assert bimodality_coefficient(bimodal) > bimodality_coefficient(uniform)


def test_doublet_finder_on_synthetic_pca():
    """Synthetic PCA where artificial doublets cluster with each other —
    ensures pANN is >0 for cells near the artificial cluster and classification
    picks up the nExp cells closest to it."""
    from pydoubletfinder import doublet_finder

    rng = np.random.default_rng(42)
    n_real = 50
    n_art = 20
    # 5 real cells sit on top of the artificial cluster -> high pANN
    real = rng.normal(0, 0.3, size=(n_real, 2))
    real[:5] = rng.normal(10, 0.3, size=(5, 2))
    art = rng.normal(10, 0.3, size=(n_art, 2))
    pca = np.vstack([real, art])

    res = doublet_finder(
        pca_coord=pca, n_real_cells=n_real,
        pN=0.25, pK=0.1, nExp=5,
    )
    assert res.pANN.shape == (n_real,)
    # The first 5 real cells are the ones sitting on the artificial cluster
    top5 = np.argsort(-res.pANN)[:5]
    assert set(top5.tolist()) == set(range(5))
    assert (res.classifications[:5] == "Doublet").all()
    assert (res.classifications[5:] == "Singlet").all()


def test_reuse_pANN_rethresholds_without_pca():
    from pydoubletfinder import doublet_finder

    pann = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    res = doublet_finder(
        pca_coord=np.zeros((5, 1)),
        n_real_cells=5,
        pN=0.25, pK=0.1, nExp=2,
        reuse_pANN=pann,
    )
    # Top-2 pANN are indices 3 (0.9) and 1 (0.5)
    assert list(res.classifications) == ["Singlet", "Doublet", "Singlet", "Doublet", "Singlet"]


def test_summarize_sweep_and_find_pK_shape():
    """End-to-end on a tiny synthetic dataset — just checks shapes, not values."""
    from pydoubletfinder import find_pK, param_sweep, summarize_sweep, DEFAULT_PN_GRID

    rng = np.random.default_rng(0)
    n_real = 80
    # Fabricate a PCA embedding at each pN
    pca_embeds = {}
    for pN in DEFAULT_PN_GRID:
        n_d = int(round(n_real / (1 - pN) - n_real))
        pca_embeds[float(pN)] = rng.normal(size=(n_real + n_d, 5))
    entries = param_sweep(
        pca_embeddings=pca_embeds,
        real_cell_names=[f"c{i}" for i in range(n_real)],
        n_real_cells=n_real,
    )
    assert len(entries) > 0
    stats = summarize_sweep(entries)
    assert set(stats.columns) == {"pN", "pK", "BCreal"}
    bcmvn = find_pK(stats)
    assert set(bcmvn.columns) >= {"ParamID", "pK", "MeanBC", "VarBC", "BCmetric"}
