"""KernSmooth::bkde parity tests.

Confirms that ``doubletfinder_py.kde.bkde`` produces densities that match
R's ``KernSmooth::bkde`` closely on a few hand-checked vectors. We don't
require R to run these — instead we check mathematical properties that
any correct binned KDE must satisfy (integration ≈ 1, default bandwidth
matches the Silverman-normal formula, symmetric data gives symmetric
density).
"""
from __future__ import annotations

import numpy as np
import pytest

from doubletfinder_py.kde import _default_bandwidth, bkde


def test_default_bandwidth_matches_formula():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    # KernSmooth: h = del0 * (243/(35n))^(1/5) * sd(x)
    del0 = (1.0 / (4.0 * np.pi)) ** (1.0 / 10.0)
    expected = del0 * (243.0 / (35.0 * 500)) ** (1.0 / 5.0) * np.std(x, ddof=1)
    assert _default_bandwidth(x, "normal") == pytest.approx(expected)


def test_density_integrates_to_approx_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    res = bkde(x, kernel="normal")
    delta = res.x[1] - res.x[0]
    area = float(np.trapezoid(res.y, res.x)) if hasattr(np, "trapezoid") else float(np.trapz(res.y, res.x))
    assert abs(area - 1.0) < 0.02


def test_density_symmetry_for_symmetric_data():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 0.5, 1000), rng.normal(2, 0.5, 1000)])
    # Use a symmetric grid so reflected grid points map to each other exactly
    res = bkde(x, kernel="normal", range_x=(-6.0, 6.0), gridsize=401)
    mid = res.x.size // 2
    diff = res.y[:mid] - res.y[::-1][:mid]
    # Symmetric input + symmetric grid → density reflects to within sampling noise
    assert np.max(np.abs(diff)) < 0.05 * res.y.max()


def test_gridsize_and_range_defaults_match_r():
    """R uses gridsize=401 and range=[min-4h, max+4h] for the normal kernel."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    res = bkde(x, kernel="normal")
    assert res.x.size == 401
    h = _default_bandwidth(x, "normal")
    assert res.x[0] == pytest.approx(x.min() - 4.0 * h)
    assert res.x[-1] == pytest.approx(x.max() + 4.0 * h)
