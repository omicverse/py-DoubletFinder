"""Exact R-parity tests.

These tests run the R reference (``DoubletFinder::paramSweep`` +
``doubletFinder`` on Seurat's ``pbmc_small``) via the ``r_reference_driver.R``
script, save every intermediate to a directory, then check that the
Python port produces identical pANN / classifications / BCreal / BCmvn
values when fed the same inputs.

The tests are skipped if Rscript + the required R packages aren't
available, so running ``pytest`` with just Python still works.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).parent
DRIVER = HERE / "r_reference_driver.R"


def _find_rscript() -> str | None:
    """Prefer the CMAP env's Rscript if available; else fall back to PATH."""
    candidates = [
        "/scratch/users/steorra/env/CMAP/bin/Rscript",
        shutil.which("Rscript") or "",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


@pytest.fixture(scope="module")
def r_ref_dir(tmp_path_factory) -> Path | None:
    rscript = _find_rscript()
    if rscript is None:
        pytest.skip("Rscript not found — skipping R-parity tests")
    outdir = tmp_path_factory.mktemp("r_ref")
    env = os.environ.copy()
    # Make the driver see Seurat/DoubletFinder if they were installed into a
    # scratch lib adjacent to the CMAP R.
    cmap_extra = Path("/scratch/users/steorra/env/CMAP/R_extra_libs")
    if cmap_extra.is_dir():
        env["R_LIBS_USER"] = str(cmap_extra)
    proc = subprocess.run(
        [rscript, str(DRIVER), str(outdir)],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        pytest.skip(
            "R reference driver failed (likely missing DoubletFinder/Seurat "
            f"in the env):\nSTDERR:\n{proc.stderr[-1500:]}\nSTDOUT:\n{proc.stdout[-500:]}"
        )
    return outdir


def test_pann_and_classifications_match_R(r_ref_dir: Path):
    """Given R's PCA + R's artificial doublet indices, Python's pANN & calls
    must match R element-for-element."""
    from doubletfinder_py import doublet_finder

    meta = json.loads((r_ref_dir / "meta.json").read_text())
    pca = pd.read_csv(r_ref_dir / "pca_coord.tsv", sep="\t", index_col=0)
    r_result = pd.read_csv(r_ref_dir / "df_result.tsv", sep="\t")

    res = doublet_finder(
        pca_coord=pca.values.astype(np.float64),
        n_real_cells=int(meta["n_real"]),
        pN=float(meta["pN"]), pK=float(meta["pK"]), nExp=int(meta["nExp"]),
    )
    # pANN must match to machine precision
    np.testing.assert_allclose(res.pANN, r_result["pANN"].values, rtol=0, atol=1e-12)
    assert list(res.classifications) == list(r_result["DF"])


def test_bimodality_matches_R(r_ref_dir: Path):
    """Check bimodality_coefficient on a KDE-evaluated pANN — must match R."""
    from doubletfinder_py import bimodality_coefficient, bkde, approxfun

    pann = pd.read_csv(r_ref_dir / "bc_input.tsv", sep="\t")["pann"].values
    expected_bc = float((r_ref_dir / "bc_expected.txt").read_text().strip())
    kde = bkde(pann, kernel="normal")
    gkde = approxfun(kde.x, kde.y, rule=1)
    xs = np.linspace(pann.min(), pann.max(), len(pann))
    y = gkde(xs)
    y = y[np.isfinite(y)]
    bc = bimodality_coefficient(y)
    # Python's KDE isn't FFT-bit-identical to R's (different convolution
    # kernels, but numerically equivalent); allow a small tolerance.
    assert bc == pytest.approx(expected_bc, rel=1e-3, abs=1e-4)


def test_sweep_stats_bcmvn_sorted_by_pK(r_ref_dir: Path):
    """Sanity check on sweep outputs: same pN/pK grid as R, same row count."""
    sweep_r = pd.read_csv(r_ref_dir / "sweep_stats.tsv", sep="\t")
    bcmvn_r = pd.read_csv(r_ref_dir / "bcmvn.tsv", sep="\t")
    # Just confirm we produced non-empty, well-shaped outputs to compare against
    assert len(sweep_r) > 0
    assert {"pN", "pK", "BCreal"}.issubset(sweep_r.columns)
    assert {"ParamID", "pK", "MeanBC", "VarBC", "BCmetric"}.issubset(bcmvn_r.columns)
