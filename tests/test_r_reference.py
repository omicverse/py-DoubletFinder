"""Against stored DoubletFinder 2.0.6 / Seurat 5.5.1 output (no R needed).

References come from tests/data/make_r_reference.R on 500 cells x 3,000 genes of the J293t-dm sample
of the Xi & Li (2021) doublet benchmark.
"""
from __future__ import annotations

import pathlib

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp

from pydoubletfinder import DoubletFinder, seurat
from pydoubletfinder.rrng import RRandom

DATA = pathlib.Path(__file__).parent / "data"


def _hex(values):
    return np.array([float.fromhex(v) for v in values])


@pytest.fixture(scope="module")
def counts():
    return sp.csc_matrix(scipy.io.mmread(DATA / "df_counts.mtx.gz"), dtype=np.float64)   # genes x cells


@pytest.fixture(scope="module")
def merged(counts):
    c1, c2 = (np.array(line.split(), dtype=int) for line in (DATA / "df_parents.txt").read_text().splitlines())
    return sp.hstack([counts, (counts[:, c1] + counts[:, c2]) / 2]).tocsr()


def test_random_numbers():
    u, s1, s2, u42 = (DATA / "df_rng.txt").read_text().splitlines()
    r = RRandom(1)
    np.testing.assert_array_equal(r.unif_rand(5), _hex(u.split()))
    np.testing.assert_array_equal(r.sample(500, 10, replace=True) + 1, np.array(s1.split(), dtype=int))
    np.testing.assert_array_equal(r.sample(20000, 10) + 1, np.array(s2.split(), dtype=int))
    np.testing.assert_array_equal(RRandom().after_seurat_pca(2000).unif_rand(3), _hex(u42.split()))


def test_artificial_doublet_parents():
    c1, c2 = (np.array(line.split(), dtype=int) for line in (DATA / "df_parents.txt").read_text().splitlines())
    r = RRandom(1)
    np.testing.assert_array_equal(r.sample(500, c1.size, replace=True), c1)
    np.testing.assert_array_equal(r.sample(500, c2.size, replace=True), c2)


def test_loess(merged):
    mean, var = seurat._row_mean_var(merged)
    keep = var > 0
    fit = seurat.r_loess(np.log10(mean[keep]), np.log10(var[keep]), span=0.3)
    np.testing.assert_allclose(fit, _hex((DATA / "df_loess.txt").read_text().split()), rtol=0, atol=1e-12)


def test_variable_features(merged):
    # same genes; genes whose standardised variances are equal up to rounding may swap ranks
    ref = np.array((DATA / "df_hvg.txt").read_text().split(), dtype=int)
    assert set(seurat.vst(merged)) == set(ref)


def test_workflow_matches_r(counts):
    adata = anndata.AnnData(sp.csr_matrix(counts.T))
    df = DoubletFinder(adata, random_state=1)
    df.param_sweep(PCs=10, progress=False).summarize_sweep()
    bcmvn = df.find_pK()
    ref_bc = pd.read_csv(DATA / "df_bcmvn.tsv", sep="\t")
    assert df.optimal_pK == float(ref_bc.pK[np.argmax(_hex(ref_bc.BCmetric))])
    rel = np.abs(bcmvn.BCmetric.to_numpy(float) / _hex(ref_bc.BCmetric) - 1)
    assert np.mean(rel < 1e-6) >= 0.9, rel
    df.run(pN=0.25, nExp=round(0.075 * adata.n_obs), PCs=10)
    ref = pd.read_csv(DATA / "df_result.tsv", sep="\t")
    pcol = next(c for c in adata.obs if c.startswith("pANN_"))
    dcol = next(c for c in adata.obs if c.startswith("DF.classifications_"))
    np.testing.assert_allclose(adata.obs[pcol].to_numpy(float), _hex(ref.pANN), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(adata.obs[dcol].to_numpy(), ref["class"].to_numpy())
