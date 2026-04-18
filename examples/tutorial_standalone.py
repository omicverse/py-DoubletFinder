"""Minimal end-to-end example — drop this into a Jupyter cell or run as a script.

Demonstrates the standalone pipeline on scanpy's built-in pbmc3k counts.
"""
from __future__ import annotations

import numpy as np
import scanpy as sc

from pydoubletfinder import DoubletFinder, model_homotypic


def main() -> None:
    adata = sc.datasets.pbmc3k()           # raw counts in .X
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    df = DoubletFinder(adata, random_state=0)
    df.param_sweep(PCs=10)
    df.summarize_sweep()
    bcmvn = df.find_pK()
    print(bcmvn.sort_values("BCmetric", ascending=False).head())

    # Estimate nExp from a 7.5% doublet rate, adjusted for homotypic doublets
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)
    homotypic = model_homotypic(adata.obs["leiden"])
    nExp = round((1 - homotypic) * 0.075 * adata.n_obs)

    df.run(pN=0.25, nExp=nExp, annotations="leiden")
    cols = [c for c in adata.obs.columns if c.startswith(("pANN_", "DF."))]
    print(adata.obs[cols].head())


if __name__ == "__main__":
    main()
