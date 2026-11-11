"""pydoubletfinder run on one task and seed, mirroring 02_run_doubletfinder.R.

Usage: python 03_run_pydoubletfinder.py <task_dir> <out_dir> <seed>
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import anndata
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import pydoubletfinder


def main():
    td, od, seed = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
    od.mkdir(parents=True, exist_ok=True)
    m = sp.csr_matrix(scipy.io.mmread(td / "counts.mtx.gz").T)          # cells x genes
    cells = (td / "cells.tsv").read_text().split("\n")[: m.shape[0]]
    adata = anndata.AnnData(m)
    adata.obs_names = cells
    t0 = time.perf_counter()
    df = pydoubletfinder.DoubletFinder(adata, random_state=seed)
    df.param_sweep(PCs=10, progress=False).summarize_sweep()
    bcmvn = df.find_pK()
    nexp = int(round(0.075 * adata.n_obs))
    df.run(pN=0.25, nExp=nexp, PCs=10)
    elapsed = time.perf_counter() - t0
    pcol = next(c for c in adata.obs if c.startswith("pANN_"))
    dcol = next(c for c in adata.obs if c.startswith("DF.classifications_"))
    pd.DataFrame({"cell": cells, "score": adata.obs[pcol].to_numpy(),
                  "class": adata.obs[dcol].str.lower().to_numpy()}).to_csv(
        od / "scores.tsv.gz", sep="\t", index=False, float_format="%.17g")
    bcmvn[["pK", "BCmetric"]].to_csv(od / "bcmvn.tsv", sep="\t", index=False, float_format="%.17g")
    (od / "timing.json").write_text(json.dumps({
        "engine": "pydoubletfinder", "version": pydoubletfinder.__version__, "python": sys.version.split()[0],
        "seed": seed, "pK": df.optimal_pK, "nExp": nexp, "seconds": elapsed}))
    print(td.name, seed, f"pK={df.optimal_pK:g} {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
