"""Head-to-head speed benchmark: R DoubletFinder vs doubletfinder_py.

Runs both on the pbmc3k counts TSV that examples/compare_R_vs_Python.ipynb
already produced. Reports wall time for:

  * pANN kernel alone (fed a pre-computed PCA embedding — the fair,
    algorithm-only comparison)
  * end-to-end doubletFinder() (preprocess + PCA + pANN)

The preprocessing phase differs between Seurat (R) and sklearn (Python),
so the end-to-end numbers reflect the whole stack rather than the
DoubletFinder algorithm in isolation.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import doubletfinder_py as dfp


HERE = Path(__file__).parent
WORK = HERE / "compare_out"
COUNTS = WORK / "counts.tsv"
RSCRIPT = "/scratch/users/steorra/env/CMAP/bin/Rscript"
R_LIBS = "/scratch/users/steorra/env/CMAP/R_extra_libs"


R_FULL_SCRIPT = """
suppressPackageStartupMessages({
  library(Seurat); library(SeuratObject); library(DoubletFinder)
})
args <- commandArgs(trailingOnly = TRUE)
counts_path <- args[[1]]; outpath <- args[[2]]
pN <- 0.25; pK <- 0.09; n_pcs <- 10
nExp <- as.integer(args[[3]])

counts <- as.matrix(read.table(counts_path, sep = "\\t", header = TRUE,
                               row.names = 1, check.names = FALSE))
set.seed(42)

t0 <- proc.time()[[3]]
seu <- CreateSeuratObject(counts = counts)
seu <- NormalizeData(seu, verbose = FALSE)
seu <- FindVariableFeatures(seu, selection.method = "vst",
                            nfeatures = 2000, verbose = FALSE)
seu <- ScaleData(seu, verbose = FALSE)
seu <- RunPCA(seu, npcs = n_pcs, verbose = FALSE)

real_cells <- colnames(seu)
n_real <- length(real_cells)
n_d <- round(n_real / (1 - pN) - n_real)
set.seed(42)
rc1 <- sample(real_cells, n_d, replace = TRUE)
rc2 <- sample(real_cells, n_d, replace = TRUE)
doublets <- (counts[, rc1] + counts[, rc2]) / 2
colnames(doublets) <- paste0("X", 1:n_d)
merged <- cbind(counts, doublets)

seu_w <- CreateSeuratObject(counts = merged)
seu_w <- NormalizeData(seu_w, verbose = FALSE)
seu_w <- FindVariableFeatures(seu_w, selection.method = "vst",
                              nfeatures = 2000, verbose = FALSE)
seu_w <- ScaleData(seu_w, verbose = FALSE)
seu_w <- RunPCA(seu_w, npcs = n_pcs, verbose = FALSE)
t_pre <- proc.time()[[3]] - t0

t1 <- proc.time()[[3]]
pca <- seu_w@reductions$pca@cell.embeddings[, 1:n_pcs]
dist.mat <- fields::rdist(pca)
nCells <- nrow(pca)
k <- round(nCells * pK)
pANN <- numeric(n_real)
for (i in 1:n_real) {
  neighbors <- order(dist.mat[, i])[2:(k + 1)]
  pANN[i] <- length(which(neighbors > n_real)) / k
}
classifications <- rep("Singlet", n_real)
classifications[order(pANN, decreasing = TRUE)[1:nExp]] <- "Doublet"
t_pann <- proc.time()[[3]] - t1

cat(sprintf("R_PRE=%.3f\\nR_PANN=%.3f\\n", t_pre, t_pann))
write.table(data.frame(cell = real_cells, pANN = pANN, DF = classifications),
            outpath, sep = "\\t", quote = FALSE, row.names = FALSE)
write.table(pca, sub("\\\\.tsv$", "_pca.tsv", outpath),
            sep = "\\t", quote = FALSE, col.names = NA)
"""


def time_r_end_to_end(nExp: int, runs: int = 3) -> tuple[float, float]:
    """Return (mean_pre_s, mean_pann_s) across `runs` runs."""
    script = WORK / "_bench_r.R"
    script.write_text(R_FULL_SCRIPT)
    pre_times, pann_times = [], []
    env = os.environ.copy(); env["R_LIBS_USER"] = R_LIBS
    for _ in range(runs):
        out_tsv = WORK / "_bench_out.tsv"
        proc = subprocess.run(
            [RSCRIPT, str(script), str(COUNTS), str(out_tsv), str(nExp)],
            env=env, capture_output=True, text=True, check=True,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("R_PRE="):
                pre_times.append(float(line.split("=")[1]))
            elif line.startswith("R_PANN="):
                pann_times.append(float(line.split("=")[1]))
    return float(np.mean(pre_times)), float(np.mean(pann_times))


def time_python_end_to_end(nExp: int, n_cells: int, runs: int = 3) -> tuple[float, float]:
    """Same decomposition on the Python side — preprocess timing vs pANN kernel timing."""
    import anndata as ad
    pre_times, pann_times = [], []
    counts = pd.read_csv(COUNTS, sep="\t", index_col=0)
    X = counts.values.T  # cells x genes (anndata orientation)
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=counts.columns),
                       var=pd.DataFrame(index=counts.index))
    for _ in range(runs):
        t0 = time.perf_counter()
        dfer = dfp.DoubletFinder(adata.copy(), random_state=42)
        rng = np.random.default_rng(42)
        counts_gxc = dfer._counts_gene_by_cell()
        n_real = counts_gxc.shape[1]
        c1, c2, _ = dfp.sample_artificial_doublets(n_real_cells=n_real, pN=0.25, rng=rng)
        doublets = (counts_gxc[:, c1] + counts_gxc[:, c2]) / 2.0
        merged = np.concatenate([counts_gxc, doublets], axis=1)
        from doubletfinder_py.preprocessing import preprocess_and_pca
        emb = preprocess_and_pca(merged, n_pcs=10, n_top_genes=2000, random_state=42)
        pre_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        res = dfp.doublet_finder(
            pca_coord=emb, n_real_cells=n_real,
            pN=0.25, pK=0.09, nExp=nExp,
        )
        pann_times.append(time.perf_counter() - t1)
    return float(np.mean(pre_times)), float(np.mean(pann_times))


def main():
    n_cells = pd.read_csv(COUNTS, sep="\t", index_col=0, nrows=1).shape[1]
    nExp = round(0.075 * n_cells)
    print(f"Dataset: pbmc3k — n_cells={n_cells}, nExp={nExp}")

    print("\n[R] timing end-to-end (3 runs)…")
    r_pre, r_pann = time_r_end_to_end(nExp, runs=3)
    print(f"  preprocess+PCA:  {r_pre*1000:8.1f} ms")
    print(f"  pANN kernel:     {r_pann*1000:8.1f} ms")

    print("\n[py] timing end-to-end (3 runs)…")
    py_pre, py_pann = time_python_end_to_end(nExp, n_cells, runs=3)
    print(f"  preprocess+PCA:  {py_pre*1000:8.1f} ms")
    print(f"  pANN kernel:     {py_pann*1000:8.1f} ms")

    print(f"\nSpeed-ups (R time / Python time):")
    print(f"  preprocess+PCA:  {r_pre/py_pre:5.2f}x")
    print(f"  pANN kernel:     {r_pann/py_pann:5.2f}x")
    print(f"  end-to-end:      {(r_pre+r_pann)/(py_pre+py_pann):5.2f}x")


if __name__ == "__main__":
    main()
