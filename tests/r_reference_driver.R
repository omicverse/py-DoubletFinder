#!/usr/bin/env Rscript
# Drives the original R DoubletFinder on a tiny deterministic dataset and
# saves every numeric intermediate the Python port needs to verify parity.
#
# Output is a plain directory of TSVs/JSONs so there's no dependency on
# rpy2/feather/RData on the Python side.
#
# Usage (in an env with Seurat + DoubletFinder installed):
#
#   Rscript r_reference_driver.R <outdir>

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(DoubletFinder)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[[1]] else "r_ref_out"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

set.seed(123)

# --- Use Seurat's built-in pbmc_small to keep everything deterministic ---
data("pbmc_small", package = "SeuratObject")
seu <- UpdateSeuratObject(pbmc_small)
# Re-run normalization/PCA pipeline so @commands is populated for paramSweep
seu <- NormalizeData(seu, verbose = FALSE)
seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
seu <- ScaleData(seu, verbose = FALSE)
seu <- RunPCA(seu, npcs = 10, verbose = FALSE)

# Save the raw counts so Python consumes the exact same matrix
counts <- as.matrix(LayerData(seu, assay = "RNA", layer = "counts"))
write.table(
  counts,
  file = file.path(outdir, "counts.tsv"),
  sep = "\t", quote = FALSE, col.names = NA
)

# ---- run doubletFinder once with a fixed (pN, pK, nExp) ------------------
pN <- 0.25
pK <- 0.09
nExp <- round(0.05 * ncol(seu))
PCs <- 1:5

# Re-seed right before the stochastic doublet sampling so we can
# reproduce real.cells1 / real.cells2 on the Python side.
set.seed(42)
real_cells <- colnames(seu)
n_real <- length(real_cells)
n_doublets <- round(n_real / (1 - pN) - n_real)

# This mirrors the exact sampling pattern R does inside doubletFinder()
rc1 <- sample(real_cells, n_doublets, replace = TRUE)
rc2 <- sample(real_cells, n_doublets, replace = TRUE)
rc1_idx <- match(rc1, real_cells) - 1L  # 0-based for Python
rc2_idx <- match(rc2, real_cells) - 1L

# Build the merged data exactly as doubletFinder would.
doublets <- (counts[, rc1] + counts[, rc2]) / 2
colnames(doublets) <- paste0("X", seq_len(n_doublets))
data_wdoublets <- cbind(counts, doublets)

# Preprocess the merged matrix through the same Seurat pipeline that
# doubletFinder internally runs.
seu_w <- CreateSeuratObject(counts = data_wdoublets)
seu_w <- NormalizeData(seu_w, verbose = FALSE)
seu_w <- FindVariableFeatures(seu_w, selection.method = "vst",
                              nfeatures = 2000, verbose = FALSE)
seu_w <- ScaleData(seu_w, verbose = FALSE)
seu_w <- RunPCA(seu_w, npcs = length(PCs), verbose = FALSE)
pca_coord <- seu_w@reductions$pca@cell.embeddings[, PCs]

# Save the PCA coords that Python must feed into doublet_finder()
write.table(
  pca_coord, file = file.path(outdir, "pca_coord.tsv"),
  sep = "\t", quote = FALSE, col.names = NA
)

# Now compute pANN by hand, exactly the way doubletFinder does -------------
dist.mat <- fields::rdist(pca_coord)
nCells <- nrow(pca_coord)
k <- round(nCells * pK)
pANN <- numeric(n_real)
for (i in seq_len(n_real)) {
  neighbors <- order(dist.mat[, i])[2:(k + 1)]
  pANN[i] <- length(which(neighbors > n_real)) / k
}
classifications <- rep("Singlet", n_real)
classifications[order(pANN, decreasing = TRUE)[1:nExp]] <- "Doublet"

write.table(
  data.frame(cell = real_cells, pANN = pANN, DF = classifications),
  file = file.path(outdir, "df_result.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)

# ---- bimodality coefficient on a synthetic pANN-like vector --------------
set.seed(11)
pann_like <- c(rbeta(60, 2, 10), rbeta(20, 10, 2))   # clearly bimodal
gkde <- approxfun(KernSmooth::bkde(pann_like, kernel = "normal"))
xs <- seq(min(pann_like), max(pann_like), length.out = length(pann_like))
kde_y <- gkde(xs)
kde_y <- kde_y[is.finite(kde_y)]
bc <- DoubletFinder:::bimodality_coefficient(kde_y)
write.table(
  data.frame(pann = pann_like),
  file = file.path(outdir, "bc_input.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)
writeLines(as.character(bc), file.path(outdir, "bc_expected.txt"))

# ---- run paramSweep end-to-end and save sweep.stats / bcmvn --------------
set.seed(7)
sweep.list <- paramSweep(seu, PCs = PCs, sct = FALSE, num.cores = 1)
sweep.stats <- summarizeSweep(sweep.list, GT = FALSE)
bcmvn <- find.pK(sweep.stats)
write.table(
  sweep.stats, file = file.path(outdir, "sweep_stats.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)
write.table(
  bcmvn, file = file.path(outdir, "bcmvn.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)

# Meta: pN/pK/nExp used, cell index pairs
meta <- list(
  pN = pN, pK = pK, nExp = nExp, PCs = PCs,
  n_real = n_real, n_doublets = n_doublets,
  real_cells = real_cells,
  rc1_idx = rc1_idx, rc2_idx = rc2_idx
)
write(toJSON(meta, auto_unbox = TRUE), file = file.path(outdir, "meta.json"))

cat(sprintf("[r_reference_driver] wrote results to %s\n", outdir))
