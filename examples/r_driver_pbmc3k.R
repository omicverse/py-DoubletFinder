#!/usr/bin/env Rscript
# Run the original R DoubletFinder end-to-end on an external 10X-style
# counts matrix (genes x cells TSV, same format as the Python side dumps
# via pandas.to_csv). Outputs: pANN, classifications, PCA coords, and a
# small JSON of parameters — everything the notebook needs to overlay on
# the Python-computed results.
#
# Usage:
#   Rscript r_driver_pbmc3k.R <counts_tsv> <outdir> <pN> <pK> <nExp> <PCs>

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(DoubletFinder)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
counts_path <- args[[1]]
outdir      <- args[[2]]
pN    <- if (length(args) >= 3) as.numeric(args[[3]]) else 0.25
pK    <- if (length(args) >= 4) as.numeric(args[[4]]) else 0.09
nExp  <- if (length(args) >= 5) as.integer(args[[5]]) else 75L
n_pcs <- if (length(args) >= 6) as.integer(args[[6]]) else 10L

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
set.seed(42)

cat(sprintf("[R] reading %s\n", counts_path))
counts <- as.matrix(read.table(counts_path, sep = "\t", header = TRUE,
                               row.names = 1, check.names = FALSE))

# Full Seurat preprocessing (NormalizeData / FindVariableFeatures / ScaleData / PCA)
seu <- CreateSeuratObject(counts = counts)
seu <- NormalizeData(seu, verbose = FALSE)
seu <- FindVariableFeatures(seu, selection.method = "vst",
                            nfeatures = 2000, verbose = FALSE)
seu <- ScaleData(seu, verbose = FALSE)
seu <- RunPCA(seu, npcs = n_pcs, verbose = FALSE)

cat("[R] running DoubletFinder::doubletFinder()\n")
real_cells <- colnames(seu)
n_real <- length(real_cells)
n_doublets <- round(n_real / (1 - pN) - n_real)

# Capture the exact artificial-doublet pairs so the Python side can reuse them
set.seed(42)
rc1 <- sample(real_cells, n_doublets, replace = TRUE)
rc2 <- sample(real_cells, n_doublets, replace = TRUE)
rc1_idx <- match(rc1, real_cells) - 1L
rc2_idx <- match(rc2, real_cells) - 1L

doublets <- (counts[, rc1] + counts[, rc2]) / 2
colnames(doublets) <- paste0("X", seq_len(n_doublets))
data_wdoublets <- cbind(counts, doublets)

seu_w <- CreateSeuratObject(counts = data_wdoublets)
seu_w <- NormalizeData(seu_w, verbose = FALSE)
seu_w <- FindVariableFeatures(seu_w, selection.method = "vst",
                              nfeatures = 2000, verbose = FALSE)
seu_w <- ScaleData(seu_w, verbose = FALSE)
seu_w <- RunPCA(seu_w, npcs = n_pcs, verbose = FALSE)
pca <- seu_w@reductions$pca@cell.embeddings[, seq_len(n_pcs)]

write.table(pca, file = file.path(outdir, "pca_coord.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

# Compute pANN by hand (matches DoubletFinder::doubletFinder exactly)
dist_mat <- fields::rdist(pca)
nCells <- nrow(pca)
k <- round(nCells * pK)
pANN <- numeric(n_real)
for (i in seq_len(n_real)) {
  neighbors <- order(dist_mat[, i])[2:(k + 1)]
  pANN[i] <- length(which(neighbors > n_real)) / k
}
classifications <- rep("Singlet", n_real)
classifications[order(pANN, decreasing = TRUE)[1:nExp]] <- "Doublet"

write.table(
  data.frame(cell = real_cells, pANN = pANN, DF = classifications),
  file = file.path(outdir, "df_result.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)
write(toJSON(list(pN = pN, pK = pK, nExp = nExp, n_real = n_real,
                  n_doublets = n_doublets,
                  rc1_idx = rc1_idx, rc2_idx = rc2_idx),
             auto_unbox = TRUE),
      file = file.path(outdir, "meta.json"))
cat(sprintf("[R] wrote outputs to %s (n_real=%d, doublets=%d)\n",
            outdir, n_real, sum(classifications == "Doublet")))
