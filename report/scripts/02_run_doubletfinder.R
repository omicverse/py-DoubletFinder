# Reference run: DoubletFinder 2.0.6 on Seurat 5, the workflow of the DoubletFinder README, on one task and seed.
# Usage: Rscript 02_run_doubletfinder.R <task_dir> <out_dir> <seed>
suppressMessages({library(Matrix); library(Seurat); library(DoubletFinder); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
td <- args[1]; od <- args[2]; seed <- as.integer(args[3])
dir.create(od, recursive = TRUE, showWarnings = FALSE)
m <- as(readMM(gzfile(file.path(td, "counts.mtx.gz"))), "CsparseMatrix")
rownames(m) <- readLines(file.path(td, "genes.tsv")); colnames(m) <- readLines(file.path(td, "cells.tsv"))
t0 <- proc.time()[["elapsed"]]
seu <- CreateSeuratObject(counts = m)
seu <- NormalizeData(seu, verbose = FALSE)
seu <- FindVariableFeatures(seu, verbose = FALSE)
seu <- ScaleData(seu, verbose = FALSE)
seu <- RunPCA(seu, verbose = FALSE)
set.seed(seed)
invisible(capture.output({
  sweep <- paramSweep(seu, PCs = 1:10, sct = FALSE)
  stats <- summarizeSweep(sweep, GT = FALSE)
  bcmvn <- find.pK(stats)
}))
pK <- as.numeric(as.character(bcmvn$pK[which.max(bcmvn$BCmetric)]))
nExp <- round(0.075 * ncol(seu))
invisible(capture.output(seu <- doubletFinder(seu, PCs = 1:10, pN = 0.25, pK = pK, nExp = nExp, sct = FALSE)))
elapsed <- proc.time()[["elapsed"]] - t0
pc <- grep("^pANN_", colnames(seu@meta.data), value = TRUE); dc <- grep("^DF.classifications_", colnames(seu@meta.data), value = TRUE)
write.table(data.frame(cell = colnames(seu), score = sprintf("%.17g", seu@meta.data[[pc]]), class = tolower(seu@meta.data[[dc]])),
            gzfile(file.path(od, "scores.tsv.gz")), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(pK = as.character(bcmvn$pK), BCmetric = sprintf("%.17g", bcmvn$BCmetric)), file.path(od, "bcmvn.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write_json(list(engine = "DoubletFinder", version = as.character(packageVersion("DoubletFinder")),
                Seurat = as.character(packageVersion("Seurat")), R = R.version.string, seed = seed, pK = pK, nExp = nExp,
                seconds = elapsed), file.path(od, "timing.json"), auto_unbox = TRUE, digits = NA)
cat(basename(td), seed, sprintf("pK=%g %.1fs", pK, elapsed), "\n")
