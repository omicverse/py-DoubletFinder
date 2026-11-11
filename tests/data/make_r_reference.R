# Reference values for tests/test_r_reference.py: DoubletFinder 2.0.6 / Seurat 5.5.1 / R 4.5.3.
# Fixture: J293t-dm of the Xi & Li (2021) doublet benchmark (Zenodo 4562782), 500 cells x 3,000 genes.
# Usage: Rscript make_r_reference.R <J293t-dm task dir>   (writes into the working directory)
suppressMessages({library(Matrix); library(Seurat); library(DoubletFinder)})
td <- commandArgs(TRUE)[1]
hex <- function(x) sprintf("%a", x)
m <- as(readMM(gzfile(file.path(td, "counts.mtx.gz"))), "CsparseMatrix")
rownames(m) <- readLines(file.path(td, "genes.tsv")); colnames(m) <- readLines(file.path(td, "cells.tsv"))
set.seed(42); keep <- sort(sample(which(Matrix::rowSums(m) > 0), 3000)); m <- m[keep, ]
writeMM(m, "df_counts.mtx"); system2("gzip", c("-f", "df_counts.mtx"))
writeLines(readLines(file.path(td, "labels.tsv")), "df_labels.tsv")

# R's random numbers
set.seed(1); u <- runif(5); s1 <- sample(500, 10, replace = TRUE); s2 <- sample(20000, 10)
set.seed(42); invisible(rnorm(2000)); u42 <- runif(3)
writeLines(c(paste(hex(u), collapse = " "), paste(s1, collapse = " "), paste(s2, collapse = " "),
             paste(hex(u42), collapse = " ")), "df_rng.txt")

# One merged matrix (pN = 0.25, set.seed(1)): parents, loess fit, variable genes
real <- colnames(m); nd <- round(500 / 0.75 - 500)
set.seed(1); r1 <- sample(real, nd, replace = TRUE); r2 <- sample(real, nd, replace = TRUE)
dat <- cbind(m, (m[, r1] + m[, r2]) / 2); colnames(dat) <- c(real, paste0("X", 1:nd))
writeLines(c(paste(match(r1, real) - 1L, collapse = " "), paste(match(r2, real) - 1L, collapse = " ")), "df_parents.txt")
mu <- Matrix::rowMeans(dat); v <- Seurat:::SparseRowVar2(dat, mu, FALSE); nc <- v > 0
fit <- loess(log10(v[nc]) ~ log10(mu[nc]), span = 0.3)$fitted
writeLines(hex(fit), "df_loess.txt")
sw <- FindVariableFeatures(NormalizeData(CreateSeuratObject(dat), verbose = FALSE), verbose = FALSE)
writeLines(as.character(match(VariableFeatures(sw), rownames(m)) - 1L), "df_hvg.txt")

# The README workflow, set.seed(1) before paramSweep
seu <- RunPCA(ScaleData(FindVariableFeatures(NormalizeData(CreateSeuratObject(m), verbose = FALSE), verbose = FALSE),
                        verbose = FALSE), verbose = FALSE)
set.seed(1)
invisible(capture.output({sweep <- paramSweep(seu, PCs = 1:10); bcmvn <- find.pK(summarizeSweep(sweep, GT = FALSE))}))
pK <- as.numeric(as.character(bcmvn$pK[which.max(bcmvn$BCmetric)]))
invisible(capture.output(seu <- doubletFinder(seu, PCs = 1:10, pN = 0.25, pK = pK, nExp = round(0.075 * 500))))
pc <- grep("^pANN_", colnames(seu@meta.data), value = TRUE); dc <- grep("^DF.class", colnames(seu@meta.data), value = TRUE)
write.table(data.frame(pK = as.character(bcmvn$pK), BCmetric = hex(bcmvn$BCmetric)), "df_bcmvn.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(pANN = hex(seu@meta.data[[pc]]), class = seu@meta.data[[dc]]), "df_result.tsv", sep = "\t",
            quote = FALSE, row.names = FALSE)
cat("pK", pK, "\n")
