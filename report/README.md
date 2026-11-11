# Benchmark: pydoubletfinder vs R DoubletFinder 2.0.6

Real-data comparison on the 16 datasets of the Xi & Li (2021) doublet benchmark
(Zenodo 10.5281/zenodo.4562782, `real_datasets.zip`), seeds 1 and 2 (`set.seed` before `paramSweep`).

```
Rscript scripts/01_prepare.R <real_datasets_dir> <tasks_dir>
python  scripts/run_all.py <tasks_dir> <runs_dir> <Rscript> <python> 12 1,2   # one core per run
python  scripts/05_figures.py <runs_dir> <tasks_dir> figures
```

Both run the DoubletFinder README workflow: Seurat preprocessing, `paramSweep(PCs = 1:10)`,
`summarizeSweep`, `find.pK` (pK at the BCmvn maximum), `doubletFinder(pN = 0.25, nExp = 7.5%)`.
R: DoubletFinder 2.0.6, Seurat 5.5.1, R 4.5.3 (bioconda). `figures/comparison_per_dataset.tsv` holds
the per-dataset numbers.
