# pydoubletfinder

A **pure-Python re-implementation of [DoubletFinder](https://github.com/chris-mcginnis-ucsf/DoubletFinder)** (McGinnis et al., *Cell Systems* 2019) for computational doublet detection in single-cell RNA-seq data.

- AnnData-native — drop-in for the scanpy ecosystem
- **No `rpy2`**, no R install — the full pN/pK sweep, bimodality coefficient, BCmvn, and pANN scoring are all implemented directly in NumPy/SciPy
- Same function surface as the R workflow (`paramSweep` → `summarizeSweep` → `find.pK` → `doubletFinder`)
- Same seed, same result as R DoubletFinder 2.0.6: R's random draws and Seurat 5's preprocessing are reproduced, so the artificial doublets, chosen pK, pANN and calls match (see [Relationship to R DoubletFinder](#relationship-to-r-doubletfinder))

> omicverse uses this package for `ov.pp.qc(doublets_method='doubletfinder')`.

## Install

```bash
pip install pydoubletfinder
```

## Quick-start (class API)

```python
import anndata as ad
from pydoubletfinder import DoubletFinder

adata = ad.read_h5ad("mydata.h5ad")          # cells × genes, raw counts in .X

df = DoubletFinder(adata)

# 1) pN/pK parameter sweep
df.param_sweep(PCs=10)

# 2) Bimodality coefficient summary
df.summarize_sweep()

# 3) Optimal pK via BCmvn
bcmvn = df.find_pK()

# 4) Final scoring + classification
df.run(pN=0.25, nExp=round(0.075 * adata.n_obs))

adata.obs[[c for c in adata.obs.columns if c.startswith("DF.")]]
```

## Low-level functional API (mirrors R one-to-one)

```python
from pydoubletfinder import (
    param_sweep, summarize_sweep, find_pK,
    doublet_finder, model_homotypic,
    bimodality_coefficient,
)

# Per-real-cell pANN (needs a PCA embedding of [real + artificial] cells)
result = doublet_finder(
    pca_coord=my_pca,              # (n_real + n_doublets, n_PCs)
    n_real_cells=n_real,
    pN=0.25, pK=0.09, nExp=250,
)
result.pANN                          # np.ndarray
result.classifications               # {"Singlet", "Doublet"} per real cell
result.column_name_DF                # "DF.classifications_0.25_0.09_250"

# Homotypic-doublet proportion (match R modelHomotypic)
homotypic = model_homotypic(adata.obs["cluster"])
```

## What's included

| Python | R counterpart | Purpose |
|---|---|---|
| `DoubletFinder` class | — | AnnData-native lifecycle wrapper (like `Milo`, `Monocle`) |
| `param_sweep` | `paramSweep` | pN/pK sweep, one `SweepEntry` per (pN, pK) |
| `summarize_sweep` | `summarizeSweep` | bimodality coefficient per sweep entry, optional AUC |
| `find_pK` | `find.pK` | BCmvn + optimal-pK table |
| `doublet_finder` | `doubletFinder` | pANN + `Doublet`/`Singlet` classification |
| `model_homotypic` | `modelHomotypic` | homotypic-doublet proportion from cluster freqs |
| `bimodality_coefficient`, `skewness`, `kurtosis` | same | exported for direct use/testing |
| `bkde`, `approxfun` | `KernSmooth::bkde`, `stats::approxfun` | KernSmooth-compatible KDE + R approxfun |
| `sample_artificial_doublets` | internal | expose doublet-pair sampling for reproducibility |

## Relationship to R DoubletFinder

Since 0.2.1 the class API follows DoubletFinder 2.0.6 on Seurat 5 step by step:

- **Preprocessing of every merged real + artificial matrix** as Seurat's defaults:
  LogNormalize, `FindVariableFeatures(vst)` (including R's `loess` with its interpolated surface),
  `ScaleData` (clipped above at 10) and the PCA cell embeddings (`pydoubletfinder.seurat`).
- **Random draws** from R's generator (`pydoubletfinder.rrng`): Mersenne-Twister seeded as
  `set.seed`, `sample()` by rejection sampling. Seurat's `RunPCA` calls `set.seed(42)`, so in R
  the artificial doublets after the first pN of `paramSweep`, and those of `doubletFinder`, come
  from that state. `DoubletFinder(adata, random_state=None)` (default) starts from the same state;
  `random_state=s` plays the role of `set.seed(s)` before `paramSweep`.
- **Neighbour order** as R's `order()`: Euclidean distance, ties by cell index.

With the same seed, pydoubletfinder therefore draws the same artificial doublets as R and returns
the same pANN values and calls. The one step that is not reproduced bit for bit is the loess fit
of `FindVariableFeatures` (R's LINPACK QR differs from NumPy's SVD in the last bits): genes whose
standardised variances are equal up to rounding can swap at the 2,000-gene cut-off, which can move
a few pANN values by one neighbour.

`tests/test_r_reference.py` checks all of this against stored output of DoubletFinder 2.0.6 /
Seurat 5.5.1 (no R needed): R's random numbers and artificial-doublet parents, the loess fit
(< 1e-12), the variable genes, the chosen pK, and pANN and calls of the full workflow.

## Use in omicverse

pydoubletfinder is one of the doublet detectors of omicverse quality control:
`ov.pp.qc(adata, doublets_method='doubletfinder')` runs it on the raw counts, once per sample when
`batch_key` is given.

## Citation

If you use this package, please cite the original DoubletFinder paper:

> McGinnis, C.S., Murrow, L.M. & Gartner, Z.J. **DoubletFinder: Doublet Detection in Single-Cell RNA Sequencing Data Using Artificial Nearest Neighbors.** *Cell Systems* 8, 329–337 (2019).

and acknowledge omicverse / this repo for the Python port.

## License

CC0 — matches the upstream R package.
