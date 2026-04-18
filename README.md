# doubletfinder-py

A **pure-Python re-implementation of [DoubletFinder](https://github.com/chris-mcginnis-ucsf/DoubletFinder)** (McGinnis et al., *Cell Systems* 2019) for computational doublet detection in single-cell RNA-seq data.

- AnnData-native — drop-in for the scanpy ecosystem
- **No `rpy2`**, no R install — the full pN/pK sweep, bimodality coefficient, BCmvn, and pANN scoring are all implemented directly in NumPy/SciPy
- Same function surface as the R workflow (`paramSweep` → `summarizeSweep` → `find.pK` → `doubletFinder`)
- Bit-for-bit reproducibility against the R reference when fed matching PCA embeddings + artificial-doublet cell pairs (see `tests/test_exact_match.py`)

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse) (`omicverse.single.DoubletFinder`). All algorithmic work is developed upstream in omicverse and synced here for users who want DoubletFinder without the full omicverse stack.

## Install

```bash
pip install doubletfinder-py
```

## Quick-start (class API)

```python
import anndata as ad
from doubletfinder_py import DoubletFinder

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
from doubletfinder_py import (
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

## Reproducing R results exactly

The pipeline's randomness has two sources: which cell pairs become artificial doublets, and the PCA embedding of the merged matrix. To get identical outputs to an R run, provide both directly:

```python
from doubletfinder_py import doublet_finder

result = doublet_finder(
    pca_coord=r_pca_embedding,      # from Seurat's reductions$pca@cell.embeddings
    n_real_cells=len(real_cells),
    pN=0.25, pK=0.09, nExp=250,
)
```

`tests/test_exact_match.py` runs the R reference (`DoubletFinder::paramSweep` + `doubletFinder`) inside the `CMAP` conda env, saves PCA coords and cell-pair indices, and checks that the Python port reproduces the pANN, BCreal, and classification vectors bit-for-bit.

## Relationship to omicverse

Developed **upstream** in [`omicverse`](https://github.com/Starlitnightly/omicverse):

- Canonical implementation: `omicverse.single.DoubletFinder`
- Standalone mirror (this repo): same code, same API, minus the omicverse packaging

## Citation

If you use this package, please cite the original DoubletFinder paper:

> McGinnis, C.S., Murrow, L.M. & Gartner, Z.J. **DoubletFinder: Doublet Detection in Single-Cell RNA Sequencing Data Using Artificial Nearest Neighbors.** *Cell Systems* 8, 329–337 (2019).

and acknowledge omicverse / this repo for the Python port.

## License

CC0 — matches the upstream R package.
