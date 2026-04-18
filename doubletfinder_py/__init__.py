"""
doubletfinder_py: Pure-Python DoubletFinder for scRNA-seq doublet detection.

A standalone mirror of the canonical implementation that lives in
``omicverse.single.DoubletFinder``. This repo exists for users who want
DoubletFinder without pulling in the full omicverse stack — all
algorithmic work is developed upstream in omicverse and synced here.

Input: AnnData objects (cells x genes). All analysis state is written to
``adata.obs`` and ``adata.uns['doubletfinder']`` so the annotated object
stays compatible with the scanpy ecosystem.

Quick-start
-----------
>>> from doubletfinder_py import DoubletFinder
>>> df = DoubletFinder(adata)
>>> df.param_sweep(PCs=10).summarize_sweep().find_pK()
>>> df.run(pN=0.25, nExp=250)

Low-level functional API (matches R one-to-one)
-----------------------------------------------
>>> from doubletfinder_py import (
...     param_sweep, summarize_sweep, find_pK, doublet_finder,
...     model_homotypic, bimodality_coefficient,
... )
"""
from __future__ import annotations

from .bimodality import (
    bimodality_coefficient,
    kurtosis,
    skewness,
)
from .core import (
    DEFAULT_PK_GRID,
    DEFAULT_PN_GRID,
    DoubletFinderResult,
    SweepEntry,
    doublet_finder,
    find_pK,
    model_homotypic,
    param_sweep,
    sample_artificial_doublets,
    summarize_sweep,
)
from .doubletfinder import DoubletFinder
from .kde import BKDEResult, approxfun, bkde

__version__ = "0.1.0"

__all__ = [
    # class API
    "DoubletFinder",
    # functional API
    "param_sweep",
    "summarize_sweep",
    "find_pK",
    "doublet_finder",
    "model_homotypic",
    "sample_artificial_doublets",
    # statistics
    "bimodality_coefficient",
    "kurtosis",
    "skewness",
    # KDE helpers (exposed for testing/parity with R)
    "bkde",
    "approxfun",
    "BKDEResult",
    # data-class outputs
    "DoubletFinderResult",
    "SweepEntry",
    # constants
    "DEFAULT_PK_GRID",
    "DEFAULT_PN_GRID",
]
