"""KernSmooth::bkde-compatible binned kernel density estimate.

Port of the FORTRAN/R binned KDE used in R's ``KernSmooth::bkde``
(Wand & Jones 1995). Matches the default settings that DoubletFinder's
``summarizeSweep`` relies on:

    bkde(pANN, kernel = "normal")   # → approxfun → evaluated on n-point grid

Default bandwidth follows the Silverman "normal reference" rule exactly
as KernSmooth computes it, default grid size is 401, and the range
extends the data by ``tau*h`` on each side with ``tau=4`` for the
Gaussian kernel (``tau`` = effective truncation radius of the kernel).

Outputs match the R function to within floating-point rounding error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Kernel-specific truncation radius used in KernSmooth::bkde's default
# range.x. These are the values hard-coded in KernSmooth's `bkde.R`.
_KERNEL_TAU = {
    "normal": 4.0,
    "box": 1.0,
    "epanech": 1.0,
    "biweight": 1.0,
    "triweight": 1.0,
}

# Kernel-specific delta_0 (canonical bandwidth factor) from KernSmooth.
_KERNEL_DEL0 = {
    "normal": (1.0 / (4.0 * np.pi)) ** (1.0 / 10.0),
    "box": (9.0 / 2.0) ** (1.0 / 5.0),
    "epanech": 15.0 ** (1.0 / 5.0),
    "biweight": 35.0 ** (1.0 / 5.0),
    "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
}


@dataclass
class BKDEResult:
    """Mirror of R's ``bkde`` return value (a list with ``$x`` and ``$y``)."""

    x: np.ndarray
    y: np.ndarray


def _default_bandwidth(x: np.ndarray, kernel: str) -> float:
    """KernSmooth::bkde default bandwidth = del0 * (243/(35 n))^(1/5) * sd(x).

    Note: R uses ``var(x)`` (unbiased, denominator n-1) inside ``sqrt``;
    we match that with ``ddof=1``.
    """
    n = x.size
    sd = float(np.std(x, ddof=1))
    del0 = _KERNEL_DEL0[kernel]
    return del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * sd


def _linbin(x: np.ndarray, a: float, b: float, M: int, truncate: bool) -> np.ndarray:
    """Linear binning of ``x`` onto ``M`` equally-spaced grid points on [a, b].

    This replicates KernSmooth's internal ``linbin`` routine: each data
    point contributes a weight split between its two neighboring grid
    points in proportion to its fractional position. Points outside
    [a, b] are dropped when ``truncate`` is True.
    """
    gcounts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)
    # Index in the grid (0-based) as a continuous quantity
    lxi = (x - a) / delta
    li = np.floor(lxi).astype(np.int64)
    rem = lxi - li
    # Drop points outside the grid when truncating
    if truncate:
        inside = (li >= 0) & (li < M - 1)
        li = li[inside]
        rem = rem[inside]
    else:
        # Clamp to edges
        li = np.clip(li, 0, M - 2)
        rem = np.clip(rem, 0.0, 1.0)
    np.add.at(gcounts, li, 1.0 - rem)
    np.add.at(gcounts, li + 1, rem)
    return gcounts


def bkde(
    x: np.ndarray,
    kernel: str = "normal",
    canonical: bool = False,
    bandwidth: float | None = None,
    gridsize: int = 401,
    range_x: tuple[float, float] | None = None,
    truncate: bool = True,
) -> BKDEResult:
    """Binned kernel density estimate matching ``KernSmooth::bkde``.

    Parameters mirror the R function one-to-one.
    """
    if kernel not in _KERNEL_TAU:
        raise ValueError(f"kernel must be one of {list(_KERNEL_TAU)}, got {kernel!r}")
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 finite values for KDE")

    h = bandwidth if bandwidth is not None else _default_bandwidth(x, kernel)
    if canonical:
        # KernSmooth's "canonical" rescaling — not used by DoubletFinder.
        h = h / _KERNEL_DEL0[kernel]

    tau = _KERNEL_TAU[kernel]
    if range_x is None:
        a = float(np.min(x) - tau * h)
        b = float(np.max(x) + tau * h)
    else:
        a, b = float(range_x[0]), float(range_x[1])

    M = int(gridsize)
    delta = (b - a) / (M - 1)

    # 1) Linear bin the data
    gcounts = _linbin(x, a, b, M, truncate=truncate)

    # 2) Build the kernel weight vector on a symmetric grid of 2M samples.
    #    KernSmooth uses L = min(floor(tau*h/delta), M-1) evaluations of
    #    the scaled kernel, then zero-pads to length 2M for FFT.
    L = min(int(np.floor(tau * h / delta)), M - 1)
    lvec = np.arange(-L, L + 1)
    arg = lvec * delta / h
    if kernel == "normal":
        kweights = np.exp(-0.5 * arg * arg) / np.sqrt(2.0 * np.pi)
    elif kernel == "box":
        kweights = np.where(np.abs(arg) <= 1.0, 0.5, 0.0)
    elif kernel == "epanech":
        kweights = np.where(np.abs(arg) <= 1.0, 0.75 * (1.0 - arg * arg), 0.0)
    elif kernel == "biweight":
        kweights = np.where(
            np.abs(arg) <= 1.0, (15.0 / 16.0) * (1.0 - arg * arg) ** 2, 0.0
        )
    else:  # triweight
        kweights = np.where(
            np.abs(arg) <= 1.0, (35.0 / 32.0) * (1.0 - arg * arg) ** 3, 0.0
        )
    kweights = kweights / (n * h)

    # 3) Convolve gcounts with kweights via direct sum — matches R's FFT
    #    answer to machine precision and keeps the code dependency-free.
    y = np.zeros(M, dtype=np.float64)
    for j in range(-L, L + 1):
        w = kweights[j + L]
        if w == 0.0:
            continue
        if j >= 0:
            y[j:] += w * gcounts[: M - j]
        else:
            y[:M + j] += w * gcounts[-j:]

    grid = np.linspace(a, b, M)
    return BKDEResult(x=grid, y=y)


def approxfun(x: np.ndarray, y: np.ndarray, rule: int = 1) -> callable:
    """Mimic R's ``stats::approxfun`` for linear interpolation.

    Returns a callable that linearly interpolates (x, y). Outside
    [min(x), max(x)] returns NaN (``rule=1``) or the nearest endpoint
    (``rule=2``), matching R's behavior.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    x_min, x_max = xs[0], xs[-1]

    def _interp(u):
        u = np.asarray(u, dtype=np.float64)
        # np.interp returns y[0]/y[-1] outside the range by default
        out = np.interp(u, xs, ys)
        if rule == 1:
            mask = (u < x_min) | (u > x_max)
            out = np.where(mask, np.nan, out)
        return out

    return _interp
