"""Bimodality coefficient and its moment-based ingredients.

Ported verbatim from the R implementation in
``DoubletFinder/R/{skewness,kurtosis,bimodality_coefficient}.R``
which in turn cite the ``modes`` R package (v0.7). Formulas reproduced
exactly so BCreal values match R to machine precision.
"""
from __future__ import annotations

import numpy as np


def skewness(x: np.ndarray) -> float:
    """Sample skewness using the R-``modes``-package formula.

    Matches R:
        S <- (1/n) * sum((x-mean(x))^3) / ((1/n) * sum((x-mean(x))^2))^1.5
        S <- S * sqrt(n*(n-1)) / (n-2)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 3:
        return np.nan
    mu = x.mean()
    d = x - mu
    m2 = np.mean(d * d)
    m3 = np.mean(d * d * d)
    if m2 == 0.0:
        return np.nan
    s = m3 / (m2 ** 1.5)
    s *= np.sqrt(n * (n - 1.0)) / (n - 2.0)
    return float(s)


def kurtosis(x: np.ndarray) -> float:
    """Sample kurtosis using the R-``modes``-package formula.

    Matches R:
        K <- (1/n)*sum((x-mean(x))^4) / ((1/n)*sum((x-mean(x))^2))^2 - 3
        K <- ((n - 1)*((n+1)*K - 3*(n-1)) / ((n-2)*(n-3))) + 3
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        return np.nan
    mu = x.mean()
    d = x - mu
    m2 = np.mean(d * d)
    m4 = np.mean(d * d * d * d)
    if m2 == 0.0:
        return np.nan
    k = m4 / (m2 * m2) - 3.0
    k = ((n - 1.0) * ((n + 1.0) * k - 3.0 * (n - 1.0)) / ((n - 2.0) * (n - 3.0))) + 3.0
    return float(k)


def bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient.

    Matches R: B <- (G^2 + 1) / (K + 3*(n-1)^2 / ((n-2)*(n-3)))
    with G = skewness(x), K = kurtosis(x), n = length(x).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        return np.nan
    g = skewness(x)
    k = kurtosis(x)
    denom = k + (3.0 * (n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0))
    if denom == 0.0:
        return np.nan
    return float((g * g + 1.0) / denom)
