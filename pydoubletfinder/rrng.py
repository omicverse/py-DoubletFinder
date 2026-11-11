"""R's default random number generator, so that a seed gives the same draws as in R.

R (>= 3.6) defaults: Mersenne-Twister, normals by inversion, ``sample()`` by rejection sampling.
``set_seed`` follows ``RNG_Init`` (RNG.c): the seed is scrambled 50 times with the linear
congruential step ``seed = 69069 * seed + 1`` and the next 625 values fill the generator state
(the first is the position, reset to 624). ``unif_rand`` is MT19937 scaled by 2^-32;
``unif_index`` is ``R_unif_index``. Only the number of draws of ``rnorm`` matters here (two
uniforms each), because DoubletFinder uses normals only through Seurat's ``RunPCA``.
"""
from __future__ import annotations

import math

import numpy as np

_I2_32M1 = 2.328306437080797e-10      # 1 / (2^32 - 1), RNG.c fixup()
_SEURAT_PCA_SEED = 42                 # RunPCA(seed.use = 42)


class RRandom:
    """A stream of R's ``runif``/``sample`` draws (Mersenne-Twister, rejection sampling)."""

    def __init__(self, seed: int | None = None):
        self._bits = np.random.MT19937()
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int) -> "RRandom":
        """R ``set.seed(seed)``."""
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        key = np.empty(625, dtype=np.uint32)
        for j in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            key[j] = s
        self._bits.state = {"bit_generator": "MT19937", "state": {"key": key[1:], "pos": 624}}
        return self

    def unif_rand(self, n: int) -> np.ndarray:
        """``n`` draws of R ``unif_rand()``."""
        y = self._bits.random_raw(n).astype(np.float64) * 2.3283064365386963e-10
        y[y <= 0.0] = 0.5 * _I2_32M1
        y[1.0 - y <= 0.0] = 1.0 - 0.5 * _I2_32M1
        return y

    def unif_index(self, dn: int) -> int:
        """R ``R_unif_index(dn)``: an integer in [0, dn) by rejection from ``bits`` random bits."""
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        mask = (1 << bits) - 1
        chunks = bits // 16 + 1
        while True:
            u = np.floor(self.unif_rand(chunks) * 65536).astype(np.int64)
            v = 0
            for c in u:
                v = 65536 * v + int(c)
            v &= mask
            if v < dn:
                return v

    def _sample_replace(self, n: int, size: int) -> np.ndarray:
        """``size`` draws of ``R_unif_index(n)``, vectorised: draw a batch of attempts, keep the
        first ``size`` accepted ones and advance the stream by exactly the uniforms they used."""
        if n <= 1:
            self._bits.random_raw(size)
            return np.zeros(size, dtype=np.int64)
        bits = int(math.ceil(math.log2(n)))
        chunks = bits // 16 + 1
        mask = (1 << bits) - 1
        state = self._bits.state
        attempts = size + size // 2 + 64
        while True:
            u = np.floor(self.unif_rand(attempts * chunks) * 65536).astype(np.int64).reshape(attempts, chunks)
            v = np.zeros(attempts, dtype=np.int64)
            for c in range(chunks):
                v = 65536 * v + u[:, c]
            v &= mask
            ok = np.flatnonzero(v < n)
            if ok.size >= size:
                break
            attempts *= 2
            self._bits.state = state
        self._bits.state = state
        self._bits.random_raw((ok[size - 1] + 1) * chunks)
        return v[ok[:size]]

    def sample(self, n: int, size: int, replace: bool = False) -> np.ndarray:
        """R ``sample.int(n, size, replace)`` as 0-based indices (no hashing, n <= 1e7)."""
        if replace or size < 2:
            return self._sample_replace(n, size)
        pool = np.arange(n)
        out = np.empty(size, dtype=np.int64)
        left = n
        for i in range(size):
            j = self.unif_index(left)
            out[i] = pool[j]
            left -= 1
            pool[j] = pool[left]
        return out

    def skip_normals(self, n: int) -> "RRandom":
        """Advance the stream past ``rnorm(n)`` (inversion uses two uniforms per normal)."""
        self._bits.random_raw(2 * n)
        return self

    def after_seurat_pca(self, n_features: int) -> "RRandom":
        """The state Seurat's ``RunPCA`` leaves behind: ``set.seed(42)``, then irlba's ``rnorm``."""
        return self.set_seed(_SEURAT_PCA_SEED).skip_normals(n_features)
