"""Synthetic data with known structure, used by the tests and the demo."""

from __future__ import annotations

import numpy as np


def heteroscedastic(n: int, rng=None):
    """y = sin(2x) + 0.5x + noise whose scale grows with |x|. x ~ U(-3, 3)."""
    rng = np.random.default_rng(rng)
    x = rng.uniform(-3, 3, size=n)
    sigma = 0.1 + 0.5 * np.abs(x)
    y = np.sin(2 * x) + 0.5 * x + sigma * rng.standard_normal(n)
    return x[:, None], y, sigma


def gaussian_blobs(n: int, k: int = 5, spread: float = 1.6, rng=None):
    """k overlapping 2D Gaussian classes on a circle: some inputs are truly ambiguous."""
    rng = np.random.default_rng(rng)
    y = rng.integers(0, k, size=n)
    angles = 2 * np.pi * np.arange(k) / k
    centers = 3 * np.c_[np.cos(angles), np.sin(angles)]
    X = centers[y] + spread * rng.standard_normal((n, 2))
    return X, y


def regime_shift_series(T: int, rng=None):
    """AR(1) series whose noise volatility jumps between regimes (non exchangeable)."""
    rng = np.random.default_rng(rng)
    vol = np.ones(T)
    vol[T // 4: T // 2] = 4.0
    vol[3 * T // 4:] = 0.5
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.8 * y[t - 1] + vol[t] * rng.standard_normal()
    return y, vol
