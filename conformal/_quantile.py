"""The one function every conformal method in this package leans on."""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores, alpha: float) -> float:
    """Finite sample corrected (1 - alpha) quantile of calibration scores.

    With n exchangeable calibration scores, the test score is below the
    k-th smallest calibration score with probability at least 1 - alpha
    when k = ceil((n + 1) * (1 - alpha)). If k exceeds n there are not
    enough calibration points to certify that level and the only valid
    answer is +inf (an infinitely wide interval / every label).
    """
    s = np.asarray(scores, dtype=float).ravel()
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    n = s.size
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return float(np.partition(s, k - 1)[k - 1])


def lower_order_stat(values, alpha: float, axis: int = -1):
    """floor(alpha (n+1))-th smallest along ``axis``; -inf when that index is 0.

    Used by jackknife+ / CV+ for the lower interval endpoint.
    """
    v = np.asarray(values, dtype=float)
    n = v.shape[axis]
    k = math.floor(alpha * (n + 1))
    if k < 1:
        shape = list(v.shape)
        del shape[axis]
        return np.full(shape, -math.inf)
    return np.take(np.partition(v, k - 1, axis=axis), k - 1, axis=axis)


def upper_order_stat(values, alpha: float, axis: int = -1):
    """ceil((1-alpha)(n+1))-th smallest along ``axis``; +inf when it exceeds n."""
    v = np.asarray(values, dtype=float)
    n = v.shape[axis]
    k = math.ceil((1.0 - alpha) * (n + 1))
    if k > n:
        shape = list(v.shape)
        del shape[axis]
        return np.full(shape, math.inf)
    return np.take(np.partition(v, k - 1, axis=axis), k - 1, axis=axis)
