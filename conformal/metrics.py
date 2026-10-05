"""Diagnostics. Marginal coverage alone is easy to game; look at the conditional views too."""

from __future__ import annotations

import numpy as np


def interval_coverage(y, lo, hi) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def set_coverage(sets, y_idx) -> float:
    sets = np.asarray(sets, dtype=bool)
    return float(np.mean(sets[np.arange(len(y_idx)), y_idx]))


def mean_set_size(sets) -> float:
    return float(np.asarray(sets, dtype=bool).sum(axis=1).mean())


def binned_coverage(feature, covered, n_bins: int = 5):
    """Coverage within quantile bins of a feature: exposes conditional undercoverage.

    Returns a list of (bin_low, bin_high, coverage, count).
    """
    f = np.asarray(feature, dtype=float)
    c = np.asarray(covered, dtype=float)
    edges = np.quantile(f, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = which == b
        out.append((float(edges[b]), float(edges[b + 1]),
                    float(c[m].mean()) if m.any() else float("nan"), int(m.sum())))
    return out


def size_stratified_coverage(sets, y_idx):
    """Coverage grouped by set size: {size: (coverage, count)}."""
    sets = np.asarray(sets, dtype=bool)
    sizes = sets.sum(axis=1)
    hit = sets[np.arange(len(y_idx)), y_idx]
    return {int(s): (float(hit[sizes == s].mean()), int((sizes == s).sum()))
            for s in np.unique(sizes)}


def worst_bin_coverage(feature, covered, n_bins: int = 5) -> float:
    return min(c for _, _, c, n in binned_coverage(feature, covered, n_bins) if n > 0)
