"""Compare every method on synthetic data with known ground truth.

    python examples/benchmark.py

Prints marginal coverage, average width / set size, and the worst
conditional coverage over feature bins, which is where methods differ.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conformal import (AdaptiveConformal, ConformalClassifier,  # noqa: E402
                       ConformalizedQuantileRegressor, JackknifePlusRegressor,
                       RidgeRegressor, SoftmaxClassifier, SplitConformalRegressor,
                       binned_coverage, interval_coverage, mean_set_size, mean_width,
                       set_coverage)
from conformal.datasets import gaussian_blobs, heteroscedastic, regime_shift_series  # noqa: E402

ALPHA = 0.1
TRIALS = 20


def regression():
    print(f"Regression, heteroscedastic noise, alpha={ALPHA}, {TRIALS} trials")
    print(f"{'method':<22}{'coverage':>10}{'width':>9}{'worst |x| bin':>15}")
    makers = {
        "split (absolute)": lambda t: SplitConformalRegressor(RidgeRegressor(degree=5), random_state=t),
        "split (normalized)": lambda t: SplitConformalRegressor(RidgeRegressor(degree=5),
                                                                score="normalized", random_state=t),
        "CQR": lambda t: ConformalizedQuantileRegressor(random_state=t),
        "CV+ (10 folds)": lambda t: JackknifePlusRegressor(RidgeRegressor(degree=5), random_state=t),
    }
    for name, make in makers.items():
        cov, wid, worst = [], [], []
        for t in range(TRIALS):
            X, y, _ = heteroscedastic(800, rng=t)
            Xt, yt, _ = heteroscedastic(2000, rng=1000 + t)
            lo, hi = make(t).fit(X, y).predict_interval(Xt)
            cov.append(interval_coverage(yt, lo, hi))
            wid.append(mean_width(lo, hi))
            bins = binned_coverage(np.abs(Xt[:, 0]), (yt >= lo) & (yt <= hi), n_bins=5)
            worst.append(min(c for *_, c, _n in bins))
        print(f"{name:<22}{np.mean(cov):>10.3f}{np.mean(wid):>9.3f}{np.mean(worst):>15.3f}")


def classification():
    print(f"\nClassification, 5 overlapping classes, alpha={ALPHA}, {TRIALS} trials")
    print(f"{'method':<22}{'coverage':>10}{'avg size':>10}")
    configs = {
        "LAC": dict(method="lac"),
        "APS (randomized)": dict(method="aps"),
        "RAPS lam=0.05": dict(method="raps", lam=0.05, k_reg=2),
        "LAC Mondrian": dict(method="lac", class_conditional=True),
    }
    for name, kw in configs.items():
        cov, size = [], []
        for t in range(TRIALS):
            X, y = gaussian_blobs(2000, rng=t)
            Xt, yt = gaussian_blobs(2000, rng=1000 + t)
            clf = ConformalClassifier(SoftmaxClassifier(), alpha=ALPHA, random_state=t, **kw).fit(X, y)
            s = clf.predict_set(Xt)
            cov.append(set_coverage(s, yt))
            size.append(mean_set_size(s))
        print(f"{name:<22}{np.mean(cov):>10.3f}{np.mean(size):>10.2f}")


def online():
    print(f"\nOnline, AR(1) with volatility regime shifts, alpha={ALPHA}")
    y, _ = regime_shift_series(8000, rng=0)
    s = np.abs(y - 0.8 * np.r_[0.0, y[:-1]])
    T = len(s)
    print(f"{'method':<26}{'overall miss':>13}{'miss in high vol':>18}")
    for name, aci in [("static rolling (gamma=0)", AdaptiveConformal(ALPHA, gamma=0.0)),
                      ("ACI gamma=0.02, window", AdaptiveConformal(ALPHA, gamma=0.02, window=300))]:
        aci.run(s)
        e = np.array(aci.errors)
        print(f"{name:<26}{e.mean():>13.3f}{e[T // 4 + 50:T // 2].mean():>18.3f}")


if __name__ == "__main__":
    regression()
    classification()
    online()
