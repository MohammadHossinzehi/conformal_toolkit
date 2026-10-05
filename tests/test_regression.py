import math

import numpy as np
import pytest

from conformal import (ConformalizedQuantileRegressor, JackknifePlusRegressor,
                       QuantileRegressor, RidgeRegressor, SplitConformalRegressor,
                       conformal_quantile, interval_coverage, mean_width)
from conformal.datasets import heteroscedastic

ALPHA = 0.1


def test_conformal_quantile_index():
    s = np.arange(1, 10)  # n = 9, k = ceil(10 * 0.9) = 9
    assert conformal_quantile(s, 0.1) == 9
    assert conformal_quantile(s, 0.5) == 5  # ceil(10 * 0.5) = 5
    assert math.isinf(conformal_quantile(np.arange(5), 0.1))  # ceil(6*0.9)=6 > 5
    with pytest.raises(ValueError):
        conformal_quantile(s, 1.0)


def _avg_coverage(make, trials=60, n=600, n_test=500):
    covs, widths = [], []
    for t in range(trials):
        X, y, _ = heteroscedastic(n, rng=t)
        Xt, yt, _ = heteroscedastic(n_test, rng=10_000 + t)
        lo, hi = make(t).fit(X, y).predict_interval(Xt)
        covs.append(interval_coverage(yt, lo, hi))
        widths.append(mean_width(lo, hi))
    return float(np.mean(covs)), float(np.mean(widths))


def test_split_conformal_marginal_coverage():
    cov, _ = _avg_coverage(lambda t: SplitConformalRegressor(
        RidgeRegressor(degree=5), alpha=ALPHA, random_state=t))
    # theory: 1-alpha <= E[cov] <= 1-alpha + 1/(n_cal+1)
    assert 0.885 <= cov <= 0.92


def test_normalized_scores_adapt_to_noise():
    X, y, _ = heteroscedastic(3000, rng=1)
    Xt = np.array([[0.0], [2.8]])
    m = SplitConformalRegressor(RidgeRegressor(degree=5), score="normalized",
                                random_state=0).fit(X, y)
    lo, hi = m.predict_interval(Xt)
    width = hi - lo
    assert width[1] > 3 * width[0]  # noise is ~15x larger at |x|=2.8 than at 0
    cov, _ = _avg_coverage(lambda t: SplitConformalRegressor(
        RidgeRegressor(degree=5), score="normalized", random_state=t), trials=40)
    assert 0.88 <= cov <= 0.93


def test_cqr_coverage_and_adaptivity():
    cov, w = _avg_coverage(lambda t: ConformalizedQuantileRegressor(
        alpha=ALPHA, degree=5, random_state=t), trials=40)
    assert 0.88 <= cov <= 0.93
    _, w_abs = _avg_coverage(lambda t: SplitConformalRegressor(
        RidgeRegressor(degree=5), random_state=t), trials=40)
    assert w < w_abs  # adaptive bands are narrower on average under heteroscedasticity


def test_cqr_asymmetric_controls_each_tail():
    X, y, _ = heteroscedastic(4000, rng=3)
    Xt, yt, _ = heteroscedastic(20000, rng=4)
    m = ConformalizedQuantileRegressor(alpha=0.2, asymmetric=True, random_state=0).fit(X, y)
    lo, hi = m.predict_interval(Xt)
    assert abs(np.mean(yt < lo) - 0.1) < 0.025
    assert abs(np.mean(yt > hi) - 0.1) < 0.025


def test_cv_plus_coverage():
    cov, _ = _avg_coverage(lambda t: JackknifePlusRegressor(
        RidgeRegressor(degree=5), alpha=ALPHA, n_folds=10, random_state=t),
        trials=30, n=300)
    assert cov >= 1 - 2 * ALPHA
    assert 0.87 <= cov <= 0.95


def test_jackknife_plus_loo_runs_and_covers():
    X, y, _ = heteroscedastic(120, rng=7)
    Xt, yt, _ = heteroscedastic(2000, rng=8)
    m = JackknifePlusRegressor(RidgeRegressor(degree=3), n_folds=None).fit(X, y)
    assert len(m.models_) == 120
    lo, hi = m.predict_interval(Xt)
    assert np.all(lo <= hi)
    assert interval_coverage(yt, lo, hi) > 0.8


def test_smaller_alpha_gives_wider_intervals():
    X, y, _ = heteroscedastic(1000, rng=2)
    m = SplitConformalRegressor(RidgeRegressor(degree=5), random_state=0).fit(X, y)
    w = [mean_width(*m.predict_interval(X[:50], alpha=a)) for a in (0.3, 0.1, 0.02)]
    assert w[0] < w[1] < w[2]


def test_quantile_regressor_hits_target_quantile():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (4000, 1))
    y = 2 * X[:, 0] + rng.standard_normal(4000)
    for tau in (0.1, 0.5, 0.9):
        q = QuantileRegressor(tau).fit(X, y)
        assert abs(np.mean(y <= q.predict(X)) - tau) < 0.02


def test_ridge_recovers_linear_coefficients():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 3.0
    p = RidgeRegressor(lam=1e-8).fit(X, y).predict(X)
    assert np.allclose(p, y, atol=1e-4)
