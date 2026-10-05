"""Conformal prediction intervals for regression.

Three families, each with a different trade off:

* ``SplitConformalRegressor``: one model fit, one calibration pass. Cheapest.
  Optional "normalized" scores divide residuals by a learned difficulty
  estimate so intervals widen where the data is noisy.
* ``ConformalizedQuantileRegressor`` (CQR, Romano et al. 2019): calibrates a
  pair of quantile regressors, so the band is adaptive by construction.
* ``JackknifePlusRegressor`` (Barber et al. 2021): jackknife+ or CV+; no data
  is thrown away for calibration, at the cost of K model fits.
"""

from __future__ import annotations

import numpy as np

from ._quantile import conformal_quantile, lower_order_stat, upper_order_stat
from .models import QuantileRegressor, RidgeRegressor, clone


def _split(n, cal_fraction, rng):
    if not 0 < cal_fraction < 1:
        raise ValueError("cal_fraction must be in (0, 1)")
    perm = rng.permutation(n)
    n_cal = max(1, int(round(n * cal_fraction)))
    return perm[n_cal:], perm[:n_cal]


class SplitConformalRegressor:
    """Split (inductive) conformal regression.

    Parameters
    ----------
    model : estimator with fit/predict
    alpha : target miscoverage; intervals cover with probability >= 1 - alpha
    score : "absolute" for |y - mu(x)|, or "normalized" for |y - mu(x)| / sigma(x)
    difficulty_model : regressor used to learn sigma(x) from training residuals
        (only for score="normalized"). Defaults to quadratic ridge.
    """

    def __init__(self, model=None, alpha: float = 0.1, score: str = "absolute",
                 difficulty_model=None, cal_fraction: float = 0.3, random_state=None):
        if score not in ("absolute", "normalized"):
            raise ValueError("score must be 'absolute' or 'normalized'")
        self.model = model if model is not None else RidgeRegressor()
        self.alpha = alpha
        self.score = score
        self.difficulty_model = difficulty_model
        self.cal_fraction = cal_fraction
        self.random_state = random_state

    # training -----------------------------------------------------------------
    def fit(self, X, y):
        """Split into proper training and calibration sets, fit, calibrate."""
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        tr, cal = _split(len(y), self.cal_fraction, rng)
        self.model.fit(X[tr], y[tr])
        if self.score == "normalized":
            self._fit_difficulty(X[tr], y[tr])
        return self.calibrate(X[cal], y[cal])

    def _fit_difficulty(self, X, y):
        resid = np.abs(y - self.model.predict(X))
        self.sigma_model_ = (self.difficulty_model or RidgeRegressor(degree=2))
        self.sigma_model_.fit(X, resid)
        # floor so a near zero sigma(x) cannot blow scores up
        self.sigma_floor_ = max(1e-8, 0.05 * float(np.mean(resid)))

    def calibrate(self, X_cal, y_cal):
        """Compute calibration scores for an already fitted model."""
        if self.score == "normalized" and not hasattr(self, "sigma_model_"):
            raise RuntimeError("normalized score needs fit() or fit_difficulty() first")
        X_cal, y_cal = np.asarray(X_cal, dtype=float), np.asarray(y_cal, dtype=float)
        self.cal_scores_ = np.abs(y_cal - self.model.predict(X_cal)) / self._sigma(X_cal)
        return self

    def fit_difficulty(self, X, y):
        """Public hook to learn sigma(x) when using a prefit point model."""
        self._fit_difficulty(np.asarray(X, float), np.asarray(y, float))
        return self

    def _sigma(self, X):
        if self.score == "absolute":
            return np.ones(len(X))
        return np.maximum(self.sigma_model_.predict(X), self.sigma_floor_)

    # inference ----------------------------------------------------------------
    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=float))

    def predict_interval(self, X, alpha: float | None = None):
        X = np.asarray(X, dtype=float)
        q = conformal_quantile(self.cal_scores_, alpha or self.alpha)
        mu = self.model.predict(X)
        half = q * self._sigma(X)
        return mu - half, mu + half


class ConformalizedQuantileRegressor:
    """CQR: conformal calibration of a lower/upper quantile regression pair.

    Score: E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)), positive when y_i falls
    outside the raw band. The final band is [q_lo - Q, q_hi + Q], which can
    shrink (Q < 0) as well as grow if the raw quantiles were too conservative.

    With ``asymmetric=True`` the two tails are calibrated separately at
    alpha/2 each, which also controls each tail's miscoverage.
    """

    def __init__(self, lower_model=None, upper_model=None, alpha: float = 0.1,
                 degree: int = 5, asymmetric: bool = False,
                 cal_fraction: float = 0.3, random_state=None):
        self.lower_model = lower_model
        self.upper_model = upper_model
        self.alpha = alpha
        self.degree = degree
        self.asymmetric = asymmetric
        self.cal_fraction = cal_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        tr, cal = _split(len(y), self.cal_fraction, rng)
        self.lo_ = self.lower_model or QuantileRegressor(self.alpha / 2, degree=self.degree)
        self.hi_ = self.upper_model or QuantileRegressor(1 - self.alpha / 2, degree=self.degree)
        self.lo_.fit(X[tr], y[tr])
        self.hi_.fit(X[tr], y[tr])
        return self.calibrate(X[cal], y[cal])

    def calibrate(self, X_cal, y_cal):
        X_cal, y_cal = np.asarray(X_cal, dtype=float), np.asarray(y_cal, dtype=float)
        lo, hi = self._raw(X_cal)
        self.lo_scores_ = lo - y_cal
        self.hi_scores_ = y_cal - hi
        self.cal_scores_ = np.maximum(self.lo_scores_, self.hi_scores_)
        return self

    def _raw(self, X):
        lo, hi = self.lo_.predict(X), self.hi_.predict(X)
        # quantile crossing: sort so the band is never inverted
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def predict_interval(self, X, alpha: float | None = None):
        a = alpha or self.alpha
        lo, hi = self._raw(np.asarray(X, dtype=float))
        if self.asymmetric:
            return (lo - conformal_quantile(self.lo_scores_, a / 2),
                    hi + conformal_quantile(self.hi_scores_, a / 2))
        q = conformal_quantile(self.cal_scores_, a)
        return lo - q, hi + q


class JackknifePlusRegressor:
    """Jackknife+ (n_folds=None, leave one out) or CV+ (n_folds=K).

    For each training point i we keep the out of fold residual R_i and the
    model fitted without i's fold. At a test x the interval is

        [ q_lo{ mu_{-k(i)}(x) - R_i },  q_hi{ mu_{-k(i)}(x) + R_i } ]

    which is guaranteed to cover with probability >= 1 - 2 alpha and in
    practice sits close to 1 - alpha.
    """

    def __init__(self, model=None, alpha: float = 0.1, n_folds: int | None = 10,
                 random_state=None):
        self.model = model if model is not None else RidgeRegressor()
        self.alpha = alpha
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n = len(y)
        k = n if self.n_folds is None else int(self.n_folds)
        if not 2 <= k <= n:
            raise ValueError("n_folds must be between 2 and n_samples")
        rng = np.random.default_rng(self.random_state)
        folds = np.array_split(rng.permutation(n), k)
        self.fold_of_ = np.empty(n, dtype=int)
        self.residuals_ = np.empty(n)
        self.models_ = []
        for f, idx in enumerate(folds):
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            m = clone(self.model).fit(X[mask], y[mask])
            self.models_.append(m)
            self.fold_of_[idx] = f
            self.residuals_[idx] = np.abs(y[idx] - m.predict(X[idx]))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.mean([m.predict(X) for m in self.models_], axis=0)

    def predict_interval(self, X, alpha: float | None = None):
        a = alpha or self.alpha
        X = np.asarray(X, dtype=float)
        fold_preds = np.stack([m.predict(X) for m in self.models_], axis=1)
        M = fold_preds[:, self.fold_of_]  # (n_test, n_train)
        lo = lower_order_stat(M - self.residuals_, a, axis=1)
        hi = upper_order_stat(M + self.residuals_, a, axis=1)
        return lo, hi
