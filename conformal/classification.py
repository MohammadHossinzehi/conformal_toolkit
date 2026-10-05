"""Conformal prediction sets for classification.

Every method here is "pick a score s(x, y), calibrate its quantile, return
{y : s(x, y) <= qhat}". They differ only in the score:

* LAC  (least ambiguous set valued classifier): s = 1 - p_y.
  Smallest average sets, but coverage concentrates on easy inputs.
* APS  (adaptive prediction sets): s = total probability mass of all labels
  ranked at or above y. Set size adapts to how unsure the model is.
* RAPS (regularized APS): APS plus lam * (rank(y) - k_reg)^+, which stops
  long tails of tiny probabilities from bloating the sets.

``class_conditional=True`` turns any of them into a Mondrian conformal
predictor: a separate threshold per class, so coverage holds within every
class rather than only on average.
"""

from __future__ import annotations

import numpy as np

from ._quantile import conformal_quantile
from .models import SoftmaxClassifier

METHODS = ("lac", "aps", "raps")


def label_scores(P, method: str = "aps", U=None, lam: float = 0.01, k_reg: int = 1):
    """Score matrix S[i, k] = s(x_i, k) for every candidate label k.

    U holds one uniform per row for randomized APS/RAPS (None = deterministic).
    """
    P = np.asarray(P, dtype=float)
    if method == "lac":
        return 1.0 - P
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    n, K = P.shape
    order = np.argsort(-P, axis=1, kind="stable")
    sorted_p = np.take_along_axis(P, order, axis=1)
    cum_sorted = np.cumsum(sorted_p, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(K)[None, :].repeat(n, axis=0), axis=1)
    S = np.take_along_axis(cum_sorted, ranks, axis=1)  # mass at or above each label
    if U is not None:
        S = S - np.asarray(U, dtype=float)[:, None] * P
    if method == "raps":
        S = S + lam * np.maximum(0, ranks + 1 - k_reg)
    return S


class ConformalClassifier:
    """Split conformal classifier producing label sets with >= 1 - alpha coverage.

    Parameters
    ----------
    model : estimator with fit / predict_proba (and optionally ``classes_``)
    method : "lac", "aps" or "raps"
    randomize : use the randomized APS/RAPS score (exact coverage, not just >=)
    lam, k_reg : RAPS regularisation strength and the rank after which it bites
    class_conditional : Mondrian calibration, one threshold per class
    allow_empty : if False, an empty set is replaced by the top-1 label
    """

    def __init__(self, model=None, alpha: float = 0.1, method: str = "aps",
                 randomize: bool = True, lam: float = 0.01, k_reg: int = 1,
                 class_conditional: bool = False, allow_empty: bool = False,
                 cal_fraction: float = 0.3, random_state=None):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.model = model if model is not None else SoftmaxClassifier()
        self.alpha = alpha
        self.method = method
        self.randomize = randomize
        self.lam, self.k_reg = lam, k_reg
        self.class_conditional = class_conditional
        self.allow_empty = allow_empty
        self.cal_fraction = cal_fraction
        self.rng_ = np.random.default_rng(random_state)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        perm = self.rng_.permutation(len(y))
        n_cal = max(1, int(round(len(y) * self.cal_fraction)))
        cal, tr = perm[:n_cal], perm[n_cal:]
        self.model.fit(X[tr], y[tr])
        return self.calibrate(X[cal], y[cal])

    def calibrate(self, X_cal, y_cal):
        X_cal, y_cal = np.asarray(X_cal, dtype=float), np.asarray(y_cal)
        P = self.model.predict_proba(X_cal)
        self.classes_ = np.asarray(getattr(self.model, "classes_", np.arange(P.shape[1])))
        idx = np.searchsorted(self.classes_, y_cal)
        S = self._scores(P)
        self.cal_scores_ = S[np.arange(len(idx)), idx]
        self.cal_labels_ = idx
        return self

    def _scores(self, P):
        U = self.rng_.uniform(size=len(P)) if (self.randomize and self.method != "lac") else None
        return label_scores(P, self.method, U, self.lam, self.k_reg)

    def thresholds(self, alpha: float | None = None):
        """qhat as a length K vector (all equal unless class_conditional)."""
        a = alpha or self.alpha
        K = len(self.classes_)
        if not self.class_conditional:
            return np.full(K, conformal_quantile(self.cal_scores_, a))
        return np.array([conformal_quantile(self.cal_scores_[self.cal_labels_ == k], a)
                         for k in range(K)])

    def predict_set(self, X, alpha: float | None = None):
        """Boolean matrix (n, K): column k is True when class k is in the set."""
        P = self.model.predict_proba(np.asarray(X, dtype=float))
        sets = self._scores(P) <= self.thresholds(alpha)[None, :]
        if not self.allow_empty:
            empty = ~sets.any(axis=1)
            sets[empty, np.argmax(P[empty], axis=1)] = True
        return sets

    def predict_labels(self, X, alpha: float | None = None):
        """Prediction sets as python lists of class labels."""
        return [list(self.classes_[row]) for row in self.predict_set(X, alpha)]
