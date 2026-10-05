"""Adaptive Conformal Inference (Gibbs and Candes, 2021) for non exchangeable streams.

Split conformal assumes calibration and test points are exchangeable. Under
distribution shift (volatility regimes, drifting sensors) that breaks and
coverage silently decays. ACI treats the working miscoverage level alpha_t as
a control variable updated online:

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t),   err_t = 1{y_t not covered}

Every miss lowers alpha_t (wider intervals), every hit raises it a little.
Whatever the data does, the long run miss rate obeys

    | (1/T) sum err_t - alpha | <= (max(alpha_1, 1 - alpha_1) + gamma) / (gamma T)

so it converges to alpha deterministically, with no distributional assumption.
"""

from __future__ import annotations

import math

import numpy as np

from ._quantile import conformal_quantile


class AdaptiveConformal:
    """Online interval radius from a stream of nonconformity scores.

    Typical loop with a point forecaster f::

        aci = AdaptiveConformal(alpha=0.1, gamma=0.01)
        for x_t, y_t in stream:
            r = aci.radius()                  # interval is f(x_t) +- r
            aci.update(abs(y_t - f(x_t)))     # reveal truth, adapt

    gamma=0 recovers a plain rolling split conformal baseline.
    ``window`` keeps only the most recent scores (None = all history).
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.005, window: int | None = None):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if gamma < 0:
            raise ValueError("gamma must be non negative")
        self.alpha = alpha
        self.gamma = gamma
        self.window = window
        self.alpha_t = alpha
        self.scores: list[float] = []
        self.errors: list[int] = []
        self.alpha_path: list[float] = []

    def radius(self) -> float:
        hist = self.scores[-self.window:] if self.window else self.scores
        if self.alpha_t >= 1.0:
            return 0.0
        if self.alpha_t <= 0.0 or not hist:
            return math.inf
        return conformal_quantile(hist, self.alpha_t)

    def update(self, score: float) -> int:
        """Record the realised score; returns 1 if it was a miss."""
        r = self.radius()
        err = int(score > r)
        self.alpha_path.append(self.alpha_t)
        self.errors.append(err)
        self.alpha_t += self.gamma * (self.alpha - err)
        self.scores.append(float(score))
        return err

    def run(self, scores):
        """Process a whole score sequence. Returns (radii, errors, alpha_path)."""
        radii = []
        for s in scores:
            radii.append(self.radius())
            self.update(s)
        return np.array(radii), np.array(self.errors), np.array(self.alpha_path)

    def miscoverage(self, burn_in: int = 0) -> float:
        e = np.asarray(self.errors[burn_in:])
        return float(e.mean()) if e.size else float("nan")

    def bound(self) -> float:
        """Theoretical bound on |empirical miss rate - alpha| so far."""
        T = len(self.errors)
        if T == 0 or self.gamma == 0:
            return math.inf
        return (max(self.alpha, 1 - self.alpha) + self.gamma) / (self.gamma * T)
