import math

import numpy as np
import pytest

from conformal import AdaptiveConformal
from conformal.datasets import regime_shift_series


def _scores(T=6000, seed=0):
    y, _ = regime_shift_series(T, rng=seed)
    pred = 0.8 * np.r_[0.0, y[:-1]]  # oracle AR(1) mean forecast
    return np.abs(y - pred)


def test_aci_long_run_miscoverage_matches_bound():
    s = _scores()
    aci = AdaptiveConformal(alpha=0.1, gamma=0.01)
    aci.run(s)
    assert abs(aci.miscoverage() - 0.1) <= aci.bound() + 1e-12


def test_aci_beats_static_under_regime_shift():
    s = _scores(seed=3)
    T = len(s)
    seg = slice(T // 4 + 50, T // 2)  # inside the high volatility regime
    static = AdaptiveConformal(alpha=0.1, gamma=0.0)
    static.run(s)
    adaptive = AdaptiveConformal(alpha=0.1, gamma=0.02, window=300)
    adaptive.run(s)
    miss_static = np.mean(static.errors[seg])
    miss_adapt = np.mean(adaptive.errors[seg])
    assert miss_adapt < miss_static
    assert abs(miss_adapt - 0.1) < 0.05


def test_radius_edge_cases():
    aci = AdaptiveConformal(alpha=0.1)
    assert math.isinf(aci.radius())  # no history yet
    aci.alpha_t = 1.2
    aci.scores = [1.0, 2.0]
    assert aci.radius() == 0.0
    aci.alpha_t = -0.1
    assert math.isinf(aci.radius())


def test_alpha_moves_in_the_right_direction():
    aci = AdaptiveConformal(alpha=0.1, gamma=0.1)
    aci.scores = [1.0] * 50
    aci.update(100.0)  # a miss should lower alpha_t -> wider future intervals
    assert aci.alpha_t < 0.1
    a = aci.alpha_t
    aci.update(0.0)    # a hit nudges it back up
    assert aci.alpha_t > a


def test_invalid_args():
    with pytest.raises(ValueError):
        AdaptiveConformal(alpha=0)
    with pytest.raises(ValueError):
        AdaptiveConformal(gamma=-1)
