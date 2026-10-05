"""conformal: distribution free uncertainty quantification in pure NumPy."""

from ._quantile import conformal_quantile
from .classification import ConformalClassifier, label_scores
from .metrics import (binned_coverage, interval_coverage, mean_set_size, mean_width,
                      set_coverage, size_stratified_coverage, worst_bin_coverage)
from .models import QuantileRegressor, RidgeRegressor, SoftmaxClassifier
from .online import AdaptiveConformal
from .regression import (ConformalizedQuantileRegressor, JackknifePlusRegressor,
                         SplitConformalRegressor)

__all__ = [
    "conformal_quantile", "ConformalClassifier", "label_scores",
    "SplitConformalRegressor", "ConformalizedQuantileRegressor", "JackknifePlusRegressor",
    "AdaptiveConformal", "RidgeRegressor", "QuantileRegressor", "SoftmaxClassifier",
    "interval_coverage", "mean_width", "set_coverage", "mean_set_size",
    "binned_coverage", "size_stratified_coverage", "worst_bin_coverage",
]
__version__ = "0.1.0"
