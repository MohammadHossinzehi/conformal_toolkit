# conformal_toolkit

Prediction intervals and prediction sets with a **finite sample coverage guarantee**, for any model, in about 700 lines of NumPy.

A point prediction says "the price will be 412". A conformal prediction says "the price will be in [371, 455], and that statement is right at least 90% of the time", with no assumptions about the model being correct or the noise being Gaussian. The only requirement is that calibration and test data are exchangeable (and for the online method in this repo, not even that).

This repo implements the main conformal methods from the literature from first principles, wraps any estimator with `fit` / `predict` / `predict_proba` (the bundled NumPy models or anything scikit-learn shaped), and ships a test suite that checks the coverage guarantees statistically rather than just checking that code runs.

## What is in here

| Problem | Method | Module |
|---|---|---|
| Regression intervals | Split conformal, absolute residual score | `regression.SplitConformalRegressor` |
| | Normalized (locally weighted) scores: residual / learned sigma(x) | `score="normalized"` |
| | Conformalized Quantile Regression (Romano, Patterson, Candes 2019), symmetric or per tail | `regression.ConformalizedQuantileRegressor` |
| | Jackknife+ and CV+ (Barber, Candes, Ramdas, Tibshirani 2021) | `regression.JackknifePlusRegressor` |
| Classification sets | LAC (Sadinle, Lei, Wasserman 2019) | `ConformalClassifier(method="lac")` |
| | APS, randomized or not (Romano, Sesia, Candes 2020) | `method="aps"` |
| | RAPS (Angelopoulos et al. 2021) | `method="raps"` |
| | Mondrian / class conditional calibration | `class_conditional=True` |
| Streams with drift | Adaptive Conformal Inference (Gibbs and Candes 2021) | `online.AdaptiveConformal` |
| Diagnostics | coverage, width, set size, binned (conditional) coverage, size stratified coverage | `metrics` |

Base learners (`RidgeRegressor`, an IRLS `QuantileRegressor`, `SoftmaxClassifier`) live in `models.py` so the whole thing runs with NumPy only.

## Quick start

```bash
git clone https://github.com/MohammadHossinzehi/conformal_toolkit
cd conformal_toolkit
pip install numpy pytest
python -m pytest            # 24 tests, ~10 s
python examples/benchmark.py
```

```python
from conformal import ConformalizedQuantileRegressor, ConformalClassifier, AdaptiveConformal
from conformal.datasets import heteroscedastic, gaussian_blobs

X, y, _ = heteroscedastic(1000, rng=0)
cqr = ConformalizedQuantileRegressor(alpha=0.1).fit(X, y)
lo, hi = cqr.predict_interval(X[:5])            # 90% intervals

Xc, yc = gaussian_blobs(2000, rng=0)
clf = ConformalClassifier(alpha=0.1, method="aps").fit(Xc, yc)
clf.predict_labels(Xc[:3])                      # e.g. [[2], [0, 4], [1, 2, 3]]

aci = AdaptiveConformal(alpha=0.1, gamma=0.01)  # online
# inside your loop, with your own forecast and observed value:
r = aci.radius()                                 # interval = forecast +/- r
aci.update(abs(y_true - forecast))               # adapts after each observation
```

Bring your own model: anything with `fit(X, y)` and `predict(X)` (regression) or `predict_proba(X)` and `classes_` (classification). Already trained? Call `.calibrate(X_cal, y_cal)` instead of `.fit`.

## Benchmark output

`python examples/benchmark.py` (20 trials each, target coverage 0.90):

```
Regression, heteroscedastic noise, alpha=0.1, 20 trials
method                  coverage    width  worst |x| bin
split (absolute)           0.901    3.271          0.742
split (normalized)         0.897    2.925          0.853
CQR                        0.904    2.956          0.861
CV+ (10 folds)             0.902    3.250          0.742

Classification, 5 overlapping classes, alpha=0.1, 20 trials
method                  coverage  avg size
LAC                        0.902      1.54
APS (randomized)           0.909      1.72
RAPS lam=0.05              0.908      1.69
LAC Mondrian               0.906      1.57

Online, AR(1) with volatility regime shifts, alpha=0.1
method                     overall miss  miss in high vol
static rolling (gamma=0)          0.117             0.344
ACI gamma=0.02, window            0.100             0.098
```

How to read it:

* **Every method hits 90% marginal coverage.** That is the guarantee, and it holds regardless of how good the model is.
* **Marginal coverage hides a lot.** The constant width methods (absolute split, CV+) only reach about 74% coverage on the noisiest fifth of inputs and overcover the quiet ones. Normalized scores and CQR adapt the width to the noise, which buys narrower intervals *and* much better conditional coverage.
* **LAC gives the smallest sets; APS and RAPS spend a little size** to spread coverage more evenly across easy and hard inputs.
* **Under distribution shift the static method breaks** (34% misses during the high volatility regime, more than three times the target), while ACI holds about 10% in every regime.

## Design notes

**One quantile function, used everywhere.** `conformal_quantile` takes the `ceil((n+1)(1-alpha))`th smallest score, which is exactly the finite sample correction that makes the guarantee hold. If there are too few calibration points to certify the level it returns `+inf` rather than quietly undercovering. Jackknife+ uses the matching lower and upper order statistics.

**Classification is "score, threshold, filter".** `label_scores` builds an `(n, K)` matrix of `s(x, k)` for every candidate label, and a set is simply `S <= qhat`. Calibration and prediction share the same function, so a score can never be computed one way at calibration time and another at test time, which is the classic bug in hand rolled APS implementations. Mondrian calibration is just a per column threshold vector.

**Randomized APS** subtracts `U * p_k` from the cumulative mass. That makes coverage exact (not merely at least 1 - alpha) at the cost of occasionally empty sets; by default an empty set is replaced with the top label, which can only increase coverage.

**CQR handles quantile crossing** by sorting the two raw quantiles before calibrating, and the conformal correction can be negative, so an overly conservative quantile model gets tightened rather than only widened. The asymmetric variant calibrates each tail at alpha/2, which also controls the miss rate on each side separately (tested).

**ACI** keeps a running `alpha_t` and nudges it by `gamma * (alpha - err_t)`. `bound()` returns the Gibbs and Candes deterministic bound on the long run miss rate, and the test asserts the empirical rate respects it.

## How it is tested

Conformal guarantees are statements about averages over random draws, so the tests are statistical: each coverage test repeats the full fit, calibrate, evaluate cycle over 30 to 60 seeded datasets and checks that the mean coverage lands inside the theoretical window `[1 - alpha, 1 - alpha + 1/(n_cal + 1)]` with a small Monte Carlo tolerance. Other tests check:

* the quantile index arithmetic and the `+inf` edge case
* that normalized intervals are wider where the noise is larger
* that CQR bands are narrower than constant width bands under heteroscedasticity
* per tail miscoverage for asymmetric CQR
* jackknife+ (leave one out) and CV+ coverage, including the 1 - 2 alpha worst case
* hand computed APS and RAPS scores
* per class coverage for Mondrian calibration on an imbalanced problem
* the ACI bound, ACI beating a static baseline during a volatility spike, and its edge cases
* the base learners themselves (ridge recovers exact coefficients, quantile regression hits its target quantile)

All randomness is seeded, so the suite is deterministic.

## Layout

```
conformal/
  _quantile.py        finite sample quantile and order statistics
  regression.py       split, normalized, CQR, jackknife+ / CV+
  classification.py   LAC, APS, RAPS, Mondrian
  online.py           adaptive conformal inference
  metrics.py          coverage diagnostics
  models.py           NumPy ridge, quantile regression, softmax
  datasets.py         synthetic data with known structure
examples/benchmark.py
tests/
```

## License

MIT
