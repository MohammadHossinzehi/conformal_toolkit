import numpy as np
import pytest

from conformal import (ConformalClassifier, SoftmaxClassifier, label_scores,
                       mean_set_size, set_coverage, size_stratified_coverage)
from conformal.datasets import gaussian_blobs

ALPHA = 0.1


def _run(method, trials=30, **kw):
    covs, sizes = [], []
    for t in range(trials):
        X, y = gaussian_blobs(1500, rng=t)
        Xt, yt = gaussian_blobs(1000, rng=5000 + t)
        clf = ConformalClassifier(SoftmaxClassifier(n_iter=300), alpha=ALPHA, method=method,
                                  random_state=t, **kw).fit(X, y)
        sets = clf.predict_set(Xt)
        covs.append(set_coverage(sets, yt))
        sizes.append(mean_set_size(sets))
    return float(np.mean(covs)), float(np.mean(sizes))


@pytest.mark.parametrize("method", ["lac", "aps", "raps"])
def test_marginal_coverage(method):
    cov, size = _run(method)
    assert 0.885 <= cov <= 0.93
    assert 1.0 <= size < 5.0


def test_lac_gives_smallest_sets():
    _, s_lac = _run("lac", trials=15)
    _, s_aps = _run("aps", trials=15)
    assert s_lac <= s_aps


def test_aps_score_is_cumulative_mass():
    P = np.array([[0.5, 0.3, 0.2]])
    S = label_scores(P, "aps")
    assert np.allclose(S, [[0.5, 0.8, 1.0]])
    S_r = label_scores(P, "aps", U=np.array([1.0]))
    assert np.allclose(S_r, [[0.0, 0.5, 0.8]])  # subtract own mass at U=1


def test_raps_penalises_deep_ranks():
    P = np.array([[0.4, 0.3, 0.2, 0.1]])
    S = label_scores(P, "raps", lam=0.5, k_reg=2)
    assert np.allclose(S, [[0.4, 0.7, 0.9 + 0.5, 1.0 + 1.0]])


def test_mondrian_gives_per_class_coverage():
    # imbalanced classes: marginal calibration under covers the rare class
    rng = np.random.default_rng(0)
    covs = []
    for t in range(10):
        X, y = gaussian_blobs(6000, k=4, spread=1.8, rng=t)
        keep = (y != 3) | (rng.uniform(size=len(y)) < 0.15)
        X, y = X[keep], y[keep]
        Xt, yt = gaussian_blobs(4000, k=4, spread=1.8, rng=900 + t)
        clf = ConformalClassifier(SoftmaxClassifier(), alpha=ALPHA, method="lac",
                                  class_conditional=True, random_state=t).fit(X, y)
        sets = clf.predict_set(Xt)
        covs.append([set_coverage(sets[yt == k], yt[yt == k]) for k in range(4)])
    per_class = np.mean(covs, axis=0)
    assert np.all(per_class >= 0.87)


def test_size_stratified_and_labels():
    X, y = gaussian_blobs(800, rng=1)
    clf = ConformalClassifier(alpha=0.2, random_state=0).fit(X, y)
    labels = clf.predict_labels(X[:5])
    assert all(len(l) >= 1 for l in labels)
    strat = size_stratified_coverage(clf.predict_set(X), y)
    assert sum(n for _, n in strat.values()) == len(y)


def test_softmax_classifier_learns():
    X, y = gaussian_blobs(2000, spread=0.5, rng=0)
    assert np.mean(SoftmaxClassifier().fit(X, y).predict(X) == y) > 0.95
