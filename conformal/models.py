"""Small NumPy base learners so the package runs with zero extra dependencies.

The conformal wrappers only need duck typing: ``fit(X, y)`` plus ``predict``
(regression) or ``predict_proba`` (classification). Any scikit-learn style
estimator works just as well; these exist so the examples and tests are
self-contained.
"""

from __future__ import annotations

import copy

import numpy as np


def clone(model):
    """Fresh unfitted copy of a model (sklearn's clone if available)."""
    try:  # pragma: no cover - only when sklearn is installed
        from sklearn.base import clone as sk_clone

        return sk_clone(model)
    except Exception:
        return copy.deepcopy(model)


class PolynomialFeatures:
    """Per feature powers 1..degree (no cross terms), standardised."""

    def __init__(self, degree: int = 3):
        self.degree = degree

    def fit(self, X):
        Z = self._raw(X)
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0) + 1e-12
        return self

    def transform(self, X):
        return (self._raw(X) - self.mean_) / self.std_

    def _raw(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.hstack([X ** d for d in range(1, self.degree + 1)])


class RidgeRegressor:
    """Closed form ridge regression with optional polynomial expansion."""

    def __init__(self, lam: float = 1e-3, degree: int = 1):
        self.lam = lam
        self.degree = degree

    def fit(self, X, y):
        self.poly_ = PolynomialFeatures(self.degree).fit(X)
        Z = self.poly_.transform(X)
        y = np.asarray(y, dtype=float)
        self.intercept_ = y.mean()
        A = Z.T @ Z + self.lam * len(y) * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ (y - self.intercept_))
        return self

    def predict(self, X):
        return self.poly_.transform(X) @ self.coef_ + self.intercept_


class QuantileRegressor:
    """Linear (or polynomial) quantile regression via IRLS on the pinball loss.

    Each iteration solves a weighted least squares problem whose weights
    tau/|r| or (1-tau)/|r| make the squared loss match the pinball loss at
    the current residuals. A small floor on |r| keeps it stable.
    """

    def __init__(self, tau: float = 0.5, degree: int = 1, lam: float = 1e-4,
                 n_iter: int = 100, eps: float = 1e-4):
        if not 0 < tau < 1:
            raise ValueError("tau must be in (0, 1)")
        self.tau, self.degree, self.lam = tau, degree, lam
        self.n_iter, self.eps = n_iter, eps

    def fit(self, X, y):
        self.poly_ = PolynomialFeatures(self.degree).fit(X)
        Z = self.poly_.transform(X)
        Z1 = np.hstack([np.ones((Z.shape[0], 1)), Z])
        y = np.asarray(y, dtype=float)
        reg = self.lam * len(y) * np.eye(Z1.shape[1])
        reg[0, 0] = 0.0
        beta = np.linalg.lstsq(Z1, y, rcond=None)[0]
        for _ in range(self.n_iter):
            r = y - Z1 @ beta
            w = np.where(r >= 0, self.tau, 1 - self.tau) / np.maximum(np.abs(r), self.eps)
            Zw = Z1 * w[:, None]
            new = np.linalg.solve(Z1.T @ Zw + reg, Zw.T @ y)
            if np.max(np.abs(new - beta)) < 1e-8:
                beta = new
                break
            beta = new
        self.beta_ = beta
        return self

    def predict(self, X):
        Z = self.poly_.transform(X)
        return self.beta_[0] + Z @ self.beta_[1:]


class SoftmaxClassifier:
    """Multinomial logistic regression trained with full batch gradient descent."""

    def __init__(self, lam: float = 1e-3, lr: float = 0.5, n_iter: int = 500, degree: int = 1):
        self.lam, self.lr, self.n_iter, self.degree = lam, lr, n_iter, degree

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        idx = np.searchsorted(self.classes_, y)
        self.poly_ = PolynomialFeatures(self.degree).fit(X)
        Z = self.poly_.transform(X)
        n, d = Z.shape
        K = len(self.classes_)
        Y = np.eye(K)[idx]
        W = np.zeros((d, K))
        b = np.zeros(K)
        for _ in range(self.n_iter):
            P = self._softmax(Z @ W + b)
            G = (P - Y) / n
            W -= self.lr * (Z.T @ G + self.lam * W)
            b -= self.lr * G.sum(axis=0)
        self.W_, self.b_ = W, b
        return self

    def predict_proba(self, X):
        return self._softmax(self.poly_.transform(X) @ self.W_ + self.b_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @staticmethod
    def _softmax(L):
        L = L - L.max(axis=1, keepdims=True)
        E = np.exp(L)
        return E / E.sum(axis=1, keepdims=True)
