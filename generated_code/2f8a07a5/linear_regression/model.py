"""Linear regression baseline for EHRSHOT benchmarking.

This module provides a small, importable wrapper around scikit-learn's
``LinearRegression`` with a classification-oriented convenience API.

Assumptions
-----------
- The literature review describes ordinary least squares linear regression as a
  baseline for binary outcomes.
- Because linear regression outputs are not constrained to ``[0, 1]``, this
  implementation exposes both raw regression predictions and clipped
  probabilities for binary-classification use cases.
- If targets are binary and the caller requests classification-style outputs,
  thresholding is done at 0.5 after clipping probabilities to ``[0, 1]``.
- Multi-output regression is supported through the underlying scikit-learn
  estimator, but only binary classification helper methods are provided.

Known limitations
-----------------
- This is not a calibrated probabilistic model.
- For binary classification, the model can produce values outside the unit
  interval before clipping, which may reduce interpretability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class LinearRegressionConfig:
    fit_intercept: bool = True
    copy_X: bool = True
    positive: bool = False
    tol: float = 1e-6


class LinearRegressionModel:
    def __init__(self, config: Optional[LinearRegressionConfig] = None):
        self.config = config or LinearRegressionConfig()
        self.model = LinearRegression(
            fit_intercept=self.config.fit_intercept,
            copy_X=self.config.copy_X,
            positive=self.config.positive,
            tol=self.config.tol,
        )
        self.is_fitted = False

    def fit(self, X: Any, y: Any, sample_weight: Any = None):
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict(self, X: Any):
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: Any):
        preds = np.asarray(self.predict(X), dtype=float)
        if preds.ndim == 1:
            p1 = np.clip(preds, 0.0, 1.0)
            return np.column_stack([1.0 - p1, p1])
        return np.clip(preds, 0.0, 1.0)

    def predict_label(self, X: Any, threshold: float = 0.5):
        proba = self.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return (proba[:, 1] >= threshold).astype(int)
        raise ValueError("predict_label is only defined for binary targets.")

    def score(self, X: Any, y: Any, sample_weight: Any = None):
        self._check_fitted()
        return self.model.score(X, y, sample_weight=sample_weight)

    def evaluate_regression(self, X: Any, y: Any, sample_weight: Any = None):
        preds = self.predict(X)
        return {
            "r2": r2_score(y, preds, sample_weight=sample_weight),
            "mse": mean_squared_error(y, preds, sample_weight=sample_weight),
        }

    def get_params(self, deep: bool = True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("LinearRegressionModel must be fitted before use.")


def build_model(**kwargs) -> LinearRegressionModel:
    return LinearRegressionModel(LinearRegressionConfig(**kwargs))


__all__ = ["LinearRegressionConfig", "LinearRegressionModel", "build_model"]
