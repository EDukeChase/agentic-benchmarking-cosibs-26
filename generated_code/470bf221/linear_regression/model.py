"""Linear regression baseline for binary outcomes.

This module implements a scikit-learn compatible wrapper around
sklearn.linear_model.LinearRegression. For binary classification benchmarking,
probabilities are obtained by clipping raw regression outputs to [0, 1].

Assumptions:
- y is binary or can be interpreted as binary labels.
- predict_proba is preferred for benchmarking, but predict is also provided.
- Linear regression is used as a linear-probability model baseline, not a
  calibrated probabilistic classifier.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression


class LinearRegressionModel:
    def __init__(self, **kwargs: Any) -> None:
        self.model = LinearRegression(**kwargs)
        self._is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X):
        preds = np.asarray(self.model.predict(X), dtype=float)
        return (preds >= 0.5).astype(int)

    def predict_proba(self, X):
        raw = np.asarray(self.model.predict(X), dtype=float)
        pos = np.clip(raw, 0.0, 1.0)
        neg = 1.0 - pos
        return np.column_stack([neg, pos])


Model = LinearRegressionModel
