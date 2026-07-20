"""Linear regression baseline for EHRSHOT-style benchmarking.

This module provides a lightweight wrapper around scikit-learn's
`LinearRegression` for regression or binary-label linear-probability
baseline use cases.

Assumptions:
- Inputs are tabular numeric features convertible to a 2D array.
- For binary classification, predictions are treated as scores/probabilities
  via clipping to [0, 1]. The default threshold is 0.5.
- Missing values are not imputed here; callers must preprocess them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class LinearRegressionModel:
    """Wrapper for ordinary least squares linear regression."""

    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: Optional[int] = None
    positive: bool = False

    def __post_init__(self) -> None:
        self._model = LinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs,
            positive=self.positive,
        )
        self.is_fitted_ = False

    def fit(self, X: Any, y: Any) -> "LinearRegressionModel":
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        self._model.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        return self._model.predict(np.asarray(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        preds = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - preds, preds])

    def predict_binary(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        return (np.clip(self.predict(X), 0.0, 1.0) >= threshold).astype(int)

    def evaluate_regression(self, X: Any, y: Any) -> Dict[str, float]:
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def evaluate_binary(self, X: Any, y: Any, threshold: float = 0.5) -> Dict[str, float]:
        y_true = np.asarray(y).astype(int)
        y_score = np.clip(self.predict(X), 0.0, 1.0)
        y_pred = (y_score >= threshold).astype(int)
        accuracy = float((y_pred == y_true).mean())
        return {"accuracy": accuracy}

    def get_params(self) -> Dict[str, Any]:
        return {
            "fit_intercept": self.fit_intercept,
            "copy_X": self.copy_X,
            "n_jobs": self.n_jobs,
            "positive": self.positive,
        }

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("LinearRegressionModel must be fitted before calling predict().")


Model = LinearRegressionModel
