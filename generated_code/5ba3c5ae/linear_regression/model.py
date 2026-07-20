"""Linear regression baseline for EHRSHOT-style benchmarking.

This module implements a small, self-contained wrapper around
``sklearn.linear_model.LinearRegression`` so it can be imported by benchmark
harnesses without depending on project-specific code.

The model is intended as a linear-probability baseline for binary outcomes.
Predictions are therefore optionally clipped to ``[0, 1]`` when probabilities
are requested.

Missing documentation assumptions:
- The literature review does not specify preprocessing, so this module expects
  features to already be numeric and reasonably cleaned.
- For binary evaluation, continuous predictions may be thresholded at 0.5.
- Sample weighting is supported through scikit-learn's native API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class LinearRegressionConfig:
    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: Optional[int] = None
    positive: bool = False
    clip_predictions: bool = True


class LinearRegressionModel:
    def __init__(self, config: Optional[LinearRegressionConfig] = None):
        self.config = config or LinearRegressionConfig()
        self.model = LinearRegression(
            fit_intercept=self.config.fit_intercept,
            copy_X=self.config.copy_X,
            n_jobs=self.config.n_jobs,
            positive=self.config.positive,
        )
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray, clip: Optional[bool] = None) -> np.ndarray:
        self._check_is_fitted()
        preds = self.model.predict(X)
        use_clip = self.config.clip_predictions if clip is None else clip
        if use_clip:
            preds = np.clip(preds, 0.0, 1.0)
        return preds

    def predict_labels(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict(X) >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        return {
            "mse": float(mean_squared_error(y, preds)),
            "r2": float(r2_score(y, preds)),
        }

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params(deep=True)

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("LinearRegressionModel must be fitted before use.")


__all__ = ["LinearRegressionConfig", "LinearRegressionModel"]
