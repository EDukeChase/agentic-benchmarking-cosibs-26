"""Linear regression baseline for EHRSHOT-style benchmarking.

This module provides a small, importable wrapper around scikit-learn's
``LinearRegression`` for binary outcomes, treating the regression output as a
linear-probability score.

Assumptions
-----------
- Targets are binary or can be coerced to binary labels.
- Predictions are clipped to ``[0, 1]`` when probability-like outputs are needed.
- Missing values are not imputed here; callers should preprocess inputs or use
  an upstream imputer/pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


@dataclass
class LinearRegressionModel:
    random_state: Optional[int] = None
    fit_intercept: bool = True
    n_jobs: Optional[int] = None

    def __post_init__(self) -> None:
        self.model = LinearRegression(fit_intercept=self.fit_intercept, n_jobs=self.n_jobs)

    def fit(self, X: Any, y: Any) -> "LinearRegressionModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        scores = np.asarray(self.model.predict(X), dtype=float)
        scores = np.clip(scores, 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def evaluate(self, X: Any, y: Any, threshold: float = 0.5) -> dict:
        y_true = np.asarray(y)
        proba = self.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        metrics = {"accuracy": accuracy_score(y_true, pred)}
        try:
            metrics["auroc"] = roc_auc_score(y_true, proba)
        except Exception:
            metrics["auroc"] = np.nan
        try:
            metrics["auprc"] = average_precision_score(y_true, proba)
        except Exception:
            metrics["auprc"] = np.nan
        return metrics


Model = LinearRegressionModel
