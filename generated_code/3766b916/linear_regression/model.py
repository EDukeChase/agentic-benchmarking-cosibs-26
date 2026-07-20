"""Linear regression baseline for EHRSHOT-style benchmarking.

This module provides a small, importable wrapper around scikit-learn's
``LinearRegression`` for binary outcomes. The model is treated as a
linear-probability baseline: outputs are real-valued scores that may be clipped
to [0, 1] when probability-like values are needed.

Assumptions
-----------
- Inputs are already numerically encoded and imputed.
- Binary labels are encoded as 0/1.
- When used for classification metrics, scores are thresholded at 0.5 after
  optional clipping to [0, 1].
- Multi-output regression is not the intended use case, but the wrapper will
  pass through any 2D target accepted by scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.linear_model import LinearRegression

ArrayLike = Union[np.ndarray, list]


@dataclass
class LinearRegressionConfig:
    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: Optional[int] = None
    positive: bool = False
    clip_predictions: bool = True
    threshold: float = 0.5


class LinearRegressionModel:
    def __init__(self, config: Optional[LinearRegressionConfig] = None):
        self.config = config or LinearRegressionConfig()
        init_kwargs = dict(
            fit_intercept=self.config.fit_intercept,
            copy_X=self.config.copy_X,
            positive=self.config.positive,
        )
        if self.config.n_jobs is not None:
            init_kwargs["n_jobs"] = self.config.n_jobs
        self.model = LinearRegression(**init_kwargs)
        self.is_fitted_ = False

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegressionModel":
        self.model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict_scores(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        scores = np.asarray(self.model.predict(X), dtype=float)
        if self.config.clip_predictions:
            scores = np.clip(scores, 0.0, 1.0)
        return scores

    def predict(self, X: ArrayLike) -> np.ndarray:
        scores = self.predict_scores(X)
        return (scores >= self.config.threshold).astype(int)

    def evaluate(self, X: ArrayLike, y: ArrayLike) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

        y_true = np.asarray(y)
        scores = self.predict_scores(X)
        preds = (scores >= self.config.threshold).astype(int)
        metrics: Dict[str, float] = {
            "mse": float(mean_squared_error(y_true, scores)),
            "accuracy": float(accuracy_score(y_true, preds)),
        }
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            metrics["auroc"] = float("nan")
        return metrics

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params(deep=True)

    def set_params(self, **params: Any) -> "LinearRegressionModel":
        self.model.set_params(**params)
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("LinearRegressionModel must be fitted before prediction.")


def build_model(**kwargs: Any) -> LinearRegressionModel:
    config = LinearRegressionConfig(**kwargs)
    return LinearRegressionModel(config=config)


__all__ = ["LinearRegressionConfig", "LinearRegressionModel", "build_model"]
