"""Logistic regression baseline for EHRSHOT-style benchmarking.

This module wraps scikit-learn's `LogisticRegression` and exposes a compact,
importable API suitable for benchmarking.

Assumptions:
- Inputs are numeric tabular features convertible to a 2D array.
- Binary classification is the primary target; multiclass is supported by
  scikit-learn, but the convenience evaluation method here is binary-focused.
- Missing values must be handled upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


@dataclass
class LogisticRegressionModel:
    penalty: str = "l2"
    C: float = 1.0
    fit_intercept: bool = True
    class_weight: Optional[Any] = None
    solver: str = "lbfgs"
    max_iter: int = 100
    random_state: Optional[int] = None
    n_jobs: Optional[int] = None
    l1_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        kwargs = dict(
            penalty=self.penalty,
            C=self.C,
            fit_intercept=self.fit_intercept,
            class_weight=self.class_weight,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            l1_ratio=self.l1_ratio,
        )
        self._model = LogisticRegression(**kwargs)
        self.is_fitted_ = False

    def fit(self, X: Any, y: Any) -> "LogisticRegressionModel":
        self._model.fit(np.asarray(X), np.asarray(y))
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        return self._model.predict(np.asarray(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        return self._model.predict_proba(np.asarray(X))

    def decision_function(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        return self._model.decision_function(np.asarray(X))

    def evaluate_binary(self, X: Any, y: Any) -> Dict[str, float]:
        y_true = np.asarray(y).astype(int)
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        positive_scores = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
        metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, positive_scores))
        return metrics

    def get_params(self) -> Dict[str, Any]:
        return {
            "penalty": self.penalty,
            "C": self.C,
            "fit_intercept": self.fit_intercept,
            "class_weight": self.class_weight,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "l1_ratio": self.l1_ratio,
        }

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("LogisticRegressionModel must be fitted before calling predict().")


Model = LogisticRegressionModel
