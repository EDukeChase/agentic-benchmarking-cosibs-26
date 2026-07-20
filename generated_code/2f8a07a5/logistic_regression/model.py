"""Logistic regression baseline for EHRSHOT benchmarking.

This module wraps scikit-learn's ``LogisticRegression`` in a small importable
API suitable for training, prediction, and evaluation.

Assumptions
-----------
- The review targets binary outcome prediction.
- We default to a binary-friendly solver/configuration and expose probability
  predictions through the underlying model.
- If the caller provides multiclass labels, the underlying scikit-learn model
  may still handle them depending on solver choice, but the benchmarking focus
  here is binary classification.

Known limitations
-----------------
- Hyperparameter tuning is not implemented here.
- Solver/penalty compatibility is delegated to scikit-learn; invalid
  combinations will raise the upstream errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


@dataclass
class LogisticRegressionConfig:
    penalty: str = "l2"
    C: float = 1.0
    fit_intercept: bool = True
    class_weight: Optional[str] = None
    solver: str = "lbfgs"
    max_iter: int = 100
    tol: float = 1e-4
    random_state: Optional[int] = None
    n_jobs: Optional[int] = None


class LogisticRegressionModel:
    def __init__(self, config: Optional[LogisticRegressionConfig] = None):
        self.config = config or LogisticRegressionConfig()
        self.model = LogisticRegression(
            penalty=self.config.penalty,
            C=self.config.C,
            fit_intercept=self.config.fit_intercept,
            class_weight=self.config.class_weight,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
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
        self._check_fitted()
        return self.model.predict_proba(X)

    def decision_function(self, X: Any):
        self._check_fitted()
        return self.model.decision_function(X)

    def score(self, X: Any, y: Any, sample_weight: Any = None):
        self._check_fitted()
        return self.model.score(X, y, sample_weight=sample_weight)

    def evaluate_binary(self, X: Any, y: Any, sample_weight: Any = None):
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        positive = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
        metrics = {
            "accuracy": accuracy_score(y, y_pred, sample_weight=sample_weight),
        }
        try:
            metrics["auroc"] = roc_auc_score(y, positive, sample_weight=sample_weight)
        except ValueError:
            metrics["auroc"] = float("nan")
        try:
            metrics["auprc"] = average_precision_score(y, positive, sample_weight=sample_weight)
        except ValueError:
            metrics["auprc"] = float("nan")
        return metrics

    def get_params(self, deep: bool = True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionModel must be fitted before use.")


def build_model(**kwargs) -> LogisticRegressionModel:
    return LogisticRegressionModel(LogisticRegressionConfig(**kwargs))


__all__ = ["LogisticRegressionConfig", "LogisticRegressionModel", "build_model"]
