"""Logistic regression baseline for EHRSHOT-style benchmarking.

This module wraps scikit-learn's ``LogisticRegression`` with a small API for
training, scoring, prediction, and evaluation.

Assumptions
-----------
- Inputs are already numerically encoded and imputed.
- Binary labels are encoded as 0/1.
- The default solver/penalty is left to scikit-learn unless explicitly provided.
- For binary classification, `predict_scores` returns the positive-class
  probability when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.linear_model import LogisticRegression

ArrayLike = Union[np.ndarray, list]


@dataclass
class LogisticRegressionConfig:
    penalty: str = "l2"
    dual: bool = False
    tol: float = 1e-4
    C: float = 1.0
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    class_weight: Optional[Union[Dict[Any, float], str]] = None
    random_state: Optional[int] = None
    solver: str = "lbfgs"
    max_iter: int = 100
    multi_class: str = "auto"
    verbose: int = 0
    warm_start: bool = False
    n_jobs: Optional[int] = None
    l1_ratio: Optional[float] = None
    threshold: float = 0.5


class LogisticRegressionModel:
    def __init__(self, config: Optional[LogisticRegressionConfig] = None):
        self.config = config or LogisticRegressionConfig()
        self.model = LogisticRegression(
            penalty=self.config.penalty,
            dual=self.config.dual,
            tol=self.config.tol,
            C=self.config.C,
            fit_intercept=self.config.fit_intercept,
            intercept_scaling=self.config.intercept_scaling,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            multi_class=self.config.multi_class,
            verbose=self.config.verbose,
            warm_start=self.config.warm_start,
            n_jobs=self.config.n_jobs,
            l1_ratio=self.config.l1_ratio,
        )
        self.is_fitted_ = False

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegressionModel":
        self.model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict_scores(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return np.asarray(proba[:, 1], dtype=float)
            return np.asarray(proba).reshape(-1).astype(float)
        decision = self.model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-np.asarray(decision, dtype=float)))

    def predict(self, X: ArrayLike) -> np.ndarray:
        scores = self.predict_scores(X)
        return (scores >= self.config.threshold).astype(int)

    def evaluate(self, X: ArrayLike, y: ArrayLike) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

        y_true = np.asarray(y)
        scores = self.predict_scores(X)
        preds = (scores >= self.config.threshold).astype(int)
        scores = np.clip(scores, 1e-12, 1 - 1e-12)
        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "log_loss": float(log_loss(y_true, np.column_stack([1 - scores, scores]))),
        }
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            metrics["auroc"] = float("nan")
        return metrics

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params(deep=True)

    def set_params(self, **params: Any) -> "LogisticRegressionModel":
        self.model.set_params(**params)
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("LogisticRegressionModel must be fitted before prediction.")


def build_model(**kwargs: Any) -> LogisticRegressionModel:
    config = LogisticRegressionConfig(**kwargs)
    return LogisticRegressionModel(config=config)


__all__ = ["LogisticRegressionConfig", "LogisticRegressionModel", "build_model"]
