"""Logistic regression baseline for EHRSHOT-style benchmarking.

This module wraps ``sklearn.linear_model.LogisticRegression`` in a small
importable interface suitable for benchmark code.

Missing documentation assumptions:
- Features are expected to be numeric and already preprocessed.
- The literature review does not specify a solver or regularization strength,
  so scikit-learn defaults are used unless overridden.
- Class weighting is optional and exposed directly via the config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


@dataclass
class LogisticRegressionConfig:
    penalty: str = "l2"
    dual: bool = False
    tol: float = 1e-4
    C: float = 1.0
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    class_weight: Optional[Dict[int, float] | str] = None
    random_state: Optional[int] = None
    solver: str = "lbfgs"
    max_iter: int = 100
    multi_class: str = "auto"
    verbose: int = 0
    warm_start: bool = False
    n_jobs: Optional[int] = None
    l1_ratio: Optional[float] = None


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

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model.predict_proba(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model.decision_function(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        probs = self.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] == 2:
            pos_probs = probs[:, 1]
        else:
            pos_probs = probs.ravel()
        preds = self.predict(X)
        scores = {"accuracy": float(accuracy_score(y, preds))}
        try:
            scores["auroc"] = float(roc_auc_score(y, pos_probs))
        except ValueError:
            scores["auroc"] = float("nan")
        try:
            scores["auprc"] = float(average_precision_score(y, pos_probs))
        except ValueError:
            scores["auprc"] = float("nan")
        return scores

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params(deep=True)

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("LogisticRegressionModel must be fitted before use.")


__all__ = ["LogisticRegressionConfig", "LogisticRegressionModel"]
