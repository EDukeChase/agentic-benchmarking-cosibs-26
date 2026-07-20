"""Logistic regression baseline for EHRSHOT-style benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


@dataclass
class LogisticRegressionModel:
    random_state: Optional[int] = None
    class_weight: Optional[str] = None
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 1000

    def __post_init__(self) -> None:
        self.model = LogisticRegression(
            random_state=self.random_state,
            class_weight=self.class_weight,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
        )

    def fit(self, X: Any, y: Any) -> "LogisticRegressionModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: Any, y: Any) -> dict:
        y_true = np.asarray(y)
        proba = self.predict_proba(X)[:, 1]
        pred = self.predict(X)
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


Model = LogisticRegressionModel
