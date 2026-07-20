"""Gradient boosting classifier for EHRSHOT-style benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


@dataclass
class GradientBoostingModel:
    random_state: Optional[int] = None
    learning_rate: float = 0.1
    max_iter: int = 100
    max_depth: Optional[int] = None
    max_leaf_nodes: int = 31
    class_weight: Optional[str] = None

    def __post_init__(self) -> None:
        self.model = HistGradientBoostingClassifier(
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            class_weight=self.class_weight,
        )

    def fit(self, X: Any, y: Any) -> "GradientBoostingModel":
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


Model = GradientBoostingModel
