"""Random forest classifier for EHRSHOT-style benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


@dataclass
class RandomForestModel:
    random_state: Optional[int] = None
    n_estimators: int = 200
    max_depth: Optional[int] = None
    class_weight: Optional[str] = None
    n_jobs: Optional[int] = None

    def __post_init__(self) -> None:
        self.model = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: Any, y: Any) -> "RandomForestModel":
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


Model = RandomForestModel
