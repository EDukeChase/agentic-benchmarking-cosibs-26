"""Support vector machine classifier for EHRSHOT-style benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.svm import SVC


@dataclass
class SupportVectorMachineModel:
    random_state: Optional[int] = None
    C: float = 1.0
    kernel: str = "rbf"
    class_weight: Optional[str] = None
    probability: bool = True

    def __post_init__(self) -> None:
        self.model = SVC(
            random_state=self.random_state,
            C=self.C,
            kernel=self.kernel,
            class_weight=self.class_weight,
            probability=self.probability,
        )

    def fit(self, X: Any, y: Any) -> "SupportVectorMachineModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.probability:
            raise ValueError("predict_proba requires probability=True")
        return self.model.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: Any, y: Any) -> dict:
        y_true = np.asarray(y)
        if self.probability:
            proba = self.predict_proba(X)[:, 1]
        else:
            scores = self.model.decision_function(X)
            proba = 1.0 / (1.0 + np.exp(-scores))
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


Model = SupportVectorMachineModel
