"""Gradient boosting baseline for binary classification.

Uses HistGradientBoostingClassifier, which is efficient for tabular data.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier


class GradientBoostingModel:
    def __init__(self, random_state: int | None = None, max_iter: int = 200, **kwargs: Any) -> None:
        self.model = HistGradientBoostingClassifier(
            random_state=random_state,
            max_iter=max_iter,
            **kwargs,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


Model = GradientBoostingModel
