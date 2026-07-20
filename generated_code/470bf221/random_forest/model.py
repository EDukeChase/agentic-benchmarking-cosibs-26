"""Random forest baseline for binary classification.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, random_state: int | None = None, n_estimators: int = 200, **kwargs: Any) -> None:
        self.model = RandomForestClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            **kwargs,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


Model = RandomForestModel
