"""Logistic regression baseline for binary classification.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self, random_state: int | None = None, max_iter: int = 1000, **kwargs: Any) -> None:
        self.model = LogisticRegression(random_state=random_state, max_iter=max_iter, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


Model = LogisticRegressionModel
