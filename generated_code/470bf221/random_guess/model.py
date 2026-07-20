"""Random-guess baseline classifier.

Implements sklearn.dummy.DummyClassifier with the stratified strategy, which
samples classes according to the empirical training distribution.
"""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier


class RandomGuessModel:
    def __init__(self, random_state: int | None = None, **kwargs: Any) -> None:
        self.model = DummyClassifier(strategy="stratified", random_state=random_state, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


Model = RandomGuessModel
