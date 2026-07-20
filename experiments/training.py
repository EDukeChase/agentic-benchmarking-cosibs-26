"""Training loop for the celiac disease logistic regression baseline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from config import LogisticRegressionConfig
from evaluation import evaluate_binary_classifier
from model import build_model


def train_model(
    X_train,
    y_train,
    config: Optional[LogisticRegressionConfig] = None,
) -> BaseEstimator:
    """Fit the logistic regression pipeline on training data."""

    if config is None:
        config = LogisticRegressionConfig()
    model = build_model(config)
    model.fit(X_train, y_train)
    return model


def predict(model: BaseEstimator, X, threshold: float = 0.5) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate labels and probabilities from a fitted model."""

    if hasattr(model, "predict_proba"):
        proba_matrix = model.predict_proba(X)
        classes = getattr(model, "classes_", None)
        if classes is not None and 1 in classes:
            positive_idx = int(np.where(classes == 1)[0][0])
        else:
            positive_idx = 1 if proba_matrix.shape[1] > 1 else 0
        proba = proba_matrix[:, positive_idx]
        pred = (proba >= threshold).astype(int)
        return pred, proba
    pred = model.predict(X)
    return pred, None


def train_and_evaluate(
    X_train,
    y_train,
    X_val,
    y_val,
    config: Optional[LogisticRegressionConfig] = None,
) -> Dict[str, Any]:
    """Fit model and evaluate on validation data."""

    if config is None:
        config = LogisticRegressionConfig()
    model = train_model(X_train, y_train, config=config)
    y_pred, y_proba = predict(model, X_val, threshold=config.threshold)
    metrics = evaluate_binary_classifier(y_val, y_pred, y_proba)
    return {"config": asdict(config), "metrics": metrics, "model": model}

