"""Evaluation utilities for binary classification tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class BinaryClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: Optional[float]
    auprc: Optional[float]
    brier: Optional[float]
    logloss: Optional[float]


def _safe_metric(fn, y_true, y_score):
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return None


def evaluate_binary_classifier(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute standard binary classification metrics."""

    metrics = BinaryClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auroc=_safe_metric(roc_auc_score, y_true, y_proba) if y_proba is not None else None,
        auprc=_safe_metric(average_precision_score, y_true, y_proba) if y_proba is not None else None,
        brier=_safe_metric(brier_score_loss, y_true, y_proba) if y_proba is not None else None,
        logloss=_safe_metric(lambda yt, yp: log_loss(yt, np.column_stack([1 - yp, yp])), y_true, y_proba)
        if y_proba is not None
        else None,
    )
    return metrics.__dict__

