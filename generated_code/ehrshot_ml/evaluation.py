from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_binary_classification(y_true, y_prob) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    metrics = {"auroc": float("nan"), "auprc": float("nan")}
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    return metrics
