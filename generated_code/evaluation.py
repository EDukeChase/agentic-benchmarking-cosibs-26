"""Evaluation metrics for binary EHR prediction tasks."""

from __future__ import annotations

from typing import Dict

import torch


def binary_auc_pr(y_true: torch.Tensor, y_score: torch.Tensor) -> Dict[str, float]:
    """Compute simple threshold-free metrics if sklearn is available.

    Falls back to a minimal implementation if sklearn is unavailable.
    """
    y_true = y_true.detach().cpu().float().numpy()
    y_score = y_score.detach().cpu().float().numpy()
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        return {
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score)),
        }
    except Exception:
        # Minimal fallback: return ranking-based approximations.
        order = y_score.argsort()
        y_true_sorted = y_true[order]
        cum_pos = y_true_sorted.cumsum()
        total_pos = y_true_sorted.sum()
        total_neg = len(y_true_sorted) - total_pos
        tpr = cum_pos / max(total_pos, 1)
        fpr = (torch.arange(len(y_true_sorted)) + 1 - cum_pos) / max(total_neg, 1)
        auroc = float(torch.trapz(torch.tensor(tpr), torch.tensor(fpr)).abs())
        precision = cum_pos / (torch.arange(len(y_true_sorted)) + 1)
        recall = tpr
        auprc = float(torch.trapz(torch.tensor(precision), torch.tensor(recall)).abs())
        return {"auroc": auroc, "auprc": auprc}
