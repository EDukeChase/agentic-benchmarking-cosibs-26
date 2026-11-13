"""Compare a re-executed run with the numbers that were reported (the numbers half of a re-run check).

The re-execution script runs a finished run's own `reproduce.py` again and hands the new `results.json`
and `predictions.csv` to `compare`. Nothing here runs that code or reads patient-level data beyond the
predictions files (patient id, label, probability).

Tolerance rule, read from one config file (scripts/reexec_tolerances.json) so it can change without re-coding:
- AUROC and Brier score: absolute difference at most `auroc_abs` / `brier_abs`.
- F1: recomputed from the re-executed probabilities at the threshold that was reported; the check is the
  number of test patients whose predicted-positive flag differs from the reported run (at most
  `f1_patients_flag_differs`). The F1 gap and the four counts are stored as well.
Labels used here: "none" (within tolerance), "a" (the method has no re-executed output, or the run did not
run), "b" (it ran, numbers differ beyond tolerance).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .reexec_helpers import guess_failure_cause, package_differences  # noqa: F401  (re-exported)

DEFAULT_TOLERANCES = {"auroc_abs": 0.001, "brier_abs": 0.001, "f1_patients_flag_differs": 1}
PREDICTION_COLUMNS = ["patient_id", "method", "split", "y_true", "y_prob"]


def load_tolerances(path: str | Path | None) -> dict:
    if path is None:
        return dict(DEFAULT_TOLERANCES)
    return {**DEFAULT_TOLERANCES, **json.loads(Path(path).read_text())}


def read_predictions(path: str | Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path, dtype={"patient_id": str, "method": str, "split": str})
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    return frame if set(PREDICTION_COLUMNS) <= set(frame.columns) else None


def _f1_counts(y_true: np.ndarray, flags: np.ndarray) -> dict:
    tp = int(((flags == 1) & (y_true == 1)).sum())
    fp = int(((flags == 1) & (y_true == 0)).sum())
    fn = int(((flags == 0) & (y_true == 1)).sum())
    tn = int(((flags == 0) & (y_true == 0)).sum())
    denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": (2 * tp / denominator) if denominator else None}


def _metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float | None) -> dict:
    out = {"auroc": None, "brier": None, "f1": None, "counts": None}
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, prob))
    out["brier"] = float(brier_score_loss(y_true, prob))
    if threshold is not None:
        counts = _f1_counts(y_true, (prob >= threshold).astype(int))
        out["f1"], out["counts"] = counts["f1"], {k: counts[k] for k in ("tp", "fp", "fn", "tn")}
    return out


def _diff(new: float | None, old: float | None) -> float | None:
    return None if new is None or old is None else float(new - old)


def compare_method(name: str, reported: dict, old: pd.DataFrame | None, new: pd.DataFrame | None, tolerances: dict) -> dict:
    """One method: reported numbers and original predictions against the re-executed predictions."""
    rep_metrics = (reported or {}).get("test_metrics") or {}
    threshold = (reported or {}).get("threshold")
    entry = {
        "name_reported": name,
        "reported": {"auroc": rep_metrics.get("auroc"), "brier": rep_metrics.get("brier"), "f1": rep_metrics.get("f1"), "threshold": threshold},
        "reexecuted": {"auroc": None, "brier": None, "f1": None, "threshold": threshold},
        "auroc_diff": None, "brier_diff": None, "f1_gap": None, "patients_flag_differs": None,
        "max_abs_probability_diff": None, "test_patients_compared": None, "counts": None,
        "within_tolerance": None, "taxonomy": None, "note": None,
    }
    if (reported or {}).get("status") != "ok":
        entry["note"] = "method was reported as failed; nothing to compare"
        return entry
    new_rows = None if new is None else new[(new["method"] == name) & (new["split"] == "test")]
    if new_rows is None or new_rows.empty:
        entry.update(taxonomy="a", within_tolerance=False, note="no re-executed test predictions for this method")
        return entry
    new_rows = new_rows.drop_duplicates("patient_id").set_index("patient_id")
    y_true, prob = new_rows["y_true"].to_numpy(dtype=int), new_rows["y_prob"].to_numpy(dtype=float)
    got = _metrics(y_true, prob, threshold)
    entry["reexecuted"].update(auroc=got["auroc"], brier=got["brier"], f1=got["f1"])
    entry["counts"], entry["test_patients_compared"] = got["counts"], int(len(new_rows))
    entry["auroc_diff"] = _diff(got["auroc"], rep_metrics.get("auroc"))
    entry["brier_diff"] = _diff(got["brier"], rep_metrics.get("brier"))
    entry["f1_gap"] = _diff(got["f1"], rep_metrics.get("f1"))

    flags_differ = None
    if old is not None and threshold is not None:
        old_rows = old[(old["method"] == name) & (old["split"] == "test")].drop_duplicates("patient_id").set_index("patient_id")
        joined = old_rows[["y_prob"]].join(new_rows[["y_prob"]], lsuffix="_old", rsuffix="_new", how="outer")
        missing = int(joined.isna().any(axis=1).sum())
        both = joined.dropna()
        flags_differ = int(((both["y_prob_old"] >= threshold) != (both["y_prob_new"] >= threshold)).sum()) + missing
        entry["max_abs_probability_diff"] = float((both["y_prob_old"] - both["y_prob_new"]).abs().max()) if len(both) else None
    entry["patients_flag_differs"] = flags_differ

    problems = []
    for key, limit in (("auroc_diff", tolerances["auroc_abs"]), ("brier_diff", tolerances["brier_abs"])):
        if entry[key] is None or abs(entry[key]) > limit:
            problems.append(key)
    if flags_differ is None or flags_differ > tolerances["f1_patients_flag_differs"]:
        problems.append("patients_flag_differs")
    entry["within_tolerance"] = not problems
    entry["taxonomy"] = "none" if not problems else "b"
    if problems:
        entry["note"] = "beyond tolerance: " + ", ".join(problems)
    return entry


def compare(reported_results: dict | None, old_predictions: pd.DataFrame | None, new_predictions: pd.DataFrame | None,
            tolerances: dict | None = None) -> list[dict]:
    """All methods of the reported run, in the order the agent reported them."""
    tolerances = tolerances or DEFAULT_TOLERANCES
    methods = (reported_results or {}).get("methods") or {}
    return [compare_method(name, methods[name], old_predictions, new_predictions, tolerances) for name in methods]
