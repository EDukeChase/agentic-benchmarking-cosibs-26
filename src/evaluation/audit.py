"""Format validation and consistency audit of one run's deliverables.

Works on any run's output folder, whichever arm produced it. Two layers:

1. Format: results.json, predictions.csv and splits.csv follow the layout the
   study prompt prescribes.
2. Consistency: what the run reports agrees with what the raw data says. The
   cohort, outcome and split are rebuilt from the data folder, and the reported
   metrics are recomputed from predictions.csv.

The audit never trusts the run: y_true, the cohort counts and the specified
model's threshold are all recomputed independently. It does not rerun the
run's own code (reproduce.py / score.py); that is a separate step.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

from .cohort import build_cohort

SEED = 5960
SPLITS = ("train", "validation", "test")
PREDICTION_SPLITS = ("validation", "test")
METRICS = ("auroc", "brier", "f1")
SPECIFIED = "logistic_regression_specified"
REQUIRED_METHODS = ("logistic_regression", SPECIFIED)
N_METHODS = 5
METHOD_KEYS = {
    "status", "attempts", "description", "failure_reason", "source",
    "threshold", "threshold_chosen_on", "test_metrics",
}
PREDICTION_COLUMNS = {"patient_id", "method", "split", "y_true", "y_prob"}
DEFAULT_TOLERANCE = 1e-3


@dataclass
class Check:
    name: str
    status: str  # "pass", "fail" or "warn"
    detail: str = ""


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, ok: bool, detail: str = "", warn: bool = False) -> bool:
        status = "pass" if ok else ("warn" if warn else "fail")
        self.checks.append(Check(name, status, detail))
        return ok

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def passed(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict:
        counts = {s: sum(c.status == s for c in self.checks) for s in ("pass", "warn", "fail")}
        return {"passed": self.passed, "counts": counts, "checks": [asdict(c) for c in self.checks]}


def compute_test_metrics(y_true, y_prob, threshold: float) -> dict[str, float]:
    """AUROC, Brier score, and F1 with a patient predicted positive when y_prob >= threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    predicted = (y_prob >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_count(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


# ---------------------------------------------------------------- format layer

def check_results_format(results: dict, report: Report) -> None:
    report.add("results.json: seed is 5960", results.get("seed") == SEED, f"seed={results.get('seed')!r}")

    cohort = results.get("cohort")
    cohort_ok = (
        isinstance(cohort, dict)
        and set(cohort) == set(SPLITS)
        and all(
            isinstance(cohort[s], dict)
            and set(cohort[s]) == {"patients", "events"}
            and all(_is_count(v) for v in cohort[s].values())
            for s in SPLITS
        )
    )
    report.add("results.json: cohort block has patient and event counts for each split", cohort_ok)

    methods = results.get("methods")
    if not isinstance(methods, dict):
        report.add("results.json: methods is an object", False)
        return
    report.add(f"results.json: {N_METHODS} method entries", len(methods) == N_METHODS, f"found {len(methods)}")
    missing = [m for m in REQUIRED_METHODS if m not in methods]
    report.add("results.json: both logistic regression methods are present", not missing, f"missing {missing}" if missing else "")
    for name, method in methods.items():
        _check_method_format(name, method, report)
    n_ok = sum(isinstance(m, dict) and m.get("status") == "ok" for m in methods.values())
    report.add("results.json: at least one method ran, so the run is valid", n_ok >= 1, f"{n_ok} method(s) ok")


def _check_method_format(name: str, method, report: Report) -> None:
    p = f"method {name}"
    if not isinstance(method, dict):
        report.add(f"{p}: entry is an object", False)
        return
    keys = set(method)
    report.add(f"{p}: has every required key", METHOD_KEYS <= keys, f"missing {sorted(METHOD_KEYS - keys)}" if METHOD_KEYS - keys else "")
    report.add(f"{p}: has no extra keys", keys <= METHOD_KEYS, f"extra {sorted(keys - METHOD_KEYS)}" if keys - METHOD_KEYS else "", warn=True)

    status, attempts = method.get("status"), method.get("attempts")
    report.add(f"{p}: status is ok or failed", status in ("ok", "failed"), f"status={status!r}")
    report.add(f"{p}: attempts is 1 or 2", attempts in (1, 2) and not isinstance(attempts, bool), f"attempts={attempts!r}")
    if status == "failed":
        report.add(f"{p}: a failed method used its retry (attempts is 2)", attempts == 2, f"attempts={attempts!r}", warn=True)
    description = method.get("description")
    report.add(f"{p}: description is a non-empty string", isinstance(description, str) and description.strip() != "")

    source = method.get("source")
    if name in REQUIRED_METHODS:
        report.add(f"{p}: source is null", source is None, f"source={source!r}")
    else:
        report.add(f"{p}: source is a URL", isinstance(source, str) and source.startswith(("http://", "https://")), f"source={source!r}")

    metrics, threshold, chosen = method.get("test_metrics"), method.get("threshold"), method.get("threshold_chosen_on")
    if status == "ok":
        report.add(f"{p}: failure_reason is null", method.get("failure_reason") is None, warn=True)
        metrics_ok = (
            isinstance(metrics, dict)
            and set(metrics) == set(METRICS)
            and all(_is_number(metrics[k]) and 0 <= metrics[k] <= 1 for k in METRICS)
        )
        report.add(f"{p}: test_metrics are numbers in [0, 1]", metrics_ok, f"test_metrics={metrics!r}")
        report.add(f"{p}: threshold is a number in [0, 1]", _is_number(threshold) and 0 <= threshold <= 1, f"threshold={threshold!r}")
        expected = "train outcome rate" if name == SPECIFIED else "validation"
        report.add(f"{p}: threshold_chosen_on is {expected!r}", chosen == expected, f"got {chosen!r}")
    elif status == "failed":
        reason = method.get("failure_reason")
        report.add(f"{p}: failure_reason is given", isinstance(reason, str) and reason.strip() != "")
        metrics_null = metrics is None or (isinstance(metrics, dict) and all(v is None for v in metrics.values()))
        report.add(f"{p}: metrics, threshold and threshold_chosen_on are null", metrics_null and threshold is None and chosen is None)


def check_predictions_format(preds: pd.DataFrame, ok_methods: set[str], report: Report) -> bool:
    """Returns True when the columns are right, so the consistency layer can use the frame."""
    columns_ok = set(preds.columns) == PREDICTION_COLUMNS
    report.add("predictions.csv: columns are patient_id, method, split, y_true, y_prob", columns_ok, f"found {list(preds.columns)}")
    if not columns_ok:
        return False
    report.add("predictions.csv: split is validation or test", preds["split"].isin(PREDICTION_SPLITS).all(),
               f"values {sorted(preds['split'].dropna().unique())}")
    y_true = pd.to_numeric(preds["y_true"], errors="coerce")
    report.add("predictions.csv: y_true is 0 or 1", y_true.isin([0, 1]).all())
    y_prob = pd.to_numeric(preds["y_prob"], errors="coerce")
    report.add("predictions.csv: y_prob is a number in [0, 1], no missing values", bool(y_prob.between(0, 1).all()))
    duplicates = int(preds.duplicated(["patient_id", "method"]).sum())
    report.add("predictions.csv: one row per patient per method", duplicates == 0, f"{duplicates} duplicate rows" if duplicates else "")
    present = set(preds["method"].dropna().unique())
    report.add("predictions.csv: has rows for exactly the methods marked ok", present == ok_methods,
               f"only in predictions {sorted(present - ok_methods)}, ok but absent {sorted(ok_methods - present)}")
    return True


# ----------------------------------------------------------- consistency layer

def check_consistency(results: dict, preds: pd.DataFrame, splits: pd.DataFrame | None,
                      cohort: pd.DataFrame, tolerance: float, report: Report) -> None:
    """cohort has columns patient_id, y and split, rebuilt from the raw data and the provided split."""
    _check_cohort_counts(results, cohort, report)
    _check_splits_file(splits, cohort, report)

    expected = cohort[cohort["split"].isin(PREDICTION_SPLITS)].rename(columns={"split": "split_true", "y": "y_expected"})
    methods = results.get("methods", {})
    for name, method in methods.items():
        if not (isinstance(method, dict) and method.get("status") == "ok"):
            continue
        p = f"method {name}"
        rows = preds[preds["method"] == name].drop_duplicates("patient_id")
        missing, extra = set(expected["patient_id"]) - set(rows["patient_id"]), set(rows["patient_id"]) - set(expected["patient_id"])
        report.add(f"{p}: predictions cover exactly the validation and test cohort patients", not missing and not extra,
                   f"{len(missing)} missing, {len(extra)} not in the validation/test cohort")
        both = rows.merge(expected, on="patient_id")
        if both.empty:
            continue
        both = both.assign(y_true=pd.to_numeric(both["y_true"], errors="coerce"), y_prob=pd.to_numeric(both["y_prob"], errors="coerce"))
        mislabeled = int((both["split"] != both["split_true"]).sum())
        report.add(f"{p}: split labels match the provided split", mislabeled == 0, f"{mislabeled} patients differ" if mislabeled else "")
        wrong_y = int((both["y_true"] != both["y_expected"]).sum())
        report.add(f"{p}: y_true matches the outcome rebuilt from the raw tables", wrong_y == 0, f"{wrong_y} of {len(both)} differ" if wrong_y else "")
        _check_metrics(name, method, both, tolerance, report)

    specified = methods.get(SPECIFIED)
    if isinstance(specified, dict) and specified.get("status") == "ok" and _is_number(specified.get("threshold")):
        rate = float(cohort.loc[cohort["split"] == "train", "y"].mean())
        report.add(f"method {SPECIFIED}: threshold equals the train outcome rate", abs(specified["threshold"] - rate) <= tolerance,
                   f"reported {specified['threshold']:.6f}, train outcome rate {rate:.6f}")


def check_split_usage(usage, ok_methods: set[str], cohort: pd.DataFrame, report: Report) -> None:
    """split_usage.json: the patients each method says it used to fit, make choices, and score the test metrics.

    This is the agent's own account. Fit patients must all be train patients, choice patients train or
    validation patients, and the test patients exactly the test group.
    """
    if usage is None:
        report.add("split_usage.json: present and valid JSON", False)
        return
    well_formed = isinstance(usage, dict)
    report.add("split_usage.json: is an object keyed by method name", well_formed)
    if not well_formed:
        return
    report.add("split_usage.json: has an entry for exactly the methods marked ok", set(usage) == ok_methods,
               f"missing {sorted(ok_methods - set(usage))}, extra {sorted(set(usage) - ok_methods)}")

    groups = {s: set(cohort.loc[cohort["split"] == s, "patient_id"]) for s in SPLITS}
    for name in sorted(ok_methods & set(usage)):
        entry, p = usage[name], f"method {name}"
        shaped = isinstance(entry, dict) and set(entry) == {"fit", "choices", "test"} and all(
            isinstance(entry[k], list) and all(isinstance(i, str) for i in entry[k]) for k in entry)
        report.add(f"{p}: split_usage entry has fit, choices and test lists of patient IDs", shaped)
        if not shaped:
            continue
        fit, choices, test = set(entry["fit"]), set(entry["choices"]), set(entry["test"])
        report.add(f"{p}: fit patients are all train patients", bool(fit) and fit <= groups["train"],
                   f"{len(fit - groups['train'])} of {len(fit)} are not train patients" if fit else "no fit patients listed")
        outside = choices - groups["train"] - groups["validation"]
        report.add(f"{p}: patients used for choices are train or validation patients", bool(choices) and not outside,
                   f"{len(outside)} are test or unknown patients" if choices else "no choice patients listed")
        report.add(f"{p}: test metrics use exactly the test patients", test == groups["test"],
                   f"{len(groups['test'] - test)} test patients missing, {len(test - groups['test'])} not in the test group")


def _check_cohort_counts(results: dict, cohort: pd.DataFrame, report: Report) -> None:
    reported = results.get("cohort")
    for split in SPLITS:
        part = cohort[cohort["split"] == split]
        try:
            patients, events = reported[split]["patients"], reported[split]["events"]
        except (KeyError, TypeError):
            report.add(f"cohort {split}: counts match the rebuilt cohort", False, "counts missing from results.json")
            continue
        report.add(f"cohort {split}: counts match the rebuilt cohort", patients == len(part) and events == int(part["y"].sum()),
                   f"reported {patients} patients / {events} events, rebuilt {len(part)} / {int(part['y'].sum())}")


def _check_splits_file(splits: pd.DataFrame | None, cohort: pd.DataFrame, report: Report) -> None:
    if splits is None:
        report.add("splits.csv: present", False)
        return
    if set(splits.columns) != {"patient_id", "split"}:
        report.add("splits.csv: columns are patient_id and split", False, f"found {list(splits.columns)}")
        return
    merged = splits.drop_duplicates("patient_id").merge(cohort[["patient_id", "split"]], on="patient_id", how="outer",
                                                        suffixes=("", "_true"), indicator=True)
    only_file, only_cohort = int((merged["_merge"] == "left_only").sum()), int((merged["_merge"] == "right_only").sum())
    report.add("splits.csv: lists exactly the cohort patients", only_file == 0 and only_cohort == 0,
               f"{only_file} not in the cohort, {only_cohort} cohort patients missing")
    labeled = merged[merged["_merge"] == "both"]
    differing = int((labeled["split"] != labeled["split_true"]).sum())
    report.add("splits.csv: labels match the provided split", differing == 0, f"{differing} patients differ" if differing else "")


def _check_metrics(name: str, method: dict, both: pd.DataFrame, tolerance: float, report: Report) -> None:
    p = f"method {name}"
    threshold, reported = method.get("threshold"), method.get("test_metrics")
    if not _is_number(threshold) or not isinstance(reported, dict):
        return
    test = both[both["split_true"] == "test"]
    try:
        recomputed = compute_test_metrics(test["y_expected"], test["y_prob"], threshold)
    except ValueError as error:
        report.add(f"{p}: test metrics can be recomputed from predictions.csv", False, str(error))
        return
    for metric in METRICS:
        value = reported.get(metric)
        if not _is_number(value):
            continue  # already reported by the format layer
        report.add(f"{p}: reported {metric} matches the value recomputed from predictions.csv",
                   abs(value - recomputed[metric]) <= tolerance,
                   f"reported {value:.6f}, recomputed {recomputed[metric]:.6f}")


# ------------------------------------------------------------------- top level

def audit_run(run_dir: str | Path, data_dir: str | Path, tolerance: float = DEFAULT_TOLERANCE) -> Report:
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    report = Report()

    cohort = build_cohort(data_dir)
    provided = pd.read_csv(data_dir / "patient_split.csv", dtype=str)
    cohort = cohort.merge(provided, on="patient_id", how="left")
    unlabeled = int(cohort["split"].isna().sum())
    report.add("data: every cohort patient has a split label", unlabeled == 0, f"{unlabeled} unlabeled" if unlabeled else "")
    cohort = cohort.dropna(subset=["split"])

    results = None
    results_path = run_dir / "results.json"
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text())
            if not isinstance(results, dict):
                raise ValueError("top level is not an object")
        except ValueError as error:
            results = None
            report.add("results.json: is valid JSON", False, str(error))
    else:
        report.add("results.json: present", False)

    preds = None
    if (run_dir / "predictions.csv").exists():
        preds = pd.read_csv(run_dir / "predictions.csv", dtype={"patient_id": str, "method": str, "split": str})
    else:
        report.add("predictions.csv: present", False)
    splits = pd.read_csv(run_dir / "splits.csv", dtype=str) if (run_dir / "splits.csv").exists() else None
    usage = None
    if (run_dir / "split_usage.json").exists():
        try:
            usage = json.loads((run_dir / "split_usage.json").read_text())
        except ValueError:
            usage = None

    if results is None:
        return report
    check_results_format(results, report)
    methods = results.get("methods") if isinstance(results.get("methods"), dict) else {}
    ok_methods = {n for n, m in methods.items() if isinstance(m, dict) and m.get("status") == "ok"}
    if preds is not None and check_predictions_format(preds, ok_methods, report):
        check_consistency(results, preds, splits, cohort, tolerance, report)
    else:
        _check_cohort_counts(results, cohort, report)
        _check_splits_file(splits, cohort, report)
    check_split_usage(usage, ok_methods, cohort, report)
    return report
