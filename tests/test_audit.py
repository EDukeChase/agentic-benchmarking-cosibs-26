"""Checks that the run audit accepts a clean run and catches specific faults.

Builds a small synthetic study (150 patients, 25% events), writes a clean run
for it, then breaks the run one way at a time. Run with:
    python -m unittest discover -s tests
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.audit import SPECIFIED, audit_run, compute_test_metrics
from src.evaluation.cohort import build_cohort

IHD = "414545008"
N = 150


def split_of(i: int) -> str:
    return "train" if i % 10 <= 5 else ("validation" if i % 10 <= 7 else "test")


def write_data(folder: Path) -> None:
    ids = [f"p{i:03d}" for i in range(N)]
    pd.DataFrame({"Id": ids, "BIRTHDATE": "1960-01-01", "DEATHDATE": None}).to_csv(folder / "patients.csv", index=False)
    events = [(("2020-06-01", pid, IHD)) for i, pid in enumerate(ids) if i % 4 == 0]
    pd.DataFrame(events, columns=["START", "PATIENT", "CODE"]).to_csv(folder / "conditions.csv", index=False)
    pd.DataFrame({"patient_id": ids, "split": [split_of(i) for i in range(N)]}).to_csv(folder / "patient_split.csv", index=False)


def method_entry(name: str, threshold: float, source=None) -> dict:
    return {
        "status": "ok", "attempts": 1, "description": f"{name} description", "failure_reason": None,
        "source": source, "threshold": threshold,
        "threshold_chosen_on": "train outcome rate" if name == SPECIFIED else "validation",
        "test_metrics": None,
    }


def write_run(run: Path, data: Path) -> None:
    cohort = build_cohort(data).merge(pd.read_csv(data / "patient_split.csv", dtype=str), on="patient_id")
    rng = np.random.default_rng(0)
    train_rate = float(cohort.loc[cohort["split"] == "train", "y"].mean())
    names = ["logistic_regression", SPECIFIED, "gradient_boosting", "tabpfn", "mlp"]
    ok = names[:4]
    methods, frames = {}, []
    for name in ok:
        prob = np.clip(0.15 + 0.35 * cohort["y"].to_numpy() + rng.normal(0, 0.2, len(cohort)), 0.01, 0.99)
        threshold = train_rate if name == SPECIFIED else 0.3
        entry = method_entry(name, threshold, source=None if name in ("logistic_regression", SPECIFIED) else "https://example.org/method")
        test = (cohort["split"] == "test").to_numpy()
        entry["test_metrics"] = compute_test_metrics(cohort["y"][test], prob[test], threshold)
        methods[name] = entry
        keep = cohort["split"].isin(["validation", "test"]).to_numpy()
        frames.append(pd.DataFrame({"patient_id": cohort["patient_id"][keep], "method": name, "split": cohort["split"][keep],
                                    "y_true": cohort["y"][keep], "y_prob": prob[keep]}))
    methods["mlp"] = {
        "status": "failed", "attempts": 2, "description": "small neural network", "failure_reason": "out of memory twice",
        "source": "https://example.org/mlp", "threshold": None, "threshold_chosen_on": None, "test_metrics": None,
    }
    counts = {s: {"patients": int((cohort["split"] == s).sum()), "events": int(cohort.loc[cohort["split"] == s, "y"].sum())}
              for s in ("train", "validation", "test")}
    (run / "results.json").write_text(json.dumps({"seed": 5960, "cohort": counts, "methods": methods}, indent=2))
    pd.concat(frames).to_csv(run / "predictions.csv", index=False)
    cohort[["patient_id", "split"]].to_csv(run / "splits.csv", index=False)
    ids = {s: cohort.loc[cohort["split"] == s, "patient_id"].tolist() for s in ("train", "validation", "test")}
    usage = {name: {"fit": ids["train"], "choices": ids["train"] + ids["validation"], "test": ids["test"]} for name in ok}
    (run / "split_usage.json").write_text(json.dumps(usage))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data, self.run = Path(self.tmp.name) / "data", Path(self.tmp.name) / "run"
        self.data.mkdir()
        self.run.mkdir()
        write_data(self.data)
        write_run(self.run, self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def audit(self):
        return audit_run(self.run, self.data)

    def failures(self):
        return [c.name for c in self.audit().failed]

    def edit_results(self, change):
        path = self.run / "results.json"
        results = json.loads(path.read_text())
        change(results)
        path.write_text(json.dumps(results))

    def edit_predictions(self, change):
        path = self.run / "predictions.csv"
        preds = pd.read_csv(path, dtype={"patient_id": str})
        path.write_text(change(preds).to_csv(index=False))

    def assertFails(self, fragment: str):
        failures = self.failures()
        self.assertTrue(any(fragment in name for name in failures), f"expected a failure mentioning {fragment!r}, got {failures}")

    def test_clean_run_passes(self):
        report = self.audit()
        self.assertEqual(report.failed, [], [(c.name, c.detail) for c in report.failed])
        self.assertTrue(report.passed)

    def test_tampered_metric_is_caught(self):
        self.edit_results(lambda r: r["methods"]["gradient_boosting"]["test_metrics"].update(auroc=0.99))
        self.assertFails("gradient_boosting: reported auroc")

    def test_f1_depends_on_the_stated_threshold(self):
        self.edit_results(lambda r: r["methods"]["tabpfn"].update(threshold=0.5))
        self.assertFails("tabpfn: reported f1")

    def test_wrong_y_true_is_caught(self):
        def flip(preds):
            preds.loc[(preds["method"] == "tabpfn").idxmax(), "y_true"] ^= 1
            return preds
        self.edit_predictions(flip)
        self.assertFails("tabpfn: y_true matches")

    def test_missing_patient_row_is_caught(self):
        self.edit_predictions(lambda p: p.drop(p[p["method"] == "tabpfn"].index[0]))
        self.assertFails("tabpfn: predictions cover exactly")

    def test_train_patient_in_predictions_is_caught(self):
        def add_train_row(preds):
            row = preds[preds["method"] == "tabpfn"].iloc[[0]].copy()
            row["patient_id"] = "p000"  # a train patient
            return pd.concat([preds, row])
        self.edit_predictions(add_train_row)
        self.assertFails("tabpfn: predictions cover exactly")

    def test_wrong_split_label_is_caught(self):
        def relabel(preds):
            idx = preds[(preds["method"] == "tabpfn") & (preds["split"] == "test")].index[0]
            preds.loc[idx, "split"] = "validation"
            return preds
        self.edit_predictions(relabel)
        self.assertFails("tabpfn: split labels match")

    def test_specified_threshold_must_be_the_train_outcome_rate(self):
        self.edit_results(lambda r: r["methods"][SPECIFIED].update(threshold=0.4))
        self.assertFails("threshold equals the train outcome rate")

    def test_cohort_counts_must_match_the_rebuilt_cohort(self):
        self.edit_results(lambda r: r["cohort"]["test"].update(events=0))
        self.assertFails("cohort test: counts match")

    def test_missing_key_is_caught(self):
        self.edit_results(lambda r: r["methods"]["tabpfn"].pop("attempts"))
        self.assertFails("tabpfn: has every required key")

    def test_failed_method_must_have_null_metrics(self):
        self.edit_results(lambda r: r["methods"]["mlp"].update(threshold=0.3))
        self.assertFails("mlp: metrics, threshold and threshold_chosen_on are null")

    def test_free_choice_method_needs_a_source_url(self):
        self.edit_results(lambda r: r["methods"]["tabpfn"].update(source=None))
        self.assertFails("tabpfn: source is a URL")

    def test_prediction_rows_for_a_failed_method_are_caught(self):
        def add_failed_rows(preds):
            rows = preds[preds["method"] == "tabpfn"].copy()
            rows["method"] = "mlp"
            return pd.concat([preds, rows])
        self.edit_predictions(add_failed_rows)
        self.assertFails("rows for exactly the methods marked ok")

    def test_wrong_splits_file_is_caught(self):
        splits = pd.read_csv(self.run / "splits.csv", dtype=str)
        splits.loc[0, "split"] = "test"
        splits.to_csv(self.run / "splits.csv", index=False)
        self.assertFails("splits.csv: labels match")

    def test_missing_deliverable_is_reported_without_crashing(self):
        (self.run / "predictions.csv").unlink()
        self.assertFails("predictions.csv: present")

    def test_invalid_json_is_reported_without_crashing(self):
        (self.run / "results.json").write_text("{not json")
        self.assertFails("results.json: is valid JSON")

    def edit_usage(self, change):
        path = self.run / "split_usage.json"
        usage = json.loads(path.read_text())
        change(usage)
        path.write_text(json.dumps(usage))

    def test_fit_on_a_validation_patient_is_caught(self):
        def leak(usage):
            validation_id = usage["tabpfn"]["choices"][-1]  # the last choice patient is a validation patient
            usage["tabpfn"]["fit"].append(validation_id)
        self.edit_usage(leak)
        self.assertFails("tabpfn: fit patients are all train patients")

    def test_choices_using_a_test_patient_are_caught(self):
        self.edit_usage(lambda u: u["tabpfn"]["choices"].append(u["tabpfn"]["test"][0]))
        self.assertFails("tabpfn: patients used for choices are train or validation patients")

    def test_scoring_a_subset_of_the_test_patients_is_caught(self):
        self.edit_usage(lambda u: u["tabpfn"].update(test=u["tabpfn"]["test"][1:]))
        self.assertFails("tabpfn: test metrics use exactly the test patients")

    def test_missing_split_usage_entry_or_file_is_caught(self):
        self.edit_usage(lambda u: u.pop("tabpfn"))
        self.assertFails("split_usage.json: has an entry for exactly the methods marked ok")
        (self.run / "split_usage.json").unlink()
        self.assertFails("split_usage.json: present and valid JSON")

    def test_metric_helper_matches_a_hand_computation(self):
        # four patients, perfect ranking; threshold 0.5 calls the top two positive
        metrics = compute_test_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], 0.5)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["brier"], (0.01 + 0.16 + 0.16 + 0.01) / 4)
        self.assertEqual(metrics["f1"], 1.0)
        # a probability exactly at the threshold counts as positive
        self.assertEqual(compute_test_metrics([0, 1], [0.2, 0.5], 0.5)["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
