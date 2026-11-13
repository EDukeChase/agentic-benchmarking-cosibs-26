import unittest

import numpy as np
import pandas as pd

from src.evaluation import reexec
from src.evaluation import reexec_helpers  # noqa: F401


def predictions(probabilities, method="m", labels=None, split="test"):
    n = len(probabilities)
    y = labels if labels is not None else [i % 2 for i in range(n)]
    return pd.DataFrame({"patient_id": [f"p{i}" for i in range(n)], "method": method, "split": split,
                         "y_true": y, "y_prob": probabilities})


def reported(old: pd.DataFrame, threshold=0.5, status="ok"):
    y, p = old["y_true"].to_numpy(), old["y_prob"].to_numpy()
    m = reexec._metrics(y, p, threshold)
    return {"methods": {"m": {"status": status, "threshold": threshold,
                             "test_metrics": {"auroc": m["auroc"], "brier": m["brier"], "f1": m["f1"]}}}}


class CompareTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.old = predictions(rng.uniform(0.01, 0.99, 400))

    def one(self, new, **kw):
        return reexec.compare(reported(self.old, **kw), self.old, new)[0]

    def test_identical_rerun_is_within_tolerance(self):
        result = self.one(self.old.copy())
        self.assertTrue(result["within_tolerance"])
        self.assertEqual(result["taxonomy"], "none")
        self.assertEqual(result["patients_flag_differs"], 0)
        self.assertAlmostEqual(result["auroc_diff"], 0.0)

    def test_tiny_probability_noise_that_flips_nobody_passes(self):
        new = self.old.copy()
        new["y_prob"] = new["y_prob"] + 1e-7
        result = self.one(new, threshold=0.123456789)
        self.assertTrue(result["within_tolerance"])
        self.assertLess(result["max_abs_probability_diff"], 1e-6)

    def test_one_flipped_patient_is_allowed_two_are_not(self):
        new = self.old.copy()
        order = (new["y_prob"] - 0.5).abs().sort_values().index  # the patients nearest the threshold
        for count, limit_ok in ((1, True), (2, False)):
            changed = new.copy()
            for i in order[:count]:
                changed.loc[i, "y_prob"] = 0.0 if changed.loc[i, "y_prob"] >= 0.5 else 1.0
            result = self.one(changed)
            self.assertEqual(result["patients_flag_differs"], count)
            if not limit_ok:
                self.assertIn("patients_flag_differs", result["note"])

    def test_a_shifted_auroc_is_type_b(self):
        new = self.old.copy()
        new["y_prob"] = 1 - new["y_prob"]  # reverses the ranking
        result = self.one(new)
        self.assertFalse(result["within_tolerance"])
        self.assertEqual(result["taxonomy"], "b")
        self.assertIn("auroc_diff", result["note"])

    def test_missing_method_output_is_type_a(self):
        result = reexec.compare(reported(self.old), self.old, predictions([0.1, 0.9], method="other"))[0]
        self.assertEqual(result["taxonomy"], "a")
        self.assertFalse(result["within_tolerance"])

    def test_no_predictions_file_at_all_is_type_a(self):
        self.assertEqual(reexec.compare(reported(self.old), self.old, None)[0]["taxonomy"], "a")

    def test_a_method_reported_as_failed_with_no_reexecution_entry_is_not_judged(self):
        result = reexec.compare(reported(self.old, status="failed"), self.old, self.old)[0]
        self.assertIsNone(result["taxonomy"])
        self.assertIsNone(result["within_tolerance"])

    def test_a_method_that_failed_in_the_report_and_again_agrees(self):
        new_results = {"methods": {"m": {"status": "failed", "failure_reason": "ImportError: a"}}}
        rep = reported(self.old, status="failed")
        rep["methods"]["m"]["failure_reason"] = "ImportError: a"
        result = reexec.compare(rep, self.old, self.old, None, new_results)[0]
        self.assertTrue(result["within_tolerance"])
        self.assertEqual(result["taxonomy"], "none")
        self.assertNotIn("differs", result["note"])
        new_results["methods"]["m"]["failure_reason"] = "ValueError: something else"
        self.assertIn("differs", reexec.compare(rep, self.old, self.old, None, new_results)[0]["note"])

    def test_a_method_reported_as_failed_that_now_produces_a_result_is_a_difference(self):
        new_results = {"methods": {"m": {"status": "ok"}}}
        result = reexec.compare(reported(self.old, status="failed"), self.old, self.old, None, new_results)[0]
        self.assertFalse(result["within_tolerance"])
        self.assertEqual(result["taxonomy"], "b")

    def test_a_patient_missing_from_the_rerun_counts_as_a_flag_difference(self):
        new = self.old.iloc[:-2].copy()
        result = self.one(new)
        self.assertGreaterEqual(result["patients_flag_differs"], 2)
        self.assertFalse(result["within_tolerance"])

    def test_tolerances_come_from_the_config(self):
        new = self.old.copy()
        new["y_prob"] = (new["y_prob"] * 0.98).clip(0, 1)
        loose = reexec.compare(reported(self.old), self.old, new, {"auroc_abs": 1, "brier_abs": 1, "f1_patients_flag_differs": 10_000})
        self.assertTrue(loose[0]["within_tolerance"])

    def test_a_misreported_f1_is_caught_even_when_the_predictions_reproduce(self):
        rep = reported(self.old)
        rep["methods"]["m"]["test_metrics"]["f1"] -= 0.02      # the agent wrote a different F1 than its own predictions give
        result = reexec.compare(rep, self.old, self.old.copy())[0]
        self.assertEqual(result["patients_flag_differs"], 0)
        self.assertFalse(result["within_tolerance"])
        self.assertIn("reported_f1_gap_in_patients", result["note"])
        self.assertGreater(result["reported_f1_gap_in_patients"], 5)

    def test_a_reported_f1_within_one_patient_passes(self):
        rep = reported(self.old)
        effect = reexec.one_patient_f1_effect(reexec._metrics(self.old["y_true"].to_numpy(), self.old["y_prob"].to_numpy(), 0.5)["counts"])
        rep["methods"]["m"]["test_metrics"]["f1"] += effect * 0.5
        self.assertTrue(reexec.compare(rep, self.old, self.old.copy())[0]["within_tolerance"])

    def test_one_patient_f1_effect_matches_the_formulas(self):
        counts = {"tp": 380, "fp": 470, "fn": 270, "tn": 1880}
        d = 2 * 380 + 470 + 270
        self.assertAlmostEqual(reexec.one_patient_f1_effect(counts), max(2 * (d - 380) / (d * (d + 1)), 2 * 380 / (d * (d - 1))))

    def test_f1_counts_and_gap(self):
        result = self.one(self.old.copy())
        c = result["counts"]
        self.assertEqual(c["tp"] + c["fp"] + c["fn"] + c["tn"], 400)
        self.assertAlmostEqual(result["f1_gap"], 0.0)


class HelperTests(unittest.TestCase):
    def test_environment_against_logic(self):
        self.assertEqual(reexec.guess_failure_cause("ModuleNotFoundError: No module named 'tabm'"), "environment")
        self.assertEqual(reexec.guess_failure_cause("KeyError: 'AGE'"), "logic")

    def test_a_refused_output_folder_is_recognised(self):
        log = "Traceback (most recent call last):\n  File \"x.py\", line 90\nValueError: Writes must remain under /work/src; got /work/out\n"
        self.assertTrue(reexec.refused_output_folder(log))
        self.assertFalse(reexec.refused_output_folder("ModuleNotFoundError: No module named 'x'\n"))
        self.assertFalse(reexec.refused_output_folder("note: wrote /work/out/results.json\nall fine\n"))

    def test_package_differences(self):
        diff = reexec.package_differences(["numpy==2.0.0", "tabm==0.1", "pip==25"], ["numpy==2.1.0", "scipy==1.0", "pip==26"])
        self.assertEqual(diff["missing_in_image"], ["tabm"])
        self.assertEqual(diff["extra_in_image"], ["scipy"])
        self.assertEqual(diff["version_differs"], ["numpy: run 2.0.0, image 2.1.0"])

    def test_unreadable_predictions_are_none(self):
        self.assertIsNone(reexec.read_predictions("/no/such/file.csv"))


if __name__ == "__main__":
    unittest.main()
