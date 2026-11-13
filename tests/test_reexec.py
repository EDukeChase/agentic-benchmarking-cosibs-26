import unittest

import numpy as np
import pandas as pd

from src.evaluation import reexec


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

    def test_a_method_reported_as_failed_is_not_compared(self):
        result = reexec.compare(reported(self.old, status="failed"), self.old, self.old)[0]
        self.assertIsNone(result["taxonomy"])
        self.assertIsNone(result["within_tolerance"])

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

    def test_f1_counts_and_gap(self):
        result = self.one(self.old.copy())
        c = result["counts"]
        self.assertEqual(c["tp"] + c["fp"] + c["fn"] + c["tn"], 400)
        self.assertAlmostEqual(result["f1_gap"], 0.0)


class HelperTests(unittest.TestCase):
    def test_environment_against_logic(self):
        self.assertEqual(reexec.guess_failure_cause("ModuleNotFoundError: No module named 'tabm'"), "environment")
        self.assertEqual(reexec.guess_failure_cause("KeyError: 'AGE'"), "logic")

    def test_package_differences(self):
        diff = reexec.package_differences(["numpy==2.0.0", "tabm==0.1", "pip==25"], ["numpy==2.1.0", "scipy==1.0", "pip==26"])
        self.assertEqual(diff["missing_in_image"], ["tabm"])
        self.assertEqual(diff["extra_in_image"], ["scipy"])
        self.assertEqual(diff["version_differs"], ["numpy: run 2.0.0, image 2.1.0"])

    def test_unreadable_predictions_are_none(self):
        self.assertIsNone(reexec.read_predictions("/no/such/file.csv"))


if __name__ == "__main__":
    unittest.main()
