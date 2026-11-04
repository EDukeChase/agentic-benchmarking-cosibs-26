"""Checks that the split checker accepts a good split and catches specific faults.

Run with: python -m unittest discover -s tests
"""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.cohort import build_cohort
from src.evaluation.split_check import check_split
from src.evaluation.study_split import make_three_way_split

IHD = "414545008"
N = 400  # 25% events: every fourth patient


class SplitCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        ids = [f"p{i:03d}" for i in range(N)]
        pd.DataFrame({"Id": ids, "BIRTHDATE": "1960-01-01", "DEATHDATE": None}).to_csv(self.data / "patients.csv", index=False)
        events = [("2020-06-01", pid, IHD) for i, pid in enumerate(ids) if i % 4 == 0]
        pd.DataFrame(events, columns=["START", "PATIENT", "CODE"]).to_csv(self.data / "conditions.csv", index=False)
        # a provided two-group split, 80/20, interleaved so both groups carry events
        self.provided = pd.DataFrame({"patient_id": ids, "split": ["validation" if i % 5 == 4 else "train" for i in range(N)]})
        self.good = make_three_way_split(self.provided, build_cohort(self.data))

    def tearDown(self):
        self.tmp.cleanup()

    def failures(self, split, provided=None):
        return [c.name for c in check_split(split, self.data, provided).failed]

    def assertFails(self, split, fragment, provided=None):
        failures = self.failures(split, provided)
        self.assertTrue(any(fragment in name for name in failures), f"expected a failure mentioning {fragment!r}, got {failures}")

    def test_split_made_by_the_study_script_passes_including_the_design_checks(self):
        report = check_split(self.good, self.data, self.provided)
        self.assertEqual(report.failed, [], [(c.name, c.detail) for c in report.failed])

    def test_wrong_columns(self):
        self.assertFails(self.good.rename(columns={"split": "group"}), "columns are patient_id and split")

    def test_unknown_label(self):
        bad = self.good.copy()
        bad.loc[0, "split"] = "holdout"
        self.assertFails(bad, "labels are train, validation and test")

    def test_repeated_patient(self):
        self.assertFails(pd.concat([self.good, self.good.iloc[[0]]]), "each patient appears once")

    def test_unlabeled_cohort_patient(self):
        self.assertFails(self.good.iloc[1:], "every cohort patient has a split label")

    def test_group_far_from_its_target_share(self):
        bad = self.good.copy()
        bad.loc[bad["split"] == "test", "split"] = "train"
        self.assertFails(bad, "test: share of the cohort")

    def test_unbalanced_event_rates(self):
        # put all event patients in train: rates differ wildly even though sizes are fine
        cohort = build_cohort(self.data).set_index("patient_id")
        bad = self.good.copy()
        events = set(cohort.index[cohort["y"] == 1])
        positive = bad["patient_id"].isin(events)
        bad.loc[positive, "split"] = "train"
        self.assertFails(bad, "event rate is within")

    def test_group_with_one_outcome_class(self):
        cohort = build_cohort(self.data).set_index("patient_id")
        bad = self.good.copy()
        non_events = bad["patient_id"].isin(set(cohort.index[cohort["y"] == 0]))
        bad.loc[non_events & (bad["split"] == "test"), "split"] = "train"
        bad.loc[non_events & bad["patient_id"].isin(bad[bad["split"] == "validation"]["patient_id"]), "split"] = "validation"
        self.assertFails(bad, "has both outcome classes")

    def test_test_group_must_be_the_provided_validation_patients(self):
        bad = self.good.copy()
        test_ids = bad.index[bad["split"] == "test"][:2]
        train_ids = bad.index[bad["split"] == "train"][:2]
        bad.loc[test_ids, "split"] = "train"
        bad.loc[train_ids, "split"] = "test"
        self.assertFails(bad, "the test patients are exactly the provided validation patients", provided=self.provided)

    def test_provided_train_patient_in_test_is_flagged(self):
        bad = self.good.copy()
        bad.loc[bad.index[bad["split"] == "train"][0], "split"] = "test"
        self.assertFails(bad, "no provided train patient is in test", provided=self.provided)

    def test_unknown_patient_id_is_a_warning_not_a_failure(self):
        extra = pd.concat([self.good, pd.DataFrame({"patient_id": ["ghost"], "split": ["train"]})])
        report = check_split(extra, self.data)
        self.assertEqual(report.failed, [])
        self.assertTrue(any(c.status == "warn" for c in report.checks))


if __name__ == "__main__":
    unittest.main()
