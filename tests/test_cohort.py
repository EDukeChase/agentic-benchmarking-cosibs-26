"""Edge-case checks for the cohort definition and the three-way split.

Run with: python -m unittest discover -s tests
"""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.cohort import build_cohort
from src.evaluation.study_split import make_three_way_split

IHD, MI = "414545008", "22298006"


def write_tables(folder: Path, patients: list[tuple], conditions: list[tuple]) -> None:
    pd.DataFrame(patients, columns=["Id", "BIRTHDATE", "DEATHDATE"]).to_csv(folder / "patients.csv", index=False)
    pd.DataFrame(conditions, columns=["START", "PATIENT", "CODE"]).to_csv(folder / "conditions.csv", index=False)


class CohortTests(unittest.TestCase):
    def build(self, patients, conditions):
        with tempfile.TemporaryDirectory() as tmp:
            write_tables(Path(tmp), patients, conditions)
            return build_cohort(tmp).set_index("patient_id")["y"].to_dict()

    def test_age_boundaries_on_index_date(self):
        # index date 2015-12-31: born 1975-12-31 is exactly 40, 1975-01-01 is 40, 1976-01-01 is 39,
        # born 1936-01-01 is 79, born 1935-12-31 is 80
        patients = [
            ("exactly40", "1975-12-31", None),
            ("turned40", "1975-01-01", None),
            ("age39", "1976-01-01", None),
            ("age79", "1936-01-01", None),
            ("age80", "1935-12-31", None),
        ]
        self.assertEqual(set(self.build(patients, [])), {"exactly40", "turned40", "age79"})

    def test_death_before_or_on_index_date_excluded(self):
        patients = [
            ("alive", "1960-01-01", None),
            ("died_after", "1960-01-01", "2016-01-01"),
            ("died_on_index", "1960-01-01", "2015-12-31"),
            ("died_before", "1960-01-01", "2015-06-01"),
        ]
        self.assertEqual(set(self.build(patients, [])), {"alive", "died_after"})

    def test_prior_ihd_or_mi_excluded_but_later_ones_are_outcomes(self):
        patients = [(p, "1960-01-01", None) for p in ("prior_ihd", "prior_mi", "ihd_on_index", "later_ihd", "later_mi_only", "none")]
        conditions = [
            ("2010-05-05", "prior_ihd", IHD),
            ("2012-05-05", "prior_mi", MI),
            ("2015-12-31", "ihd_on_index", IHD),
            ("2018-03-03", "later_ihd", IHD),
            ("2018-03-03", "later_mi_only", MI),
        ]
        result = self.build(patients, conditions)
        self.assertEqual(set(result), {"later_ihd", "later_mi_only", "none"})
        self.assertEqual(result, {"later_ihd": 1, "later_mi_only": 0, "none": 0})

    def test_outcome_window_and_first_event(self):
        patients = [(p, "1960-01-01", None) for p in ("in_window", "on_end", "after_end", "two_events")]
        conditions = [
            ("2020-01-01", "in_window", IHD),
            ("2025-12-31", "on_end", IHD),
            ("2026-01-01", "after_end", IHD),
            ("2030-01-01", "two_events", IHD),
            ("2018-01-01", "two_events", IHD),
        ]
        result = self.build(patients, conditions)
        self.assertEqual(result, {"in_window": 1, "on_end": 1, "after_end": 0, "two_events": 1})

    def test_death_in_window_without_condition_is_zero(self):
        patients = [("died_in_window", "1960-01-01", "2020-01-01")]
        self.assertEqual(self.build(patients, []), {"died_in_window": 0})


class SplitTests(unittest.TestCase):
    def setUp(self):
        ids = [f"p{i:03d}" for i in range(200)]
        self.cohort = pd.DataFrame({"patient_id": ids, "y": [int(i % 5 == 0) for i in range(200)]})
        labels = ["train"] * 160 + ["validation"] * 40
        self.provided = pd.DataFrame({"patient_id": ids + ["outside"], "split": labels + ["train"]})

    def test_groups_and_sizes(self):
        result = make_three_way_split(self.provided, self.cohort)
        counts = result[result["patient_id"] != "outside"]["split"].value_counts().to_dict()
        self.assertEqual(counts, {"train": 120, "validation": 40, "test": 40})

    def test_provided_validation_becomes_test_and_outsiders_stay_train(self):
        result = make_three_way_split(self.provided, self.cohort).set_index("patient_id")["split"]
        self.assertTrue((result.loc[[f"p{i:03d}" for i in range(160, 200)]] == "test").all())
        self.assertEqual(result.loc["outside"], "train")

    def test_stratified_on_outcome(self):
        result = make_three_way_split(self.provided, self.cohort).merge(self.cohort, on="patient_id")
        rates = result.groupby("split")["y"].mean().round(2).to_dict()
        self.assertEqual(rates, {"train": 0.2, "validation": 0.2, "test": 0.2})

    def test_deterministic_for_a_seed_and_different_across_seeds(self):
        a = make_three_way_split(self.provided, self.cohort, seed=5960)
        b = make_three_way_split(self.provided, self.cohort, seed=5960)
        c = make_three_way_split(self.provided, self.cohort, seed=1)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a["split"].equals(c["split"]))

    def test_cohort_patient_without_label_raises(self):
        with self.assertRaises(ValueError):
            make_three_way_split(self.provided[self.provided["patient_id"] != "p000"], self.cohort)


if __name__ == "__main__":
    unittest.main()
