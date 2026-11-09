import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.split_usage import UnknownPatientIdsError, compare_usage, record_split_usage

SPLIT = {"a1": "train", "a2": "train", "b1": "validation", "c1": "test", "c2": "test"}


def write_split(data_dir: Path) -> None:
    pd.DataFrame({"patient_id": list(SPLIT), "split": list(SPLIT.values())}).to_csv(
        data_dir / "patient_split.csv", index=False
    )


class SplitUsageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.data, self.out, self.records = root / "data", root / "out", root / "records"
        for folder in (self.data, self.out):
            folder.mkdir()
        write_split(self.data)

    def usage(self, value):
        (self.out / "split_usage.json").write_text(json.dumps(value))

    def test_clean_usage_has_no_outside_ids(self):
        result = compare_usage({"m": {"fit": ["a1", "a2"], "choices": ["a1", "b1"], "test": ["c1", "c2"]}}, SPLIT)["m"]
        for part in ("fit", "choices", "test"):
            self.assertEqual(result[part]["outside_allowed_groups"], 0)
        self.assertEqual(result["test"]["test_patients_not_scored"], 0)
        self.assertEqual(result["unknown_total"], 0)

    def test_misuse_is_recorded_and_does_not_raise(self):
        # a test patient used to fit, and a test patient missing from the scored set
        self.usage({"m": {"fit": ["a1", "c1"], "choices": ["b1"], "test": ["c2"]}})
        check = record_split_usage(self.out, self.data, self.records)
        method = check["methods"]["m"]
        self.assertEqual(method["fit"]["outside_allowed_groups"], 1)
        self.assertEqual(method["test"]["test_patients_not_scored"], 1)
        self.assertTrue((self.records / "split_check.json").exists())

    def test_ids_not_in_the_split_file_raise_after_writing_the_record(self):
        self.usage({"m": {"fit": ["a1", "zzz"], "choices": [], "test": []}})
        with self.assertRaises(UnknownPatientIdsError):
            record_split_usage(self.out, self.data, self.records)
        written = json.loads((self.records / "split_check.json").read_text())
        self.assertEqual(written["methods"]["m"]["fit"]["not_in_split_file"], 1)

    def test_numeric_looking_ids_are_compared_as_text(self):
        write_split_with = pd.DataFrame({"patient_id": ["007", "08"], "split": ["train", "test"]})
        write_split_with.to_csv(self.data / "patient_split.csv", index=False)
        self.usage({"m": {"fit": ["007"], "choices": ["007"], "test": ["08"]}})
        check = record_split_usage(self.out, self.data, self.records)
        self.assertEqual(check["methods"]["m"]["unknown_total"], 0)

    def test_missing_or_broken_file_is_recorded_not_raised(self):
        check = record_split_usage(self.out, self.data, self.records)
        self.assertEqual(check["split_usage_json"], "missing")
        (self.out / "split_usage.json").write_text("{not json")
        check = record_split_usage(self.out, self.data, self.records)
        self.assertEqual(check["split_usage_json"], "not a valid JSON object")


if __name__ == "__main__":
    unittest.main()
