import json
import tempfile
import unittest
from pathlib import Path

from scripts import reexecute_recompare as rr


class RecomputeTests(unittest.TestCase):
    def test_methods_judged_in_both_ways_make_the_run_agree(self):
        result = {"first_attempt": "ran", "all_within_tolerance": False, "taxonomy_run_level": "none"}
        methods = [{"within_tolerance": True}, {"within_tolerance": True}]
        self.assertTrue(rr.recompute_fields(result, methods)["all_within_tolerance"])

    def test_a_method_not_judged_blocks_agreement(self):
        out = rr.recompute_fields({"first_attempt": "ran"}, [{"within_tolerance": True}, {"within_tolerance": None}])
        self.assertFalse(out["all_within_tolerance"])
        self.assertEqual(out["taxonomy_run_level"], "b")

    def test_a_run_that_did_not_run_stays_type_a(self):
        self.assertEqual(rr.recompute_fields({"first_attempt": "failed"}, [])["taxonomy_run_level"], "a")


class MainTests(unittest.TestCase):
    def test_a_changed_comparison_is_written_and_the_old_one_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, runs = Path(tmp) / "res", Path(tmp) / "runs"
            folder = root / "ssa-x-abcd" / "exec1"
            folder.mkdir(parents=True)
            (runs / "ssa-x-abcd" / "out").mkdir(parents=True)
            (folder / "reexecuted_results.json").write_text("{}")
            (folder / "compare.json").write_text("{}")
            (folder / "result.json").write_text(json.dumps({"first_attempt": "ran", "all_within_tolerance": False, "methods": [{"within_tolerance": None}]}))
            code = rr.main(["--result-root", str(root), "--runs-dir", str(runs), "--tool-image", "t"],
                           compare=lambda *a: {"methods": [{"within_tolerance": True}]})
            new = json.loads((folder / "result.json").read_text())
        self.assertEqual(code, 0)
        self.assertTrue(new["all_within_tolerance"])
        self.assertIn("recompared_at", new)


if __name__ == "__main__":
    unittest.main()
