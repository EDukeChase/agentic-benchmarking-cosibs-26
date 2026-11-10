"""The pipeline's control flow, with fake agents. No model, network or real data is used."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

import main as pipeline

PROMPT = "Study prompt text for /home/llmrel/task/data.\n"
RESULTS = {
    "seed": 5960,
    "cohort": {},
    "methods": {
        "logistic_regression": {"status": "ok", "attempts": 1, "source": None},
        "logistic_regression_specified": {"status": "ok", "attempts": 1, "source": None},
        "gradient_boosting": {"status": "failed", "attempts": 2, "source": "https://example.org"},
    },
}


class Fake:
    """Stand-ins for the four stages. Each one can be told to raise."""

    def __init__(self, task_dirs, programming_error=None, benchmarking_error=None, usage=None):
        self.data, self.out = task_dirs
        self.programming_error, self.benchmarking_error = programming_error, benchmarking_error
        self.usage = usage or {"m": {"fit": ["p1"], "choices": ["p1", "p2"], "test": ["p3"]}}

    def run_programming(self, agent, literature, study_prompt):
        if self.programming_error:
            raise self.programming_error
        for candidate in literature.candidates[:2]:  # the third method is never implemented
            folder = Path(self.out) / "code" / "methods" / pipeline.slugify(candidate.model_name)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "model.py").write_text("Model = object\n")
        return {"messages": [SimpleNamespace(content="done", usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})]}

    def run_benchmarking(self, agent, task, study_prompt, literature, missing):
        self.missing_seen = missing
        if self.benchmarking_error:
            raise self.benchmarking_error
        (Path(self.out) / "results.json").write_text(json.dumps(RESULTS))
        (Path(self.out) / "split_usage.json").write_text(json.dumps(self.usage))
        return {"messages": [SimpleNamespace(content="all five methods attempted")]}

    def write_report(self, llm, study_prompt, out_dir, literature, message):
        return SimpleNamespace(content="# Report\n")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.data, self.out, self.records = root / "task" / "data", root / "task" / "out", root / "records"
        self.data.mkdir(parents=True)
        pd.DataFrame({"patient_id": ["p1", "p2", "p3"], "split": ["train", "validation", "test"]}).to_csv(
            self.data / "patient_split.csv", index=False
        )
        prompt = root / "prompt.md"
        prompt.write_text(PROMPT)
        env = {
            "STUDY_DATA_DIR": str(self.data), "STUDY_OUT_DIR": str(self.out), "STUDY_RECORDS_DIR": str(self.records),
            "STUDY_PROMPT_FILE": str(prompt), "USE_LITERATURE_FIXTURE": "1", "BENCHMARK_RUN_ID": "testrun",
        }
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_pipeline(self, **fake_kwargs):
        fake = Fake((self.data, self.out), **fake_kwargs)
        patches = [
            mock.patch.object(pipeline, "build_programming_agent", lambda *a, **k: None),
            mock.patch.object(pipeline, "run_programming_agent", fake.run_programming),
            mock.patch.object(pipeline, "build_benchmarking_agent", lambda *a, **k: None),
            mock.patch.object(pipeline, "run_benchmarking_agent", fake.run_benchmarking),
            mock.patch.object(pipeline, "build_reporting_agent", lambda *a, **k: None),
            mock.patch.object(pipeline, "write_report", fake.write_report),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        code = pipeline.main()
        manifest = json.loads((self.records / "run_manifest.json").read_text())
        return code, manifest, fake

    def test_a_normal_run_succeeds_and_records_everything(self):
        code, manifest, fake = self.run_pipeline()
        self.assertEqual(code, 0)
        self.assertEqual(manifest["run_status"], "success")
        self.assertEqual(manifest["system"], "SSA")
        self.assertEqual(len(fake.missing_seen), 1)  # the third method has no model.py
        self.assertIn("programming", manifest["stage_failures"])  # recorded, but not fatal
        self.assertEqual(manifest["results"]["methods"]["gradient_boosting"]["status"], "failed")
        self.assertEqual((self.out / "report.md").read_text(), "# Report\n")
        for name in ("literature.json", "split_check.json", "run_manifest.json"):
            self.assertTrue((self.records / name).exists(), name)
        self.assertEqual(manifest["token_usage"]["total_tokens"], 5)
        usage = json.loads((self.records / "usage.json").read_text())
        self.assertEqual(usage["tok_main_input"], 3)
        self.assertEqual(usage["approvals_by_user"], 0)
        self.assertEqual(usage["code_runs"], 0)
        for key in ("peak_memory_gb", "oom_kills", "anon_at_finish_gb", "file_cache_at_finish_gb", "peak_includes_file_cache"):
            self.assertIn(key, usage["resource_use"])
        self.assertEqual(len(usage["start"]), 16)  # "YYYY-MM-DD HH:MM", like the AISS usage.json
        self.assertEqual(len(manifest["study_prompt_sha256"]), 64)

    def test_a_programming_failure_does_not_stop_the_run(self):
        code, manifest, _ = self.run_pipeline(programming_error=RuntimeError("model call failed"))
        self.assertEqual(code, 0)
        self.assertEqual(manifest["run_status"], "success")
        self.assertIn("model call failed", manifest["stage_failures"]["programming"])

    def test_a_benchmarking_failure_fails_the_run_and_keeps_the_record(self):
        code, manifest, _ = self.run_pipeline(benchmarking_error=RuntimeError("agent gave up"))
        self.assertEqual(code, 1)
        self.assertEqual(manifest["run_status"], "failed")
        self.assertEqual(manifest["failed_stage"], "benchmarking")
        self.assertTrue((self.records / "error.json").exists())

    def test_patient_ids_outside_the_split_fail_the_run(self):
        code, manifest, _ = self.run_pipeline(usage={"m": {"fit": ["nobody"], "choices": [], "test": []}})
        self.assertEqual(code, 1)
        self.assertEqual(manifest["failed_stage"], "split check")
        self.assertEqual(manifest["split_check_file"], "split_check.json")

    def test_misusing_patients_inside_the_split_does_not_fail_the_run(self):
        code, manifest, _ = self.run_pipeline(usage={"m": {"fit": ["p3"], "choices": ["p3"], "test": ["p1"]}})
        self.assertEqual(code, 0)
        check = json.loads((self.records / "split_check.json").read_text())
        self.assertEqual(check["methods"]["m"]["fit"]["outside_allowed_groups"], 1)

    def test_a_dirty_output_folder_is_refused(self):
        self.out.mkdir(parents=True)
        (self.out / "left_over.txt").write_text("x")
        with self.assertRaises(RuntimeError):
            pipeline.main()


if __name__ == "__main__":
    unittest.main()
