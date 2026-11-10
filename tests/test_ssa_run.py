"""Host-side run script: checksums, failure classification, the docker command line, the queue."""

import hashlib
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts import ssa_run


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ChecksumTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        (self.folder / "a.csv").write_bytes(b"one")
        (self.folder / "b.csv").write_bytes(b"two")
        self.sums = ssa_run.parse_sums([f"{sha(b'one')}  a.csv", f"{sha(b'two')}  b.csv"])

    def test_parse(self):
        self.assertEqual(set(self.sums), {"a.csv", "b.csv"})
        with self.assertRaises(ssa_run.SetupError):
            ssa_run.parse_sums(["not a checksum line"])

    def test_matching_folder_is_ok(self):
        self.assertTrue(ssa_run.verify_data(self.folder, self.sums)["ok"])

    def test_changed_missing_and_extra_files_are_all_caught(self):
        (self.folder / "a.csv").write_bytes(b"changed")
        (self.folder / "c.csv").write_bytes(b"new")
        (self.folder / "b.csv").unlink()
        result = ssa_run.verify_data(self.folder, self.sums)
        self.assertFalse(result["ok"])
        self.assertEqual(result["mismatched"], ["a.csv"])
        self.assertEqual(result["missing"], ["b.csv"])
        self.assertEqual(result["extra"], ["c.csv"])


class RecordFormatTests(unittest.TestCase):
    OK = {"files_expected": 16, "missing": [], "extra": [], "mismatched": [], "ok": True}

    def test_data_check_text_matches_the_aiss_records(self):
        self.assertEqual(ssa_run.verified_text(self.OK, True),
                         "16/16 checksums OK, 16 files in folder (expected 16); mounted read-only")

    def test_data_check_text_reports_problems(self):
        bad = {"files_expected": 16, "missing": ["a.csv"], "extra": ["z.csv", "y.csv"], "mismatched": ["b.csv"], "ok": False}
        self.assertEqual(ssa_run.verified_text(bad, False),
                         "14/16 checksums OK, 17 files in folder (expected 16); NOT mounted read-only")

    def test_prompt_version_comes_from_the_file_name(self):
        self.assertEqual(ssa_run.version_of(Path("pilot-01__v05.claude-science.md")), "v05")
        self.assertIsNone(ssa_run.version_of(Path("prompt.md")))

    def test_local_stamp_has_no_zone_and_whole_seconds(self):
        stamp = ssa_run.local_stamp()
        self.assertRegex(stamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


class ClassifyFailureTests(unittest.TestCase):
    def test_infrastructure_errors_are_recognised(self):
        error = {"error_type": "APIConnectionError", "error": "Connection error."}
        self.assertEqual(ssa_run.classify_failure({}, error), "infrastructure")
        manifest = {"stage_failures": {"programming": "ClientAuthenticationError: DefaultAzureCredential failed"}}
        self.assertEqual(ssa_run.classify_failure(manifest, None), "infrastructure")

    def test_rate_limits_are_separate(self):
        error = {"error_type": "RateLimitError", "error": "Error code: 429 quota exceeded"}
        self.assertEqual(ssa_run.classify_failure({}, error), "rate_limit")

    def test_a_pipeline_failure_the_agents_caused_counts_as_a_run(self):
        error = {"error_type": "RuntimeError", "error": "Stage benchmarking exceeded its limit; code printed 429"}
        self.assertIsNone(ssa_run.classify_failure({}, error))
        self.assertIsNone(ssa_run.classify_failure({"stage_failures": {}}, None))
        self.assertIsNone(ssa_run.classify_failure({}, None))


class ContainerArgsTests(unittest.TestCase):
    def args(self, **extra):
        base = dict(name="n", image="img", run_dir=Path("/runs/r"), data_mount=Path("/mnt/wsl/csdata_x"),
                    prompt=Path("/p/prompt.md"), prompt_sha256="abc", run_id="r", max_seconds=14400, aoai_ip="10.0.0.1")
        return ssa_run.container_args(**{**base, **extra})

    def test_limits_and_no_gpu(self):
        args = self.args()
        self.assertEqual(args[args.index("--cpuset-cpus") + 1], "10-18")
        self.assertEqual(args[args.index("--memory") + 1], "11g")
        self.assertEqual(args[args.index("--memory-swap") + 1], "11g")
        self.assertEqual(args[args.index("--runtime") + 1], "runc")
        self.assertNotIn("--gpus", args)
        self.assertFalse(any("nvidia" in a for a in args))

    def test_data_is_read_only_and_output_is_not(self):
        mounts = [a for a in self.args() if a.startswith("type=bind")]
        data = next(m for m in mounts if "target=/home/llmrel/task/data" in m)
        out = next(m for m in mounts if "target=/home/llmrel/task/out" in m)
        self.assertTrue(data.endswith(",readonly"))
        self.assertFalse(out.endswith(",readonly"))

    def test_environment_and_endpoint(self):
        args = self.args()
        self.assertIn("bpsmar-ai-openai-1.openai.azure.com:10.0.0.1", args)
        self.assertIn("STUDY_DEADLINE_SECONDS=14400", args)
        self.assertIn("LOKY_MAX_CPU_COUNT=9", args)
        self.assertIn("USE_LITERATURE_FIXTURE=1", self.args(fixture_literature=True))
        self.assertNotIn("USE_LITERATURE_FIXTURE=1", args)


class DecideFailureTests(unittest.TestCase):
    connection = {"error_type": "APIConnectionError", "error": "Connection error."}

    def test_a_successful_run_is_never_an_infrastructure_failure(self):
        manifest = {"run_status": "success", "stage_failures": {"programming": "APIConnectionError: x"}}
        self.assertEqual(ssa_run.decide_failure(manifest, None, 0, False), (None, "infrastructure"))

    def test_a_failed_run_with_an_infrastructure_error_is_one(self):
        manifest = {"run_status": "failed", "stage_failures": {"benchmarking": "APIConnectionError: x"}}
        self.assertEqual(ssa_run.decide_failure(manifest, self.connection, 1, False)[0], "infrastructure")

    def test_a_failed_run_for_other_reasons_counts(self):
        manifest = {"run_status": "failed", "stage_failures": {"benchmarking": "RuntimeError: gave up"}}
        self.assertEqual(ssa_run.decide_failure(manifest, {"error_type": "RuntimeError", "error": "gave up"}, 1, False), (None, None))

    def test_no_records_at_all(self):
        self.assertEqual(ssa_run.decide_failure(None, None, 1, False)[0], "infrastructure")
        self.assertIsNone(ssa_run.decide_failure(None, None, 137, True)[0])  # killed at the time limit: a real run


class QueueTests(unittest.TestCase):
    def queue(self, outcomes, n=2, max_retries=1):
        calls, waits = [], []
        it = iter(outcomes)

        def fake_run(args, meta, attempt=1):
            calls.append(attempt)
            outcome = next(it)
            if isinstance(outcome, Exception):
                raise outcome
            return {"counts_as_study_run": outcome is None, "infrastructure_failure": outcome}

        records = ssa_run.run_queue(Namespace(n=n, max_retries=max_retries), {}, run=fake_run, sleep=waits.append)
        return records, calls, waits

    def test_runs_until_n_study_runs(self):
        records, _, waits = self.queue([None, None])
        self.assertEqual(len(records), 2)
        self.assertEqual(waits, [])

    def test_one_repeat_after_an_infrastructure_failure(self):
        records, calls, waits = self.queue(["infrastructure", None, None])
        self.assertEqual(sum(r["counts_as_study_run"] for r in records), 2)
        self.assertEqual(waits, [300])
        self.assertEqual(calls[:2], [1, 2])

    def test_rate_limit_waits_longer(self):
        _, _, waits = self.queue(["rate_limit", None, None])
        self.assertEqual(waits, [900])

    def test_gives_up_when_the_repeat_fails_too(self):
        records, _, waits = self.queue(["infrastructure", "infrastructure"], n=1)
        self.assertEqual(sum(r["counts_as_study_run"] for r in records), 0)
        self.assertEqual(waits, [300])

    def test_no_repeats_when_max_retries_is_zero(self):
        records, _, waits = self.queue(["infrastructure"], n=1, max_retries=0)
        self.assertEqual(len(records), 1)
        self.assertEqual(waits, [])

    def test_setup_problems_wait_without_using_up_the_repeat(self):
        records, _, waits = self.queue([ssa_run.SetupError("VPN"), ssa_run.SetupError("VPN"), None, None])
        self.assertEqual(sum(r["counts_as_study_run"] for r in records), 2)
        self.assertEqual(waits, [300, 900])

    def test_gives_up_when_the_setup_never_comes_back(self):
        records, _, waits = self.queue([ssa_run.SetupError("login")] * 4, n=1)
        self.assertEqual(records, [])
        self.assertEqual(waits, [300, 900, 2700])


if __name__ == "__main__":
    unittest.main()
