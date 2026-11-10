"""Host-side run script: checksums, failure classification, the docker command line, the queue."""

import hashlib
import json
import os
import tempfile
import time
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


class StudySettingsTests(unittest.TestCase):
    GOOD = dict(dataset="study_v1", fixture_literature=False, max_hours=4.0, memory="11g", cpuset="10-18",
                prompt_sha256="abc")
    LABEL = {"git_commit": "a" * 40, "git_dirty": "false"}

    def problems(self, label=None, repo_commit="a" * 40, repo_dirty=False, **changes):
        args = Namespace(**{**self.GOOD, **changes})
        return ssa_run.study_settings_problems(args, self.LABEL if label is None else label, repo_commit, repo_dirty)

    def test_the_study_settings_are_accepted(self):
        self.assertEqual(self.problems(), [])
        self.assertEqual(self.problems(cpuset="0-8"), [])

    def test_each_departure_is_reported(self):
        self.assertEqual(len(self.problems(dataset="test_small")), 1)
        self.assertEqual(len(self.problems(fixture_literature=True)), 1)
        self.assertEqual(len(self.problems(max_hours=2.0)), 1)
        self.assertEqual(len(self.problems(memory="8g")), 1)
        self.assertEqual(len(self.problems(cpuset="1-9")), 1)
        self.assertEqual(len(self.problems(prompt_sha256=None)), 1)

    def test_the_image_must_come_from_the_committed_code(self):
        self.assertEqual(len(self.problems(label={})), 1)                                            # no label
        self.assertEqual(len(self.problems(label={"git_commit": "a" * 40, "git_dirty": "true"})), 1)  # built from a dirty tree
        self.assertEqual(len(self.problems(repo_commit="b" * 40)), 1)                                # code moved on since the build
        self.assertEqual(len(self.problems(repo_dirty=True)), 1)                                     # uncommitted changes now


class CopyResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.run_dir = root / "run"
        for sub in ("out", "records"):
            (self.run_dir / sub).mkdir(parents=True)
        (self.run_dir / "out" / "results.json").write_text("{}")
        (self.run_dir / "records" / "usage.json").write_text("{}")
        (self.run_dir / "run_record.json").write_text("{}")
        (self.run_dir / "container.log").write_text("log")
        self.out_root, self.rec_root = root / "agent-runs", root / "run-records"

    def args(self, **kw):
        base = dict(kind="study", copy=False, no_copy=False, out_copy_root=str(self.out_root), records_copy_root=str(self.rec_root))
        return Namespace(**{**base, **kw})

    def test_a_study_run_is_copied_to_both_places(self):
        ssa_run.copy_results(self.args(), "ssa-x", self.run_dir)
        self.assertTrue((self.out_root / "ssa-x" / "out" / "results.json").exists())
        for name in ("records/usage.json", "run_record.json", "container.log"):
            self.assertTrue((self.rec_root / "ssa-x" / name).exists(), name)

    def test_a_test_run_is_copied_only_when_asked(self):
        ssa_run.copy_results(self.args(kind="test"), "ssa-t", self.run_dir)
        self.assertFalse(self.out_root.exists())
        ssa_run.copy_results(self.args(kind="test", copy=True), "ssa-t", self.run_dir)
        self.assertTrue((self.out_root / "ssa-t" / "out").exists())

    def test_no_copy_wins(self):
        ssa_run.copy_results(self.args(no_copy=True), "ssa-n", self.run_dir)
        self.assertFalse(self.out_root.exists() or self.rec_root.exists())


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

    def test_a_clean_success_counts(self):
        self.assertEqual(ssa_run.decide_failure({"run_status": "success", "stage_failures": {}}, None, 0, False), (None, None))

    def test_an_infrastructure_error_in_any_stage_means_the_run_does_not_count(self):
        # the pipeline "succeeded" without the programming stage's work: not the agents' doing
        manifest = {"run_status": "success", "stage_failures": {"programming": "APIConnectionError: x"}}
        self.assertEqual(ssa_run.decide_failure(manifest, None, 0, False), ("infrastructure", "infrastructure"))

    def test_a_failed_run_with_an_infrastructure_error_is_one(self):
        manifest = {"run_status": "failed", "stage_failures": {"benchmarking": "APIConnectionError: x"}}
        self.assertEqual(ssa_run.decide_failure(manifest, self.connection, 1, False)[0], "infrastructure")

    def test_agent_outcomes_count(self):
        manifest = {"run_status": "failed", "stage_failures": {"benchmarking": "StageTimeoutError: over its limit"}}
        err = {"error_type": "StageTimeoutError", "error": "over its limit"}
        self.assertEqual(ssa_run.decide_failure(manifest, err, 1, False), (None, None))
        missing = {"run_status": "success", "stage_failures": {"programming": "no model.py for: ['tabm']"}}
        self.assertEqual(ssa_run.decide_failure(missing, None, 0, False), (None, None))
        no_method_worked = {"run_status": "failed", "stage_failures": {}}
        self.assertEqual(ssa_run.decide_failure(no_method_worked, None, 1, False), (None, None))

    def test_an_unrecognised_error_is_held_not_counted(self):
        manifest = {"run_status": "failed", "stage_failures": {"benchmarking": "KeyError: 'messages'"}}
        err = {"error_type": "KeyError", "error": "'messages'"}
        self.assertEqual(ssa_run.decide_failure(manifest, err, 1, False), ("harness_suspect", None))
        bad_request = {"run_status": "success", "stage_failures": {"reporting": "BadRequestError: 400"}}
        self.assertEqual(ssa_run.decide_failure(bad_request, None, 0, False)[0], "harness_suspect")

    def test_time_limit_and_out_of_memory_are_real_runs(self):
        self.assertIsNone(ssa_run.decide_failure(None, None, 137, True)[0])
        self.assertIsNone(ssa_run.decide_failure(None, None, 137, False, oom=True)[0])

    def test_no_records_at_all(self):
        self.assertEqual(ssa_run.decide_failure(None, None, 137, False)[0], "infrastructure")  # killed from outside
        self.assertEqual(ssa_run.decide_failure(None, None, 1, False)[0], "harness_suspect")
        self.assertEqual(ssa_run.decide_failure(None, None, 0, False)[0], "harness_suspect")


class FakeHost:
    def __init__(self, active=(), memory=20.0, disk=500.0, stale=()):
        self.active, self.memory, self.disk, self.stale = list(active), memory, disk, list(stale)

    def active_run_names(self):
        return list(self.active)

    def stale_container_names(self):
        return list(self.stale)

    def available_memory_gb(self):
        return self.memory

    def free_disk_gb(self, path):
        return self.disk


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def queue(self, outcomes, host=None, stop_after=None, **extra):
        calls, waits = [], []
        it = iter(outcomes)

        def fake_run(args, meta, attempt=1):
            calls.append(attempt)
            outcome = next(it)
            if isinstance(outcome, Exception):
                raise outcome
            tokens = 1000
            record = {"run_id": f"ssa-{len(calls)}", "counts_as_study_run": outcome is None, "failure_class": outcome,
                      "usage": {"tokens_total": tokens}}
            if stop_after and len(calls) >= stop_after:
                (self.root / "STOP").write_text("")
            return record

        values = dict(n=2, runs_dir=str(self.root), out_copy_root=str(self.root), records_copy_root=str(self.root),
                      stop_file=None, max_tokens=0, max_consecutive_failures=3, setup_patience_hours=8.0,
                      min_free_gb=8.0, min_free_disk_gb=10.0)
        values.update(extra)
        records = ssa_run.run_queue(Namespace(**values), {}, run=fake_run, sleep=waits.append, host=host or FakeHost())
        return records, calls, waits

    def status(self):
        return json.loads((self.root / "queue_status.json").read_text())

    def test_runs_until_n_study_runs(self):
        records, _, waits = self.queue([None, None])
        self.assertEqual(len(records), 2)
        self.assertEqual(waits, [])
        self.assertTrue(self.status()["state"].startswith("Done"))

    def test_a_run_that_does_not_count_is_replaced_after_a_wait(self):
        records, _, waits = self.queue(["infrastructure", None, None])
        self.assertEqual(sum(r["counts_as_study_run"] for r in records), 2)
        self.assertEqual(sum(waits), 300)

    def test_rate_limit_waits_at_least_15_minutes(self):
        _, _, waits = self.queue(["rate_limit", None, None])
        self.assertEqual(sum(waits), 900)

    def test_waits_grow_and_the_queue_stops_after_three_in_a_row(self):
        records, _, waits = self.queue(["infrastructure"] * 5, n=1)
        self.assertEqual(len(records), 3)
        self.assertEqual(sum(waits), 300 + 900)  # no wait after the run that stops the queue
        self.assertIn("did not count", self.status()["state"])
        self.assertTrue(self.status()["needs_attention"])

    def test_two_held_runs_in_a_row_stop_the_queue(self):
        records, _, _ = self.queue(["harness_suspect", "harness_suspect", None], n=1)
        self.assertEqual(len(records), 2)
        self.assertIn("held for review", self.status()["state"])

    def test_a_held_run_followed_by_a_good_one_carries_on(self):
        records, _, _ = self.queue(["harness_suspect", None], n=1)
        self.assertEqual(sum(r["counts_as_study_run"] for r in records), 1)

    def test_setup_problems_wait_and_do_not_use_up_a_failure(self):
        records, calls, waits = self.queue([ssa_run.SetupError("VPN"), ssa_run.SetupError("VPN"), None], n=1)
        self.assertEqual(sum(r["counts_as_study_run"] for r in records), 1)
        self.assertEqual(sum(waits), 300 + 900)
        self.assertEqual(calls, [1, 1, 1])

    def test_gives_up_when_the_setup_never_comes_back(self):
        records, _, _ = self.queue([ssa_run.SetupError("login")] * 20, n=1, setup_patience_hours=1)
        self.assertEqual(records, [])
        self.assertTrue(self.status()["needs_attention"])

    def test_stop_file_ends_the_queue_after_the_current_run(self):
        records, _, _ = self.queue([None, None, None], n=0, stop_after=1)
        self.assertEqual(len(records), 1)
        self.assertIn("exists", self.status()["state"])

    def test_token_budget(self):
        records, _, _ = self.queue([None] * 5, n=0, max_tokens=2000)
        self.assertEqual(len(records), 2)
        self.assertIn("token budget", self.status()["state"])

    def test_waits_while_another_ssa_run_is_active(self):
        class Busy(FakeHost):
            polls = 0

            def active_run_names(self):
                Busy.polls += 1
                return ["ssa-ssa-x"] if Busy.polls <= 3 else []

        records, _, waits = self.queue([None], n=1, host=Busy())
        self.assertEqual(len(records), 1)
        self.assertGreaterEqual(sum(waits), 600)

    def test_low_memory_and_low_disk_wait_instead_of_starting(self):
        self.assertIn("memory", ssa_run.not_ready_reason(Namespace(runs_dir="x", out_copy_root="x", records_copy_root="x", min_free_gb=8, min_free_disk_gb=10), FakeHost(memory=3)))
        self.assertIn("free on the disk", ssa_run.not_ready_reason(Namespace(runs_dir="x", out_copy_root="x", records_copy_root="x", min_free_gb=8, min_free_disk_gb=10), FakeHost(disk=2)))

    def test_orphans_get_an_aborted_record(self):
        old = self.root / "ssa-20260101-000000-aaaa"
        (old / "out").mkdir(parents=True)
        running = self.root / "ssa-20260101-000001-bbbb"
        running.mkdir()
        done = self.root / "ssa-20260101-000002-cccc"
        done.mkdir()
        (done / "run_record.json").write_text("{}")
        past = time.time() - 3600
        for folder in (old, running, done):
            os.utime(folder, (past, past))
        marked = ssa_run.mark_orphans(self.root, FakeHost(active=["ssa-ssa-20260101-000001-bbbb"]))
        self.assertEqual(marked, ["ssa-20260101-000000-aaaa"])
        self.assertEqual(json.loads((old / "run_record.json").read_text())["status"], "aborted")
        self.assertFalse((running / "run_record.json").exists())


if __name__ == "__main__":
    unittest.main()
