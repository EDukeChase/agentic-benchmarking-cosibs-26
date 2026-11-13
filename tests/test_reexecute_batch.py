import json
import tempfile
import unittest
from pathlib import Path

from scripts import reexecute_batch

IDS = [f"ssa-test-{i:02d}" for i in range(13)]


def expected_draw(ids, k):
    """The rule written out independently: lowest SHA-256 score of "<seed>|SSA|<id>" first."""
    import hashlib
    return sorted(sorted(ids), key=lambda r: int(hashlib.sha256(f"5960|SSA|{r}".encode()).hexdigest(), 16))[:k]


class SubsampleTests(unittest.TestCase):
    def test_the_draw_matches_the_rule_written_out_independently(self):
        self.assertEqual(reexecute_batch.second_execution_set(sorted(IDS)), expected_draw(IDS, 3))

    def test_size_follows_the_rule(self):
        ids = [f"ssa-{i}" for i in range(50)]
        self.assertEqual(len(reexecute_batch.second_execution_set(ids)), 5)
        self.assertEqual(len(reexecute_batch.second_execution_set(ids[:20])), 3)
        self.assertEqual(len(reexecute_batch.second_execution_set(ids[:2])), 2)
        self.assertEqual(len(reexecute_batch.second_execution_set(ids[:11])), 3 + 0)  # ceil(1.1) = 2, floor of 3 wins
        self.assertEqual(len(reexecute_batch.second_execution_set([f"ssa-{i}" for i in range(100)])), 10)

    def test_the_draw_uses_only_the_run_ids(self):
        self.assertEqual(reexecute_batch.second_execution_set(IDS), reexecute_batch.second_execution_set(list(reversed(IDS))))


def make_run(runs: Path, run_id: str, finished: str, status="success", kind="study", results=True):
    folder = runs / run_id
    (folder / "out").mkdir(parents=True)
    if results:
        (folder / "out" / "results.json").write_text("{}")
    (folder / "run_record.json").write_text(json.dumps({"run_kind": kind, "pipeline_status": status, "finished": finished}))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runs, self.results = Path(self.tmp.name) / "runs", Path(self.tmp.name) / "results"
        self.runs.mkdir()

    def test_only_counted_runs_in_finish_order(self):
        make_run(self.runs, "ssa-b", "2026-11-11T02:00:00")
        make_run(self.runs, "ssa-a", "2026-11-11T03:00:00")
        make_run(self.runs, "ssa-failed", "2026-11-11T01:00:00", status="failed")
        make_run(self.runs, "ssa-smoke", "2026-11-11T01:30:00", kind="test")
        make_run(self.runs, "ssa-noresults", "2026-11-11T01:45:00", results=False)
        steps = reexecute_batch.plan(self.runs, self.results)
        self.assertEqual([r for r, n, _ in steps if n == 1], ["ssa-b", "ssa-a"])

    def test_second_executions_come_after_all_first_ones_and_done_ones_are_marked(self):
        for i in range(4):
            make_run(self.runs, f"ssa-{i}", f"2026-11-11T0{i}:00:00")
        done = self.results / "ssa-0" / "exec1"
        done.mkdir(parents=True)
        (done / "result.json").write_text("{}")
        steps = reexecute_batch.plan(self.runs, self.results)
        firsts = [s for s in steps if s[1] == 1]
        seconds = [s for s in steps if s[1] == 2]
        self.assertEqual(steps[: len(firsts)], firsts)
        self.assertEqual(len(seconds), 3)
        self.assertTrue([s for s in firsts if s[0] == "ssa-0"][0][2])
        self.assertFalse(any(s[2] for s in seconds))


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runs, self.results = Path(self.tmp.name) / "runs", Path(self.tmp.name) / "results"
        self.runs.mkdir()
        for i in range(3):
            make_run(self.runs, f"ssa-{i}", f"2026-11-11T0{i}:00:00")
        self.calls = []

    def go(self, extra=(), codes=None, busy=lambda cpuset: [], sleep=lambda s: None):
        def execute(command):
            self.calls.append(command)
            run_id, n = command[1], command[3]
            folder = self.results / Path(run_id).name / f"exec{n}"
            folder.mkdir(parents=True)
            (folder / "result.json").write_text("{}")
            return (codes or {}).get(Path(run_id).name, 0)
        argv = ["--runs-dir", str(self.runs), "--result-root", str(self.results), "--tool-image", "t", *extra]
        return reexecute_batch.main(argv, execute=execute, sleep=sleep, busy=busy)

    def test_dry_run_executes_nothing(self):
        self.assertEqual(self.go(["--dry-run"]), 0)
        self.assertEqual(self.calls, [])

    def test_runs_everything_once_and_a_second_pass_does_nothing_new(self):
        self.assertEqual(self.go(), 0)
        first = len(self.calls)
        self.assertEqual(first, 3 + 3)  # three first executions, three second ones (k = 3)
        self.go()
        self.assertEqual(len(self.calls), first)

    def test_a_failed_attempt_is_counted_and_the_batch_goes_on(self):
        self.assertEqual(self.go(codes={"ssa-1": 2}), 2)
        self.assertEqual(len(self.calls), 6)

    def test_the_stop_file_halts_between_executions(self):
        (self.results).mkdir(parents=True)
        (self.results / "REEXEC_STOP").touch()
        self.assertEqual(self.go(), 0)
        self.assertEqual(self.calls, [])

    def test_max_executions(self):
        self.go(["--max-executions", "2"])
        self.assertEqual(len(self.calls), 2)

    def test_waits_while_the_slot_is_busy(self):
        waits = []
        state = {"n": 0}

        def busy(cpuset):
            state["n"] += 1
            return ["ssa-ssa-x"] if state["n"] <= 2 else []
        self.go(["--max-executions", "1"], busy=busy, sleep=waits.append)
        self.assertEqual(len(waits), 2)
        self.assertEqual(len(self.calls), 1)


class CpuTests(unittest.TestCase):
    def test_parse_and_split(self):
        self.assertEqual(reexecute_batch.parse_cpus("10-12,15"), {10, 11, 12, 15})
        self.assertEqual(reexecute_batch.split_cpus("10-17", 2, 4), ["10,11", "12,13", "14,15", "16,17"])
        self.assertEqual(reexecute_batch.split_cpus("10-18", 2, 4), ["10,11", "12,13", "14,15", "16,17"])  # CPU 18 stays unused
        with self.assertRaises(ValueError):
            reexecute_batch.split_cpus("10-12", 2, 4)

    def test_slot_busy_only_when_the_cpus_overlap_the_study_slot(self):
        from unittest import mock
        names = mock.Mock(stdout="ssa-ssa-2026-x homebox\n")
        with mock.patch.object(reexecute_batch.ssa_run, "docker", return_value=names):
            self.assertEqual(reexecute_batch.slot_busy("12,13"), ["ssa-ssa-2026-x"])
            self.assertEqual(reexecute_batch.slot_busy("0-8"), [])
            self.assertEqual(reexecute_batch.slot_busy("9,19"), [])


class ParallelTests(BatchTests):
    def parallel(self, extra=(), execute=None):
        argv = ["--runs-dir", str(self.runs), "--result-root", str(self.results), "--tool-image", "t", "--workers", "2",
                "--cpus-per-worker", "2", "--cpuset", "10-13", "--memory", "3g", *extra]
        return reexecute_batch.main(argv, execute=execute, sleep=lambda s: None, busy=lambda cpuset: [])

    def test_two_workers_run_at_the_same_time_on_separate_cpus_with_the_small_memory(self):
        import threading
        lock, state = threading.Lock(), {"now": 0, "max": 0, "cpusets": []}
        gate = threading.Barrier(2, timeout=5)

        def execute(command):
            cpuset = command[command.index("--cpuset") + 1]
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
                state["cpusets"].append((cpuset, command[command.index("--memory") + 1]))
            try:
                gate.wait()
            except threading.BrokenBarrierError:
                pass
            run_id, n = command[1], command[3]
            folder = self.results / Path(run_id).name / f"exec{n}"
            folder.mkdir(parents=True)
            (folder / "result.json").write_text("{}")
            with lock:
                state["now"] -= 1
            return 0
        self.assertEqual(self.parallel(["--max-executions", "2"], execute), 0)
        self.assertEqual(state["max"], 2)
        self.assertEqual(sorted(c for c, _ in state["cpusets"]), ["10,11", "12,13"])
        self.assertTrue(all(m == "3g" for _, m in state["cpusets"]))

    def test_an_execution_killed_for_memory_is_kept_aside_and_redone_at_full_memory(self):
        calls = []

        def execute(command):
            memory = command[command.index("--memory") + 1]
            run_id, n = Path(command[1]).name, command[3]
            calls.append((run_id, n, memory))
            folder = self.results / run_id / f"exec{n}"
            folder.mkdir(parents=True)
            killed = memory == "3g" and run_id == "ssa-1"
            (folder / "result.json").write_text(json.dumps({"oom_killed": killed}))
            return 2 if killed else 0
        self.parallel(["--max-executions", "3"], execute)
        self.assertIn(("ssa-1", "1", "11g"), calls)
        self.assertTrue((self.results / "ssa-1" / "exec1_killed_at_3g" / "result.json").exists())
        self.assertFalse(json.loads((self.results / "ssa-1" / "exec1" / "result.json").read_text())["oom_killed"])

    def test_too_many_workers_for_the_cpus_is_not_started(self):
        argv = ["--runs-dir", str(self.runs), "--result-root", str(self.results), "--tool-image", "t", "--workers", "4",
                "--cpus-per-worker", "2", "--cpuset", "10-12"]
        self.assertEqual(reexecute_batch.main(argv, execute=lambda c: 0), 3)


if __name__ == "__main__":
    unittest.main()
