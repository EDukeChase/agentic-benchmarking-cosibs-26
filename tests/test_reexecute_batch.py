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


if __name__ == "__main__":
    unittest.main()
