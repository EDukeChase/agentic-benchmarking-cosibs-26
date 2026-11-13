import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import reexecute_repair


class Setup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.results = root / "results"
        self.run = root / "runs" / "ssa-x-abcd"
        (self.run / "out" / "code").mkdir(parents=True)
        (self.run / "out" / "code" / "reproduce.py").write_text("import nothing_here\n")
        (self.run / "records").mkdir()
        (self.run / "records" / "packages.txt").write_text("numpy==1\n")
        self.brief = root / "brief.md"
        self.brief.write_text("brief text")
        self.exec_calls, self.agent_calls = [], []

    def write_result(self, folder: str, first_attempt: str, cause=None, **extra):
        d = self.results / "ssa-x-abcd" / folder
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps({"first_attempt": first_attempt, "failure_cause_guess": cause,
                                                    "environment_image": "img", "all_within_tolerance": True, **extra}))
        (d / "reexec.log").write_text("ModuleNotFoundError: x")

    def run_main(self, first=("failed", "environment"), ran_on_cycle=None, edits=True, extra=()):
        """Fake execute: the first attempt gets `first`; a repaired attempt runs OK from cycle `ran_on_cycle`."""
        def execute(command):
            self.exec_calls.append(command)
            if "--repaired" not in command:
                self.write_result("exec1", first[0], first[1])
                return 0 if first[0] == "ran" else 2
            subdir = command[command.index("--result-subdir") + 1]
            cycle = int(subdir[-1])
            ok = ran_on_cycle is not None and cycle >= ran_on_cycle
            self.write_result(subdir, "ran" if ok else "failed", None if ok else "logic")
            return 0 if ok else 2

        def agent(workspace, brief, task, minutes, turns, budget, model):
            self.agent_calls.append(task)
            if edits:
                (workspace / "code" / "reproduce.py").write_text(f"# edit {len(self.agent_calls)}\n")
                (workspace / "CHANGE_NOTE.md").write_text("note")
            return {"seconds": 1.0, "timed_out": False, "num_turns": 3, "cost_usd_list_price": 0.05, "tokens": {"input": 1}, "models": ["m"], "is_error": False}
        argv = ["--run-dir", str(self.run), "--tool-image", "t", "--brief", str(self.brief), "--result-root", str(self.results), *extra]
        with mock.patch.object(reexecute_repair.reexecute_run, "installed_packages", return_value=["numpy==2"]), \
             mock.patch.object(reexecute_repair, "data_file_names", return_value="patients.csv\n"):
            code = reexecute_repair.main(argv, execute=execute, agent=agent)
        return code, json.loads((self.results / "ssa-x-abcd" / "exec1" / "repair.json").read_text())


class RepairTests(Setup):
    def test_no_repair_when_the_first_attempt_ran(self):
        code, summary = self.run_main(first=("ran", None))
        self.assertEqual(code, 0)
        self.assertFalse(summary["repair_attempted"])
        self.assertEqual(self.agent_calls, [])

    def test_a_time_limit_or_memory_kill_is_not_sent_to_the_agent(self):
        for cause in ("time limit", "out of memory"):
            self.setUp()
            code, summary = self.run_main(first=("failed", cause))
            self.assertEqual(code, 2)
            self.assertFalse(summary["repair_attempted"])
            self.assertIn("not repairable", summary["note"])

    def test_repaired_on_the_second_cycle(self):
        code, summary = self.run_main(ran_on_cycle=2)
        self.assertEqual(code, 0)
        self.assertTrue(summary["repaired_and_ran"])
        self.assertEqual(len(summary["cycles"]), 2)
        self.assertEqual(summary["note"], "repaired and ran")
        self.assertEqual(self.agent_calls[0], reexecute_repair.TASK_FIRST)
        self.assertEqual(self.agent_calls[1], reexecute_repair.TASK_AGAIN)
        self.assertEqual(summary["first_attempt"], "failed")  # the first attempt stays on record as failed
        self.assertTrue((self.results / "ssa-x-abcd" / "exec1_repair2" / "changes.diff").read_text())
        self.assertTrue((self.results / "ssa-x-abcd" / "exec1_repair1" / "change_note.md").exists())

    def test_gives_up_after_the_cycle_cap(self):
        code, summary = self.run_main(ran_on_cycle=None, extra=["--max-cycles", "2"])
        self.assertEqual(code, 2)
        self.assertEqual(len(summary["cycles"]), 2)
        self.assertFalse(summary["repaired_and_ran"])
        self.assertEqual(summary["note"], "could not be repaired")

    def test_an_agent_that_changes_nothing_stops_the_loop(self):
        code, summary = self.run_main(edits=False)
        self.assertEqual(len(summary["cycles"]), 1)
        self.assertIn("changed nothing", summary["cycles"][0]["note"])
        self.assertEqual(len([c for c in self.exec_calls if "--repaired" in c]), 0)

    def test_the_repaired_run_uses_the_edited_copy_and_is_labelled(self):
        self.run_main(ran_on_cycle=1)
        repaired = [c for c in self.exec_calls if "--repaired" in c][0]
        self.assertIn("--code-dir", repaired)
        self.assertTrue(repaired[repaired.index("--code-dir") + 1].endswith("repair_workspace/code"))
        self.assertEqual(repaired[repaired.index("--result-subdir") + 1], "exec1_repair1")

    def test_the_original_code_is_never_changed(self):
        self.run_main(ran_on_cycle=1)
        self.assertEqual((self.run / "out" / "code" / "reproduce.py").read_text(), "import nothing_here\n")

    def test_the_brief_hash_and_cost_are_recorded(self):
        code, summary = self.run_main(ran_on_cycle=1)
        self.assertEqual(summary["brief_sha256"], reexecute_repair.sha256_text("brief text"))
        self.assertAlmostEqual(summary["total_cost_usd_list_price"], 0.05)

    def test_a_missing_brief_is_not_started(self):
        self.brief.unlink()
        argv = ["--run-dir", str(self.run), "--tool-image", "t", "--brief", str(self.brief), "--result-root", str(self.results)]
        self.assertEqual(reexecute_repair.main(argv, execute=lambda c: 0, agent=lambda *a: {}), 3)


class AgentCallTests(unittest.TestCase):
    def test_the_command_is_restricted_and_the_output_is_parsed(self):
        out = json.dumps({"is_error": False, "num_turns": 2, "total_cost_usd": 0.02, "session_id": "s", "result": "done",
                          "usage": {"input_tokens": 1, "output_tokens": 2, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 4},
                          "modelUsage": {"claude-x": {}}})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(reexecute_repair.subprocess, "run",
                                                                   return_value=mock.Mock(stdout=out, stderr="")) as run:
            record = reexecute_repair.call_agent(Path(tmp), Path("/b.md"), "task", 15, 15, 2.0, None)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--tools") + 1], "Read,Edit,Write,Glob,Grep")
        self.assertNotIn("Bash", " ".join(command))
        self.assertIn("--no-session-persistence", command)
        self.assertEqual(run.call_args.kwargs["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"], "1")
        self.assertEqual(record["tokens"]["output"], 2)
        self.assertEqual(record["models"], ["claude-x"])

    def test_unparseable_output_is_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(reexecute_repair.subprocess, "run",
                                                                   return_value=mock.Mock(stdout="not json", stderr="e")):
            record = reexecute_repair.call_agent(Path(tmp), Path("/b.md"), "task", 15, 15, 2.0, None)
        self.assertFalse(record["parsed"])


class DataNamesTests(unittest.TestCase):
    def test_only_file_names_are_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "b.csv").write_text("secret,values\n1,2\n")
            (Path(tmp) / "a.csv").write_text("x")
            text = reexecute_repair.data_file_names(Path(tmp))
        self.assertEqual(text, "a.csv\nb.csv\n")
        self.assertNotIn("secret", text)

    def test_a_missing_folder_gives_an_empty_list(self):
        self.assertEqual(reexecute_repair.data_file_names(Path("/no/such/folder")), "")


class DiffTests(unittest.TestCase):
    def test_diff_shows_an_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            a.mkdir(), b.mkdir()
            (a / "x.py").write_text("one\n")
            (b / "x.py").write_text("two\n")
            self.assertIn("-one", reexecute_repair.make_diff(a, b))
            self.assertEqual(reexecute_repair.make_diff(a, a), "")


if __name__ == "__main__":
    unittest.main()
