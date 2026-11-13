import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import reexecute_run


RAN = {"exit_code": 0, "timed_out": False, "oom_killed": False, "wall_seconds": 600.0,
       "memory": {"anon_peak_sampled_gb": 2.0, "current_peak_sampled_gb": 3.0, "memory_samples": 120}}


def method(ok=True):
    return {"name_reported": "m", "within_tolerance": ok, "taxonomy": "none" if ok else "b"}


def assemble(ran=RAN, found=True, compare=None, log=""):
    return reexecute_run.assemble(
        run_id="ssa-x-abcd", execution=1, record={"image": "ssa-study:efb8111"}, machine="box", started="s", finished="f", ran=ran,
        outputs_found=found, compare=compare, environment_image="ssa-study:efb8111", image_note=None,
        packages={"missing_in_image": []}, cpuset="10-18", memory="11g", tolerances_sha="abc", log_text=log)


class ContainerCommandTests(unittest.TestCase):
    def setUp(self):
        self.command = reexecute_run.container_command(
            name="n", image="img", code_dir=Path("/c"), out_dir=Path("/o"), data_dir=Path("/d"), cpuset="10-18", memory="11g")
        self.text = " ".join(self.command)

    def test_no_network_no_privileges_and_the_study_limits(self):
        for part in ("--network none", "--cpuset-cpus 10-18", "--memory 11g", "--memory-swap 11g", "--cap-drop ALL",
                     "no-new-privileges", "--user 2000:2000", "--runtime runc"):
            self.assertIn(part, self.text)
        self.assertNotIn("--gpus", self.text)

    def test_data_is_read_only_and_the_code_runs_unchanged(self):
        self.assertIn("source=/d,target=/home/llmrel/task/data,readonly", self.text)
        self.assertTrue(self.text.endswith("/work/code/reproduce.py --data-dir /home/llmrel/task/data --out-dir /work/out"))
        self.assertNotIn("source=/o,target=/work/out,readonly", self.text)

    def test_the_study_thread_settings_are_passed(self):
        self.assertIn("OMP_NUM_THREADS=8", self.text)


class AssembleTests(unittest.TestCase):
    def test_ran_and_within_tolerance(self):
        result = assemble(compare={"methods": [method(), method()]})
        self.assertEqual(result["first_attempt"], "ran")
        self.assertEqual(result["taxonomy_run_level"], "none")
        self.assertTrue(result["all_within_tolerance"])
        self.assertEqual(result["wall_min"], 10.0)

    def test_ran_but_a_method_differs_is_type_b(self):
        result = assemble(compare={"methods": [method(), method(False)]})
        self.assertEqual(result["taxonomy_run_level"], "b")
        self.assertFalse(result["all_within_tolerance"])

    def test_nonzero_exit_or_missing_outputs_is_a_failed_first_attempt(self):
        failed = assemble(ran={**RAN, "exit_code": 1}, compare=None, log="ModuleNotFoundError: No module named 'x'")
        self.assertEqual(failed["first_attempt"], "failed")
        self.assertEqual(failed["taxonomy_run_level"], "a")
        self.assertEqual(failed["failure_cause_guess"], "environment")
        self.assertEqual(assemble(found=False)["first_attempt"], "failed")

    def test_time_limit_and_memory_kill_are_named(self):
        self.assertEqual(assemble(ran={**RAN, "timed_out": True})["failure_cause_guess"], "time limit")
        self.assertEqual(assemble(ran={**RAN, "exit_code": 137, "oom_killed": True})["failure_cause_guess"], "out of memory")

    def test_a_logic_error_is_guessed_as_logic(self):
        self.assertEqual(assemble(ran={**RAN, "exit_code": 1}, log="KeyError: 'AGE'")["failure_cause_guess"], "logic")

    def test_a_method_that_could_not_be_judged_blocks_all_within_tolerance(self):
        result = assemble(compare={"methods": [method(), {"name_reported": "f", "within_tolerance": None}]})
        self.assertFalse(result["all_within_tolerance"])


class ChooseImagesTests(unittest.TestCase):
    def docker(self, current_id):
        return mock.Mock(stdout=current_id + "\n")

    def test_the_runs_own_image_is_used_when_it_is_the_same_image(self):
        with mock.patch.object(reexecute_run, "image_exists", return_value=True), \
             mock.patch.object(reexecute_run.ssa_run, "docker", return_value=self.docker("sha256:a")):
            self.assertEqual(reexecute_run.choose_images({"image": "t:1", "image_id": "sha256:a"}, None, "tool"), ("t:1", None))

    def test_a_retagged_image_is_used_but_said(self):
        with mock.patch.object(reexecute_run, "image_exists", return_value=True), \
             mock.patch.object(reexecute_run.ssa_run, "docker", return_value=self.docker("sha256:b")):
            image, note = reexecute_run.choose_images({"image": "t:1", "image_id": "sha256:a"}, None, "tool")
        self.assertEqual(image, "t:1")
        self.assertIn("different image", note)

    def test_a_missing_image_falls_back_to_the_tool_image_and_says_so(self):
        with mock.patch.object(reexecute_run, "image_exists", return_value=False):
            image, note = reexecute_run.choose_images({"image": "gone:1"}, None, "tool")
        self.assertEqual(image, "tool")
        self.assertIn("not on this machine", note)

    def test_a_given_image_wins(self):
        self.assertEqual(reexecute_run.choose_images({"image": "t:1"}, "given", "tool")[0], "given")


class LogAndGuardTests(unittest.TestCase):
    def test_one_line_per_execution_with_a_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = assemble(compare={"methods": [method()]})
            reexecute_run.append_log(Path(tmp), result)
            reexecute_run.append_log(Path(tmp), result, note="x")
            rows = list(csv.reader((Path(tmp) / "reexec_log.csv").open()))
        self.assertEqual(rows[0], reexecute_run.LOG_HEADER)
        self.assertEqual(len(rows), 3)

    def test_a_run_without_success_or_outputs_is_not_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "ssa-x"
            run.mkdir()
            (run / "run_record.json").write_text('{"pipeline_status": "failed"}')
            self.assertEqual(reexecute_run.main(["--run-dir", str(run), "--tool-image", "t", "--result-root", tmp]), 3)

    def test_dry_run_runs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "ssa-x-1234"
            (run / "out" / "code").mkdir(parents=True)
            for name in ("code/reproduce.py", "results.json", "predictions.csv"):
                (run / "out" / name).write_text("x")
            (run / "run_record.json").write_text('{"pipeline_status": "success", "image": "ssa-study:nope"}')
            with mock.patch.object(reexecute_run.ssa_run, "image_meta", return_value={}), \
                 mock.patch.object(reexecute_run.ssa_run, "data_mount_for", return_value=Path("/d")), \
                 mock.patch.object(reexecute_run, "image_exists", return_value=False), \
                 mock.patch.object(reexecute_run, "run_container") as run_container:
                code = reexecute_run.main(["--run-dir", str(run), "--tool-image", "tool", "--result-root", tmp, "--dry-run"])
        self.assertEqual(code, 0)
        run_container.assert_not_called()


if __name__ == "__main__":
    unittest.main()
