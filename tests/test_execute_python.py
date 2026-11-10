import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents.benchmarking_agent import ExecutionStats, make_execute_python
from src.settings.config import StudyTaskConfig


class ExecutePythonTests(unittest.TestCase):
    def setUp(self):
        # The study image sets AGENT_UID; these tests are about the tool, not the sandbox.
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AGENT_UID", None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "task" / "data").mkdir(parents=True)
        (root / "task" / "out").mkdir(parents=True)
        self.task = StudyTaskConfig(data_dir=str(root / "task" / "data"), out_dir=str(root / "task" / "out"),
                                    records_dir=str(root / "records"))
        self.stats = ExecutionStats()
        self.tool = make_execute_python(self.task, stats=self.stats)

    def test_runs_in_the_output_folder(self):
        out = self.tool.invoke({"code": "import os; print(os.getcwd()); open('x.txt', 'w').write('hi')"})
        self.assertIn(self.task.out_dir, out)
        self.assertTrue((Path(self.task.out_dir) / "x.txt").exists())

    def test_counts_runs_and_errors(self):
        self.tool.invoke({"code": "print(1)"})
        self.tool.invoke({"code": "raise SystemExit(3)"})
        self.tool.invoke({"code": "import time; time.sleep(5)", "timeout": 1})
        self.assertEqual((self.stats.runs, self.stats.errored), (3, 2))

    def test_long_output_is_cut(self):
        out = self.tool.invoke({"code": "print('x' * 50000)"})
        self.assertLess(len(out), 25000)
        self.assertIn("cut", out)


if __name__ == "__main__":
    unittest.main()
