import json
import tempfile
import unittest
from pathlib import Path

from langchain.tools import tool

from src.utils.tool_log import MAX_ARGUMENT_CHARS, ToolCallLog


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class ToolCallLogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "tool_calls.jsonl"

    def test_a_call_writes_a_start_and_an_end_line(self):
        @tool
        def add(a: int, b: int) -> str:
            """Add two numbers."""
            return "the sum is " + str(a + b)

        add.invoke({"a": 2, "b": 3}, config={"callbacks": [ToolCallLog(self.path, "benchmarking")]})
        start, end = read_log(self.path)
        self.assertEqual((start["event"], start["tool"], start["stage"]), ("start", "add", "benchmarking"))
        self.assertIn('"a": 2', start["arguments"])
        self.assertEqual((end["event"], end["call"]), ("end", start["call"]))
        self.assertEqual(end["output_chars"], len("the sum is 5"))
        self.assertIn("pipeline_rss_gb", end)

    def test_a_failing_tool_writes_an_error_line(self):
        @tool
        def boom(x: int) -> str:
            """Always fails."""
            raise ValueError("no good")

        with self.assertRaises(ValueError):
            boom.invoke({"x": 1}, config={"callbacks": [ToolCallLog(self.path, "programming")]})
        events = [r["event"] for r in read_log(self.path)]
        self.assertEqual(events, ["start", "error"])
        self.assertIn("no good", read_log(self.path)[1]["error"])

    def test_a_call_that_never_ends_leaves_only_its_start_line(self):
        # This is what a run killed by the memory limit leaves behind.
        log = ToolCallLog(self.path, "benchmarking")
        log.on_tool_start({"name": "read_file"}, "{'file_path': '/data/observations.csv'}", run_id="abc")
        records = read_log(self.path)
        self.assertEqual([r["event"] for r in records], ["start"])
        self.assertIn("observations.csv", records[0]["arguments"])

    def test_long_arguments_are_shortened(self):
        log = ToolCallLog(self.path, "benchmarking")
        log.on_tool_start({"name": "execute_python"}, "x" * 5000, run_id="abc")
        self.assertLess(len(read_log(self.path)[0]["arguments"]), MAX_ARGUMENT_CHARS + 60)
        self.assertIn("5000 characters", read_log(self.path)[0]["arguments"])

    def test_stages_append_to_the_same_file(self):
        ToolCallLog(self.path, "programming").on_tool_start({"name": "ls"}, "{}", run_id="1")
        ToolCallLog(self.path, "benchmarking").on_tool_start({"name": "ls"}, "{}", run_id="2")
        self.assertEqual([r["stage"] for r in read_log(self.path)], ["programming", "benchmarking"])


if __name__ == "__main__":
    unittest.main()
