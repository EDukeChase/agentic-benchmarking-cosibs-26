"""A line-by-line log of every tool call an agent makes, written as the calls happen.

Each line is flushed at once, so a run that is killed (for example by the memory limit) still
shows which call was in progress: its "start" line has no matching "end" line. Each line also
holds the pipeline process's own memory use, which tells the pipeline's memory apart from the
agent code's.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

MAX_ARGUMENT_CHARS = 400
MAX_ERROR_CHARS = 400


def process_rss_gb() -> float | None:
    """Resident memory of this process in GB, from /proc (None where that is unavailable)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024 / 1024, 3)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _short(value, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text)} characters in all]"


class ToolCallLog(BaseCallbackHandler):
    """Appends one JSON line per tool start, end and error to a file."""

    raise_error = False  # a logging problem must never fail a run

    def __init__(self, path: str | Path, stage: str):
        self.path = Path(path)
        self.stage = stage
        self._open: dict = {}

    def _write(self, record: dict) -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stage": self.stage, **record,
                  "pipeline_rss_gb": process_rss_gb()}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        self._open[run_id] = (name, time.monotonic())
        self._write({"event": "start", "call": str(run_id), "tool": name,
                     "arguments": _short(inputs if inputs is not None else input_str, MAX_ARGUMENT_CHARS)})

    def on_tool_end(self, output, *, run_id, **kwargs):
        name, began = self._open.pop(run_id, ("unknown", time.monotonic()))
        content = getattr(output, "content", output)
        self._write({"event": "end", "call": str(run_id), "tool": name,
                     "seconds": round(time.monotonic() - began, 2), "output_chars": len(str(content))})

    def on_tool_error(self, error, *, run_id, **kwargs):
        name, began = self._open.pop(run_id, ("unknown", time.monotonic()))
        self._write({"event": "error", "call": str(run_id), "tool": name,
                     "seconds": round(time.monotonic() - began, 2),
                     "error": _short(f"{type(error).__name__}: {error}", MAX_ERROR_CHARS)})
