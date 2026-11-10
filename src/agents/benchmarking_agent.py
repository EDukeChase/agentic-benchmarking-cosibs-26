"""The benchmarking stage: one agent that runs the whole analysis and writes the deliverables."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from deepagents import create_deep_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool

from src.agents.file_backend import BoundedFilesystemBackend
from src.core.schemas import LiteratureReviewResult
from src.settings.config import EXECUTE_PYTHON_MAX_SECONDS, LLMConfig, StudyTaskConfig
from src.settings.prompts import BENCHMARKING_STAGE_INSTRUCTIONS
from src.settings.study_prompt import StudyPrompt, compose
from src.utils.naming import slugify
from src.utils.sandbox import run_as_agent

from .chat_model import build_chat_model

MAX_TOOL_OUTPUT_CHARS = 20_000


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return f"[first part cut: {len(text) - MAX_TOOL_OUTPUT_CHARS} characters]\n" + text[-MAX_TOOL_OUTPUT_CHARS:]


@dataclass
class ExecutionStats:
    """How often the agent ran code, and how often a run ended in an error or a timeout."""

    runs: int = 0
    errored: int = 0


def make_execute_python(task: StudyTaskConfig, deadline_monotonic: float | None = None,
                        stats: ExecutionStats | None = None):
    """Build the agent's code-running tool for this task.

    The code runs in a fresh Python process, as the unprivileged agent user when
    the container sets one up (see src/utils/sandbox.py), whose working directory is
    the output folder. A single call is capped at EXECUTE_PYTHON_MAX_SECONDS and never runs
    past the pipeline deadline. This does not by itself stop the code from opening
    other paths: what the code can read is decided by the container (see the run
    script), not by this function.
    """

    @tool
    def execute_python(code: str, timeout: int = 600) -> str:
        """Run Python code in the project environment and return stdout and stderr.

        The working directory is the output folder. timeout is in seconds.
        """
        limit = min(max(int(timeout), 1), EXECUTE_PYTHON_MAX_SECONDS)
        if stats is not None:
            stats.runs += 1
        if deadline_monotonic is not None:
            limit = min(limit, max(int(deadline_monotonic - time.monotonic()), 1))
        try:
            result = run_as_agent(["python", "-c", code], cwd=task.out_dir, timeout=limit)
        except subprocess.TimeoutExpired as exc:
            if stats is not None:
                stats.errored += 1
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return _clip(
                f"Execution exceeded the {limit}-second limit for one call.\n"
                f"Partial stdout:\n{stdout}\n\nPartial stderr:\n{stderr}"
            )
        if stats is not None and result.returncode != 0:
            stats.errored += 1
        return _clip(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nExit code: {result.returncode}")

    return execute_python


def build_benchmarking_agent(
    task: StudyTaskConfig,
    llm_config: LLMConfig = LLMConfig(),
    deadline_monotonic: float | None = None,
    stats: ExecutionStats | None = None,
):
    """File tools are confined to the task folders; code runs through execute_python."""
    return create_deep_agent(
        model=build_chat_model(llm_config),
        tools=[make_execute_python(task, deadline_monotonic, stats)],
        backend=BoundedFilesystemBackend(root_dir=task.task_root, virtual_mode=True),
    )


def _virtual(task: StudyTaskConfig, path: str) -> str:
    """The path as the file tools see it: relative to the task root, with a leading slash."""
    return "/" + os.path.relpath(path, task.task_root)


def run_benchmarking_agent(
    agent,
    task: StudyTaskConfig,
    study_prompt: StudyPrompt,
    literature_result: LiteratureReviewResult,
    missing_methods: list[str] | None = None,
):
    Path(task.out_dir).mkdir(parents=True, exist_ok=True)
    stage_text = BENCHMARKING_STAGE_INSTRUCTIONS.format(
        methods_dir=task.methods_dir, out_dir=task.out_dir, data_dir=task.data_dir,
    ) + (
        "\nPath note: the file tools (ls, read_file, write_file, edit_file) show "
        f"the data folder as {_virtual(task, task.data_dir)} and the output folder as "
        f"{_virtual(task, task.out_dir)}. execute_python uses the real paths given above.\n"
    )
    system_prompt = compose(study_prompt, stage_text)

    methods = "\n".join(
        f"- {slugify(c.model_name)}: {c.model_name} (source: {c.resource_link}, {c.source_type})"
        for c in literature_result.candidates
    )
    human_message = (
        "Methods 3 to 5, already implemented under "
        f"{task.methods_dir}:\n{methods}\n\n"
        "Literature review that chose them:\n"
        f"{literature_result.model_dump_json(indent=2, exclude={'grounding', 'gemini_usage'})}\n\n"
        + (
            "The programming stage produced no model.py for: "
            f"{', '.join(missing_methods)}. Record each of these as failed in results.json, "
            "with that as the failure_reason, unless you can implement it yourself from the review.\n\n"
            if missing_methods else ""
        )
        + "Do the whole analysis and write every deliverable except report.md."
    )
    return agent.invoke({"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]})
