"""Run the four-stage pipeline (SSA) once on the study task.

Stages: literature search (methods 3 to 5), programming (code for those three),
benchmarking (the whole analysis, all five methods, every deliverable except
report.md) and reporting (report.md). Every agent gets the same fixed study
prompt. The pipeline writes its own records (manifest, literature result, split
check) to the records folder, which the agents cannot see.

A stage that fails does not stop the run when the later stages can still produce
something. The run is a success when results.json has at least one method with
status "ok", as in the study prompt.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.agents.benchmarking_agent import ExecutionStats, build_benchmarking_agent, run_benchmarking_agent
from src.agents.literature_agent import build_literature_agent, run_literature_review
from src.agents.programming_agent import (
    build_programming_agent,
    collect_generated_models,
    run_programming_agent,
)
from src.agents.reporting_agent import build_reporting_agent, write_report
from src.evaluation.split_usage import UnknownPatientIdsError, record_split_usage
from src.fixtures.base_literature import load_base_literature
from src.settings.config import (
    LITERATURE_STAGE_SECONDS,
    PROGRAMMING_STAGE_SECONDS,
    REPORTING_STAGE_SECONDS,
    TOTAL_DEADLINE_SECONDS,
    ExperimentConfig,
    StudyTaskConfig,
)
from src.settings.study_prompt import load_study_prompt
from src.utils.messages import message_text
from src.utils.naming import slugify
from src.utils.retry_log import RetryCounter
from src.utils.sandbox import agent_identity, run_as_agent
from src.utils.telemetry import collect_token_usage

EXPERIMENT = ExperimentConfig()


class StageTimeoutError(TimeoutError):
    """Raised when a pipeline stage exceeds its wall-clock limit."""


@contextmanager
def stage_timeout(stage: str, seconds: float):
    """Bound a stage with SIGALRM (Linux, main thread). Does nothing elsewhere."""
    seconds = int(seconds)
    if seconds <= 0:
        raise StageTimeoutError(f"Stage '{stage}' has no time left")
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise StageTimeoutError(f"Stage '{stage}' exceeded its {seconds}-second time limit")

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str))


def _summarize_results(out_dir: Path) -> dict:
    """What results.json says, reduced to what the manifest needs. Never raises."""
    path = out_dir / "results.json"
    if not path.exists():
        return {"results_json": "missing", "methods": {}}
    try:
        results = json.loads(path.read_text())
        methods = {
            name: {"status": m.get("status"), "attempts": m.get("attempts"), "source": m.get("source")}
            for name, m in results["methods"].items()
        }
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
        return {"results_json": "unreadable", "methods": {}}
    return {"results_json": "present", "methods": methods}


GB = 1024 ** 3


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    text = _read_text(path)
    try:
        return int(text.split()[0]) if text else None
    except ValueError:
        return None


def _keyed(path: str) -> dict[str, int]:
    """A cgroup file of "key value" lines as a dict."""
    values = {}
    for line in (_read_text(path) or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            values[parts[0]] = int(parts[1])
    return values


def _gb(value: int | None) -> float | None:
    return None if value is None else round(value / GB, 2)


def _resource_use() -> dict:
    """The container's cgroup (v2), read at the end of the run, in the AISS records' layout.

    memory.peak includes the kernel's file cache, so after a big read it sits at the cap
    even when the processes used far less. Use oom_kills as the failure signal and
    anon (process memory) and file (cache) separately. Best effort; None where unreadable.
    """
    stat = _keyed("/sys/fs/cgroup/memory.stat")
    return {
        "peak_memory_gb": _gb(_read_int("/sys/fs/cgroup/memory.peak")),
        "memory_at_finish_gb": _gb(_read_int("/sys/fs/cgroup/memory.current")),
        "oom_kills": _keyed("/sys/fs/cgroup/memory.events").get("oom_kill"),
        "cpu_seconds": (lambda usec: None if usec is None else round(usec / 1e6, 1))(_keyed("/sys/fs/cgroup/cpu.stat").get("usage_usec")),
        "anon_at_finish_gb": _gb(stat.get("anon")),
        "file_cache_at_finish_gb": _gb(stat.get("file")),
        "peak_includes_file_cache": True,
    }


def _limits_applied() -> dict:
    """The limits the container's cgroup enforces, as the process sees them."""
    mem_max, swap_max = _read_text("/sys/fs/cgroup/memory.max"), _read_text("/sys/fs/cgroup/memory.swap.max")
    as_gb = lambda text: text if text in (None, "max") else _gb(int(text))  # noqa: E731
    return {"cpus": _read_text("/sys/fs/cgroup/cpuset.cpus.effective"), "mem_max_gb": as_gb(mem_max), "swap_max_gb": as_gb(swap_max)}


def _write_package_list(path: Path) -> None:
    """Packages visible to agent code (system site plus anything the agents installed)."""
    try:
        result = run_as_agent(["python", "-m", "pip", "list", "--format=freeze"], cwd="/tmp", timeout=120)
        path.write_text(result.stdout)
    except Exception as error:  # noqa: BLE001 - a record must never fail the run
        path.write_text(f"package list unavailable: {error}\n")


def main() -> int:
    if agent_identity() is not None:
        os.umask(0)  # files the pipeline writes into the task folders must be writable by agent code
    task = StudyTaskConfig.from_env()
    experiment = EXPERIMENT
    run_id = os.getenv("BENCHMARK_RUN_ID", uuid.uuid4().hex[:8])
    out_dir, records_dir = Path(task.out_dir), Path(task.records_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise RuntimeError(f"Output folder {out_dir} is not empty; every run starts from an empty one.")
    if any(records_dir.iterdir()):
        raise RuntimeError(f"Records folder {records_dir} is not empty; every run starts from an empty one.")

    study_prompt = load_study_prompt(task.study_prompt_file, os.getenv("STUDY_PROMPT_SHA256"))
    total_seconds = float(os.getenv("STUDY_DEADLINE_SECONDS", TOTAL_DEADLINE_SECONDS))
    started_monotonic = time.monotonic()
    deadline = started_monotonic + total_seconds

    def remaining() -> float:
        return deadline - time.monotonic()

    started_at = datetime.now(timezone.utc)
    stage_timings: dict[str, float] = {}
    stage_failures: dict[str, str] = {}
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    literature_result = None
    literature_setting = "live"
    split_check = None
    run_status = "failed"
    failed_stage = None
    stage = "initialization"
    print(f"Starting run {run_id} (study prompt SHA-256 {study_prompt.sha256})")

    retry_counter = RetryCounter().install()
    stage_tokens: dict[str, dict[str, int]] = {}
    model_versions: dict[str, str | None] = {}
    execution_stats = ExecutionStats()
    web_searches = {"queries": [], "result_urls": []}

    def add_tokens(response, stage_name: str) -> None:
        usage = collect_token_usage(response)
        last = response["messages"][-1] if isinstance(response, dict) and response.get("messages") else response
        metadata = getattr(last, "response_metadata", None) or {}
        model_versions[stage_name] = metadata.get("model_name") or metadata.get("model")
        stage_tokens[stage_name] = usage
        for key in token_usage:
            token_usage[key] += usage[key]

    try:
        # --- literature -----------------------------------------------------------
        stage = "literature"
        started = time.perf_counter()
        if os.getenv("USE_LITERATURE_FIXTURE", "").lower() in {"1", "true", "yes"}:
            print("USE_LITERATURE_FIXTURE set: using the local fixture instead of a live search (not for study runs).")
            literature_setting = "fixture"
            literature_result = load_base_literature(num_models=experiment.number_of_models)
        else:
            print(f"Searching with {experiment.literature_llm.model} for {experiment.number_of_models} methods...")
            literature_agent = build_literature_agent(experiment.literature_llm)
            with stage_timeout(stage, min(LITERATURE_STAGE_SECONDS, remaining())):
                literature_result = run_literature_review(literature_agent, experiment.number_of_models, study_prompt)
        _write_json(records_dir / "literature.json", literature_result.model_dump())
        grounding = literature_result.grounding or {}
        web_searches["queries"] = grounding.get("web_search_queries", [])
        web_searches["result_urls"] = [src.get("uri") for src in grounding.get("sources", [])]
        # Gemini's links are Vertex redirects; the title is the page's site.
        web_searches["result_sites"] = [src.get("title") for src in grounding.get("sources", [])]
        stage_timings[stage] = time.perf_counter() - started

        # --- programming ----------------------------------------------------------
        stage = "programming"
        started = time.perf_counter()
        try:
            print(f"Running the programming agent with {experiment.programming_llm.model}...")
            programming_agent = build_programming_agent(task.methods_dir, experiment.programming_llm)
            with stage_timeout(stage, min(PROGRAMMING_STAGE_SECONDS, remaining() - REPORTING_STAGE_SECONDS)):
                response = run_programming_agent(programming_agent, literature_result, study_prompt)
            add_tokens(response, "programming")
        except (Exception, StageTimeoutError) as error:  # later stages can still run methods 1 and 2
            stage_failures[stage] = f"{type(error).__name__}: {error}"
            print(f"Programming stage failed ({stage_failures[stage]}); continuing.")
        stage_timings[stage] = time.perf_counter() - started
        expected = [slugify(c.model_name) for c in literature_result.candidates]
        try:
            implemented = {m.model_name for m in collect_generated_models(task.methods_dir)}
        except FileNotFoundError:
            implemented = set()
        missing = [name for name in expected if name not in implemented]
        if missing:
            stage_failures.setdefault("programming", f"no model.py for: {missing}")

        # --- benchmarking ---------------------------------------------------------
        stage = "benchmarking"
        started = time.perf_counter()
        print(f"Running the benchmarking agent with {experiment.benchmarking_llm.model}...")
        benchmarking_agent = build_benchmarking_agent(
            task, experiment.benchmarking_llm, deadline_monotonic=deadline - REPORTING_STAGE_SECONDS,
            stats=execution_stats,
        )
        with stage_timeout(stage, remaining() - REPORTING_STAGE_SECONDS):
            benchmarking_response = run_benchmarking_agent(
                benchmarking_agent, task, study_prompt, literature_result, missing,
            )
        add_tokens(benchmarking_response, "benchmarking")
        benchmarking_message = message_text(benchmarking_response["messages"][-1])
        _write_json(records_dir / "benchmarking_message.json", {"final_message": benchmarking_message})
        stage_timings[stage] = time.perf_counter() - started

        # --- record the split usage (never fails a run, except for unknown IDs) ----
        stage = "split check"
        split_check = record_split_usage(task.out_dir, task.data_dir, task.records_dir)

        # --- reporting ------------------------------------------------------------
        stage = "reporting"
        started = time.perf_counter()
        try:
            print(f"Writing the report with {experiment.reporting_llm.model}...")
            reporting_llm = build_reporting_agent(experiment.reporting_llm)
            with stage_timeout(stage, min(REPORTING_STAGE_SECONDS, remaining())):
                report = write_report(reporting_llm, study_prompt, out_dir, literature_result, benchmarking_message)
            add_tokens(report, "reporting")
            (out_dir / "report.md").write_text(message_text(report))
        except (Exception, StageTimeoutError) as error:
            stage_failures[stage] = f"{type(error).__name__}: {error}"
            print(f"Reporting stage failed ({stage_failures[stage]}).")
        stage_timings[stage] = time.perf_counter() - started
        stage = "done"
    except (Exception, KeyboardInterrupt) as error:
        failed_stage = stage
        stage_failures[stage] = f"{type(error).__name__}: {error}"
        _write_json(records_dir / "error.json", {
            "stage": stage, "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
        })
        if isinstance(error, UnknownPatientIdsError):
            split_check = json.loads((records_dir / "split_check.json").read_text())
        print(f"Run {run_id} failed during {stage}: {error}")
        interrupted = isinstance(error, KeyboardInterrupt)
    else:
        interrupted = False

    results_summary = _summarize_results(out_dir)
    _write_package_list(records_dir / "packages.txt")
    if failed_stage is None and any(m.get("status") == "ok" for m in results_summary["methods"].values()):
        run_status = "success"
    finished_at = datetime.now(timezone.utc)
    runtime_seconds = (finished_at - started_at).total_seconds()
    resource_use, limits_applied = _resource_use(), _limits_applied()
    gemini = (literature_result.gemini_usage if literature_result else None) or {}
    azure_stages = [stage_tokens[name] for name in ("programming", "benchmarking", "reporting") if name in stage_tokens]
    local_time = lambda moment: moment.astimezone().strftime("%Y-%m-%d %H:%M")  # noqa: E731
    # Same measures as the AISS runs' usage.json where the architectures have them. Not applicable
    # here: reviewer and cache tokens, approvals (an SSA run is unattended, so none are asked for),
    # blocked network attempts (the SSA container's network is not restricted).
    usage = {
        "model": experiment.programming_llm.model,
        "effort": experiment.programming_llm.reasoning_effort,
        "literature_model": experiment.literature_llm.model,
        "literature_effort": experiment.literature_llm.reasoning_effort,
        "start": local_time(started_at), "end": local_time(finished_at), "wall_min": round(runtime_seconds / 60),
        "tok_main_input": sum(t["input_tokens"] for t in azure_stages),
        "tok_main_output": sum(t["output_tokens"] for t in azure_stages),
        "tok_literature_input": gemini.get("input_tokens"),
        "tok_literature_output": gemini.get("output_tokens"),
        "tok_by_stage": stage_tokens,
        "model_strings_reported": {"literature": gemini.get("model_version"), **model_versions},
        "tokens_total": token_usage["total_tokens"] + sum(v for v in (gemini.get("input_tokens"), gemini.get("output_tokens")) if v),
        "approvals_by_user": 0,
        "code_runs": execution_stats.runs,
        "code_runs_errored": execution_stats.errored,
        "code_runs_note": "execute_python calls of the benchmarking stage; the programming stage cannot run code.",
        "network_blocked_attempts": None,
        "web_searches": len(web_searches["queries"]),
        "web_search_queries": web_searches["queries"],
        "web_search_result_urls": web_searches["result_urls"],
        "web_search_result_sites": web_searches.get("result_sites", []),
        "azure_client_retries": retry_counter.summary(),
        "resource_use": resource_use,
        "limits_applied": limits_applied,
        "environments_captured": "packages.txt",
    }
    _write_json(records_dir / "usage.json", usage)
    _write_json(records_dir / "run_manifest.json", {
        "run_id": run_id,
        "system": "SSA",
        "run_status": run_status,
        "failed_stage": failed_stage,
        "stage_failures": stage_failures,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "runtime_seconds": runtime_seconds,
        "stage_runtime_seconds": stage_timings,
        "study_prompt_sha256": study_prompt.sha256,
        "literature_setting": literature_setting,
        "resource_use": resource_use,
        "limits_applied": limits_applied,
        "packages_file": "packages.txt",
        "usage_file": "usage.json",
        "experiment_config": asdict(experiment),
        "task": asdict(task),
        "token_usage": token_usage,
        "token_logging_note": "Provider-reported usage for the programming, benchmarking and reporting stages (Azure); the literature stage's Gemini usage is in usage.json.",
        "results": results_summary,
        "split_check_file": "split_check.json" if split_check is not None else None,
    })
    print(f"Run {run_id}: {run_status}")
    return 130 if interrupted else (0 if run_status == "success" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
