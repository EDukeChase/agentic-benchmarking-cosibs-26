import uuid
import os
import signal
import threading
import traceback
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from contextlib import contextmanager
from src.agents.literature_agent import build_literature_agent, run_literature_review
from src.agents.programming_agent import build_programming_agent, run_programming_agent, collect_generated_models
from src.agents.benchmarking_agent import (
    BENCHMARKING_SYSTEM_PROMPT,
    build_benchmarking_agent,
    run_benchmarking_agent,
)
from src.evaluation.benchmark_tools import collect_benchmark_results, collect_benchmark_scripts
from src.agents.reporting_agent import build_reporting_agent, build_report
# Local rollback option; the live pipeline uses Vertex AI literature discovery.
# from src.fixtures.base_literature import load_base_literature
from src.reporting.markdown_report import save_error_markdown, save_markdown
from src.core.schemas import BenchmarkResult
from src.settings.config import (
    LITERATURE_MODEL,
    BenchmarkTaskConfig,
    ExperimentConfig,
    LLMConfig,
    SelfConsistencyConfig,
)
# from src.evaluation.deterministic import evaluate_run
from src.utils.telemetry import collect_token_usage
from src.settings.prompts import (
    LITERATURE_SYSTEM_PROMPT,
    PROGRAMMING_SYSTEM_PROMPT,
    REPORTING_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# BENCHMARK SETTINGS: edit this block to run a different experiment.
# Prompt text lives in src/settings/prompts.py; alternatively replace a prompt below.
# self_consistency.samples=1 preserves the original single-report behavior.
# Values greater than 1 generate that many reports and combine them with the
# configured self-consistency judge model.
# ---------------------------------------------------------------------------

# Edit the parameters here
MODEL = "gpt-5.4"
TEMPERATURE = 1.0
NUMBER_OF_MODELS = 5
MAX_SEARCH_RESULTS = 1


EXPERIMENT = ExperimentConfig(
    number_of_models=NUMBER_OF_MODELS,
    max_search_results=MAX_SEARCH_RESULTS,
    literature_llm=LLMConfig(model=LITERATURE_MODEL, temperature=0.0),
    programming_llm=LLMConfig(model=MODEL, temperature=TEMPERATURE),
    benchmarking_llm=LLMConfig(model=MODEL, temperature=TEMPERATURE),
    reporting_llm=LLMConfig(model=MODEL, temperature=TEMPERATURE),
    self_consistency=SelfConsistencyConfig(
        samples=1,
        model=MODEL,
        temperature=TEMPERATURE,
    ),
)

PROGRAMMING_PROMPT = PROGRAMMING_SYSTEM_PROMPT
BENCHMARKING_PROMPT = BENCHMARKING_SYSTEM_PROMPT
REPORTING_PROMPT = REPORTING_SYSTEM_PROMPT
LITERATURE_PROMPT = LITERATURE_SYSTEM_PROMPT

literature_uncertainty = None
programming_uncertainty = None
benchmarking_uncertainty = None
reporting_uncertainty = None

def _configuration_from_environment():
    raw = json.loads(os.getenv("BENCHMARK_CONDITION_JSON", "{}"))
    model = raw.get("model", EXPERIMENT.programming_llm.model)
    temperature = float(raw.get("temperature", EXPERIMENT.programming_llm.temperature))
    llm = LLMConfig(model=model, temperature=temperature)
    literature_llm = LLMConfig(
        model=raw.get("literature_model", EXPERIMENT.literature_llm.model),
        temperature=float(
            raw.get("literature_temperature", EXPERIMENT.literature_llm.temperature)
        ),
        timeout=EXPERIMENT.literature_llm.timeout,
        max_retries=EXPERIMENT.literature_llm.max_retries,
    )
    experiment = ExperimentConfig(
        number_of_models=int(raw.get("number_of_models", EXPERIMENT.number_of_models)),
        max_search_results=int(raw.get("max_search_results", EXPERIMENT.max_search_results)),
        literature_llm=literature_llm,
        programming_llm=llm,
        benchmarking_llm=llm,
        reporting_llm=llm,
        self_consistency=SelfConsistencyConfig(samples=1, model=model, temperature=temperature),
    )
    task = BenchmarkTaskConfig(**json.loads(os.getenv("BENCHMARK_TASK_JSON", "{}")))
    return experiment, task, raw

class StageTimeoutError(TimeoutError):
    """Raised when a pipeline stage exceeds its configured wall-clock limit (takes too long)."""


@contextmanager
def stage_timeout(stage: str, seconds: int):
    """Bound a stage on Linux (the project's dev-container runtime).

    SIGALRM can interrupt blocking Python/network work in the container. On hosts
    without SIGALRM, request/subprocess timeouts still apply, but this outer bound
    cannot be enforced safely from a thread.
    """

    # If user passes 0 or negative seconds, or os does not support 'SIGALRM', or if not in the main thread,
    # yield (run code normally) and don't set a timer.
    if seconds <= 0 or not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    # When the timer expires, raise a StageTimeoutError (stop program) with the stage name and timeout duration.
    def _raise_timeout(_signum, _frame):
        raise StageTimeoutError(f"Stage '{stage}' exceeded its {seconds}-second time limit")

    # Set the SIGALRM handler to our timeout function, and start the timer.
    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        # pause execution of the code block until the timer expires or the block completes
        yield
    finally:
        # Cancel the timer and restore the previous SIGALRM handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def main():
    experiment, benchmark_task, condition = _configuration_from_environment()
    # generate a unique run ID for this benchmarking session
    run_id = os.getenv("BENCHMARK_RUN_ID", uuid.uuid4().hex[:8])
    run_dir = f"/app/generated_code/{run_id}"
    markdown_report_path = f"{run_dir}/report.md"
    os.makedirs(run_dir, exist_ok=True)

    # set the timeout for each stage of the pipeline (default: 5 minutes)
    timeout_seconds = int(os.getenv("PIPELINE_STAGE_TIMEOUT_SECONDS", "300"))
    benchmark_timeout_seconds = int(
        os.getenv("BENCHMARK_STAGE_TIMEOUT_SECONDS", "900")
    )
    stage = "initialization"

    number_of_models = experiment.number_of_models
    max_search_results = experiment.max_search_results
    started_at = datetime.now(timezone.utc)
    stage_timings = {}
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    print(f"Starting new run with ID: {run_id}")
    
    try:
        stage = "literature review"
        print(
            f"Searching with {experiment.literature_llm.model} for "
            f"{number_of_models} candidate models..."
        )
        stage_started = time.perf_counter()
        literature_agent = build_literature_agent(
            llm_config=experiment.literature_llm,
        )
        with stage_timeout(stage, timeout_seconds):
            literature_result = run_literature_review(
                literature_agent,
                num_models=number_of_models,
                system_prompt=LITERATURE_PROMPT,
            )
        stage_timings[stage] = time.perf_counter() - stage_started

        # Local rollback option when Vertex AI or web grounding is unavailable:
        # literature_result = load_base_literature(num_models=number_of_models)

        # Run when uncertainty quantification is finalized:
        # literature_output = run_literature_review_with_uncertainty(
        #     literature_agent, num_models=number_of_models
        # )
        # literature_result = literature_output["result"]
        # literature_uncertainty = literature_output["uncertainty"]

        stage = "code generation"
        print(f"Running programming agent to generate code for the models...")
        prog_root = run_dir
        prog_agent = build_programming_agent(
            prog_root,
            max_search_results=max_search_results,
            llm_config=experiment.programming_llm,
        )

        # this stage may take a long time, so we use the stage_timeout context manager to enforce a timeout
        stage_started = time.perf_counter()
        with stage_timeout(stage, timeout_seconds):
            programming_response = run_programming_agent(
                prog_agent,
                literature_result,
                system_prompt_template=condition.get("programming_prompt", PROGRAMMING_PROMPT),
            )
        stage_timings[stage] = time.perf_counter() - stage_started
        usage = collect_token_usage(programming_response)
        for key in token_usage:
            token_usage[key] += usage[key]
        model_code = collect_generated_models(prog_root)
        if len(model_code) != number_of_models:
            raise RuntimeError(
                "Programming agent generated "
                f"{len(model_code)} usable model(s); expected {number_of_models}. "
                "Each model folder must contain model.py."
            )
        
        # SHOULD IMPLEMENT run_programming_agent_with_uncertainty() here from programming_agent.py - uncomment when ready
        # programming_output = run_programming_agent_with_uncertainty(
        #     prog_agent,
        #     literature_result,
        #     n_runs= 5,
        # )

        # programming_uncertainty = programming_output["uncertainty"]


        # Because benchmarking is deterministic, it should not get uncertainty
        stage = "benchmarking"
        print(f"Running LLM benchmarking agent for run {run_id}...")
        stage_started = time.perf_counter()
        benchmarking_agent = build_benchmarking_agent(
            llm_config=experiment.benchmarking_llm,
        )
        with stage_timeout(stage, benchmark_timeout_seconds):
            benchmarking_response = run_benchmarking_agent(
                benchmarking_agent,
                run_id,
                literature_result,
                system_prompt_template=condition.get(
                    "benchmarking_prompt", BENCHMARKING_PROMPT
                ),
                benchmark_task=benchmark_task,
            )
        usage = collect_token_usage(benchmarking_response)
        for key in token_usage:
            token_usage[key] += usage[key]

        # Deterministic route retained for easy rollback/reference:
        # evaluate_run(run_id, benchmark_task)
        stage_timings[stage] = time.perf_counter() - stage_started

        stage = "artifact collection"
        print(f"Collecting benchmark results and scripts for run {run_id}...")
        raw_results = collect_benchmark_results(run_id)
        results = [
            BenchmarkResult(model_name=name, **metrics)
            for name, metrics in raw_results.items()
        ]
        benchmark_scripts = collect_benchmark_scripts(run_id)

        print(f"Benchmark results and scripts collected.")

        stage = "report generation"
        print(f"Building benchmark report for run {run_id}...")

        reporting_llm = build_reporting_agent(experiment.reporting_llm)
        stage_started = time.perf_counter()
        with stage_timeout(stage, timeout_seconds):
            report = build_report(
                reporting_llm,
                literature_result.candidates,
                model_code,
                results,
                benchmark_scripts,
                self_consistency=experiment.self_consistency,
                system_prompt=condition.get("reporting_prompt", REPORTING_PROMPT),
                usage_sink=token_usage,
                benchmark_assessment=str(benchmarking_response["messages"][-1].content),
            )
        
        reporting_uncertainty = report.uncertainty

        stage_timings[stage] = time.perf_counter() - stage_started

        report_path = f"{run_dir}/report.json"
        with open(report_path, "w") as f:
            f.write(report.model_dump_json(indent=2))

        save_markdown(report, markdown_report_path)

        finished_at = datetime.now(timezone.utc)
        run_manifest = {
            "run_id": run_id,
            "experiment_id": os.getenv("BENCHMARK_EXPERIMENT_ID", "single-run"),
            "condition_id": os.getenv("BENCHMARK_CONDITION_ID", "baseline"),
            "replicate": int(os.getenv("BENCHMARK_REPLICATE", "1")),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "runtime_seconds": (finished_at - started_at).total_seconds(),
            "condition": condition,
            "experiment_config": asdict(experiment),
            "benchmark_task": asdict(benchmark_task),
            "prompts": {
                "programming": condition.get("programming_prompt", PROGRAMMING_PROMPT),
                "reporting": condition.get("reporting_prompt", REPORTING_PROMPT),
                "benchmarking": condition.get(
                    "benchmarking_prompt", BENCHMARKING_PROMPT
                ),
            },
            "token_usage": token_usage,
            "token_logging_note": "Provider-reported usage for programming and reporting calls when exposed by LangChain; unavailable calls remain zero.",
            "stage_runtime_seconds": stage_timings,
            "uncertainty": {
                "literature": literature_uncertainty,
                "programming": programming_uncertainty,
                "benchmarking": benchmarking_uncertainty,
                "reporting": reporting_uncertainty,
            }
        }
        with open(f"{run_dir}/run_manifest.json", "w") as file:
            json.dump(run_manifest, file, indent=2)

        print(f"Report written to {report_path} and {markdown_report_path}")
        print(f"Run {run_id} completed. Benchmark results and scripts collected.")
        return 0
    # Handle exceptions and save error information to a markdown report
    except (Exception, KeyboardInterrupt) as error:
        # Save the error and traceback to a markdown report for debugging
        traceback_text = traceback.format_exc()
        save_error_markdown(
            markdown_report_path,
            run_id=run_id,
            stage=stage,
            error=error,
            traceback_text=traceback_text,
        )
        print(f"Run {run_id} failed during {stage}: {error}")
        print(f"Error report written to {markdown_report_path}")
        return 130 if isinstance(error, KeyboardInterrupt) else 1

if __name__ == "__main__":
    # Run the main function and exit with its return code
    raise SystemExit(main())
