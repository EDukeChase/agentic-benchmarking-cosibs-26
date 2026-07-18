import uuid
import os
import signal
import threading
import traceback
from contextlib import contextmanager
# from src.agents.literature_agent import build_literature_agent, run_literature_review
from src.agents.programming_agent import build_programming_agent, run_programming_agent, collect_generated_models
from src.agents.benchmarking_agent import build_benchmarking_agent, run_benchmarking_agent
from src.benchmark_tools import collect_benchmark_results, collect_benchmark_scripts
from src.agents.reporting_agent import build_reporting_agent, build_report
from src.base_literature import load_base_literature
from src.markdown_report import save_error_markdown, save_markdown
from src.schemas import BenchmarkResult

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
    # generate a unique run ID for this benchmarking session
    run_id = uuid.uuid4().hex[:8]
    run_dir = f"/app/generated_code/{run_id}"
    markdown_report_path = f"{run_dir}/report.md"

    # set the timeout for each stage of the pipeline (default: 5 minutes)
    timeout_seconds = int(os.getenv("PIPELINE_STAGE_TIMEOUT_SECONDS", "120"))
    stage = "initialization"

    # define the number of models to benchmark and the maximum number of search results to consider
    number_of_models = 5
    max_search_results = 1

    print(f"Starting new run with ID: {run_id}")
    
    try:
        stage = "literature review"
        print(f"Loading {number_of_models} models from the local base literature file...")
        # Restore these lines when Tavily use is permitted again:
        # lit_agent = build_literature_agent(max_search_results=max_search_results)
        # with stage_timeout(stage, timeout_seconds):
        #     literature_result = run_literature_review(lit_agent, num_models=number_of_models)
        literature_result = load_base_literature(num_models=number_of_models)

        stage = "code generation"
        print(f"Running programming agent to generate code for the models...")
        prog_root = run_dir
        prog_agent = build_programming_agent(prog_root)
        # this stage may take a long time, so we use the stage_timeout context manager to enforce a timeout
        with stage_timeout(stage, timeout_seconds):
            run_programming_agent(prog_agent, literature_result)
        model_code = collect_generated_models(prog_root)

        stage = "benchmarking"
        print(f"Initialized benchmarking agent for run {run_id}...")
        bench_agent = build_benchmarking_agent()

        print(f"Running benchmarking agent to evaluate the generated models...")
        with stage_timeout(stage, timeout_seconds):
            run_benchmarking_agent(bench_agent, run_id, literature_result)

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

        reporting_llm = build_reporting_agent()
        with stage_timeout(stage, timeout_seconds):
            report = build_report(
                reporting_llm,
                literature_result.candidates,
                model_code,
                results,
                benchmark_scripts,
            )

        report_path = f"{run_dir}/report.json"
        with open(report_path, "w") as f:
            f.write(report.model_dump_json(indent=2))

        save_markdown(report, markdown_report_path)

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
