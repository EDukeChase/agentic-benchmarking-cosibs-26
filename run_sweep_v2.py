"""
Runs the benchmarking pipeline (main.py) as a subprocess, once per
combination of temperature x outcome x repetition, using the
BENCHMARK_CONDITION_JSON / BENCHMARK_TASK_JSON / BENCHMARK_RUN_ID
environment variables main.py already reads.

Each run gets its own output folder at
    /app/generated_code/<run_id>/report.json
via main.py's own run_id-based isolation, so no manual clearing of
/generated_code between runs is needed (unlike the earlier agent-based
pipeline).

Resumable: skips any run whose report.json already exists, so a crashed
or interrupted sweep can just be restarted with the same command.
"""

import os
import json
import subprocess
import time

TEMPERATURES = [0.1, 0.5, 0.7, 0.9]
OUTCOMES = ["new_pancan", "new_lupus"]
N_REPS = 10
NUMBER_OF_MODELS = 5          # first 5 candidates in base_literature.py
MODEL = "gpt-5.4-mini"
MAX_SEARCH_RESULTS = 1
STAGE_TIMEOUT_SECONDS = 300   # bump if runs are timing out; 0 disables the bound

GENERATED_CODE_DIR = "/app/generated_code"
SWEEP_LOG_DIR = "/app/sweep_logs"
os.makedirs(SWEEP_LOG_DIR, exist_ok=True)


def run_id_for(temperature: float, outcome: str, rep: int) -> str:
    return f"temp{temperature}_{outcome}_rep{rep:02d}"


def report_exists(run_id: str) -> bool:
    return os.path.exists(os.path.join(GENERATED_CODE_DIR, run_id, "report.json"))


def run_one(temperature: float, outcome: str, rep: int):
    run_id = run_id_for(temperature, outcome, rep)

    if report_exists(run_id):
        print(f"SKIP (already succeeded): {run_id}")
        return

    print(f"=== RUNNING: {run_id} ===")

    condition = {
        "model": MODEL,
        "temperature": temperature,
        "number_of_models": NUMBER_OF_MODELS,
        "max_search_results": MAX_SEARCH_RESULTS,
    }
    task = {
        "outcome": outcome,
    }

    env = os.environ.copy()
    env["BENCHMARK_CONDITION_JSON"] = json.dumps(condition)
    env["BENCHMARK_TASK_JSON"] = json.dumps(task)
    env["BENCHMARK_RUN_ID"] = run_id
    env["BENCHMARK_EXPERIMENT_ID"] = "temperature-sweep"
    env["BENCHMARK_CONDITION_ID"] = f"temp{temperature}_{outcome}"
    env["BENCHMARK_REPLICATE"] = str(rep)
    env["PIPELINE_STAGE_TIMEOUT_SECONDS"] = str(STAGE_TIMEOUT_SECONDS)

    log_path = os.path.join(SWEEP_LOG_DIR, f"{run_id}.log")
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            ["python3", "main.py"],
            cwd="/app",
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    if result.returncode == 0 and report_exists(run_id):
        print(f"SUCCESS: {run_id}")
    else:
        print(f"FAILED (exit code {result.returncode}): {run_id} -- see {log_path} "
              f"and /app/generated_code/{run_id}/report.md for details")


def main():
    total = len(TEMPERATURES) * len(OUTCOMES) * N_REPS
    done = 0
    for temperature in TEMPERATURES:
        for outcome in OUTCOMES:
            for rep in range(1, N_REPS + 1):
                run_one(temperature, outcome, rep)
                done += 1
                print(f"--- progress: {done}/{total} ---")
                time.sleep(3)  # small buffer between runs


if __name__ == "__main__":
    main()
