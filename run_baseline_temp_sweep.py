"""
Temperature sweep, 10 reps per temperature, using the baseline parameters:
  Model:              gpt-5.4
  Candidate models:   Gradient Boosting, Logistic Regression, Random Forest
  Dataset:            EHRSHOT (only option the evaluator supports)
  Classification task: new_hyperlipidemia
  Prompts:            unchanged -- exactly as written in each agent's
                       SystemMessage/HumanMessage (no override passed)

Temperature is the only variable manipulated across conditions:
  0.1, 0.5, 0.7, 0.9  x  10 reps each  =  40 total runs

Requires main.py to already be patched for model_names support
(see patch_main_for_model_names.py) -- run that once, first.

Search-results-per-query and top_p are not applicable / not implemented
in the current LLMConfig, so they are not varied or passed here.
"""

import os
import json
import subprocess
import time

TEMPERATURES = [0.1, 0.5, 0.7, 0.9]
N_REPS = 10

MODEL = "gpt-5.4"
MODEL_NAMES = ["Gradient Boosting", "Logistic Regression", "Random Forest"]
OUTCOME = "new_hyperlipidemia"
STAGE_TIMEOUT_SECONDS = 300

GENERATED_CODE_DIR = "/app/generated_code"
LOG_DIR = "/app/baseline_temp_sweep_logs"
os.makedirs(LOG_DIR, exist_ok=True)


def run_id_for(temperature: float, rep: int) -> str:
    return f"temp{temperature}_hyperlipidemia_rep{rep:02d}"


def report_exists(run_id: str) -> bool:
    return os.path.exists(os.path.join(GENERATED_CODE_DIR, run_id, "report.json"))


def run_one(temperature: float, rep: int):
    run_id = run_id_for(temperature, rep)

    if report_exists(run_id):
        print(f"SKIP (already succeeded): {run_id}")
        return

    print(f"=== RUNNING: {run_id} ===")

    condition = {
        "model": MODEL,
        "temperature": temperature,
        "model_names": MODEL_NAMES,
        "number_of_models": len(MODEL_NAMES),  # kept for logging/back-compat; model_names takes precedence
    }
    task = {
        "outcome": OUTCOME,
    }

    env = os.environ.copy()
    env["BENCHMARK_CONDITION_JSON"] = json.dumps(condition)
    env["BENCHMARK_TASK_JSON"] = json.dumps(task)
    env["BENCHMARK_RUN_ID"] = run_id
    env["BENCHMARK_EXPERIMENT_ID"] = "baseline-temperature-sweep"
    env["BENCHMARK_CONDITION_ID"] = f"temp{temperature}_hyperlipidemia"
    env["BENCHMARK_REPLICATE"] = str(rep)
    env["PIPELINE_STAGE_TIMEOUT_SECONDS"] = str(STAGE_TIMEOUT_SECONDS)

    log_path = os.path.join(LOG_DIR, f"{run_id}.log")
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
        print(f"FAILED (exit code {result.returncode}): {run_id} -- see {log_path}")


def main():
    total = len(TEMPERATURES) * N_REPS
    done = 0
    for temperature in TEMPERATURES:
        for rep in range(1, N_REPS + 1):
            run_one(temperature, rep)
            done += 1
            print(f"--- progress: {done}/{total} ---")
            time.sleep(3)


if __name__ == "__main__":
    main()
