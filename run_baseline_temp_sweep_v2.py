"""
Baseline temperature sweep -- parameter compliance below matches the
project's Base Parameters table exactly, row by row:

  Model                        gpt-5.4                    -> MODEL
  Temperature                  0.1 / 0.5 / 0.7 / 0.9       -> TEMPERATURES (the manipulated variable)
  Number of Candidate Models   3 (GBM, LogReg, RF)         -> MODEL_NAMES, selected by name via the
                                                               model_names patch (bypasses live search
                                                               so the same 3 models are used every run)
  Search Results per Query     not applicable              -> left unset; not passed in condition
  Dataset                      EHRSHOT                     -> BenchmarkTaskConfig default (only
                                                               supported dataset; not overridden)
  Prompt Wording/Tone          as written in each agent's  -> no *_prompt keys are set in condition,
                                SystemMessage/HumanMessage     so main.py falls back to its own
                                                               unmodified PROGRAMMING_PROMPT /
                                                               BENCHMARKING_PROMPT / REPORTING_PROMPT
  Nucleus Sampling (top_p)     not applicable               -> no top_p field exists in LLMConfig;
                                                               nothing to set
  Repetitions                  10, fixed regardless of      -> N_REPS = 10, applied identically at
                                temperature                    every temperature value
  Classification Task          Hyperlipidemia only          -> OUTCOME = "new_hyperlipidemia",
                                                               single task, no outcome loop

Total runs: 4 temperatures x 10 reps = 40.

Requires main.py to already be patched (see patch_main_v3.py) so that
"model_names" in BENCHMARK_CONDITION_JSON bypasses the live literature
search and deterministically selects exactly these 3 named models on
every run -- without this, a live search agent could not be guaranteed
to return the same 3 models across all 40 runs, which would introduce a
second, uncontrolled variable into what is meant to be a temperature-only
experiment.
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

# Applies to literature/programming/reporting stages.
PIPELINE_STAGE_TIMEOUT_SECONDS = 300
# Benchmarking now does substantially more work (cohort construction,
# feature caching, predictions.json/.csv validation, a written critique),
# so it gets its own longer, explicit timeout rather than relying on
# main.py's implicit 900s default.
BENCHMARK_STAGE_TIMEOUT_SECONDS = 900

GENERATED_CODE_DIR = "/app/generated_code"
LOG_DIR = "/app/baseline_temp_sweep_logs"
os.makedirs(LOG_DIR, exist_ok=True)


def run_id_for(temperature: float, rep: int) -> str:
    return f"temp{temperature}_hyperlipidemia_rep{rep:02d}"


def report_exists(run_id: str) -> bool:
    return os.path.exists(os.path.join(GENERATED_CODE_DIR, run_id, "report.json"))


def print_parameter_banner():
    print("=" * 70)
    print("BASELINE TEMPERATURE SWEEP -- confirmed parameters")
    print("=" * 70)
    print(f"  Model:                      {MODEL}")
    print(f"  Temperature (manipulated):  {TEMPERATURES}")
    print(f"  Candidate models (fixed):   {MODEL_NAMES}")
    print(f"  Search results per query:   not applicable (unset)")
    print(f"  Dataset:                    EHRSHOT")
    print(f"  Prompt wording/tone:        unmodified (no *_prompt override passed)")
    print(f"  Nucleus sampling (top_p):   not applicable (not implemented)")
    print(f"  Repetitions per temperature: {N_REPS} (fixed)")
    print(f"  Classification task:        {OUTCOME}")
    print(f"  Total runs:                 {len(TEMPERATURES) * N_REPS}")
    print("=" * 70)


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
        # No "programming_prompt" / "benchmarking_prompt" / "reporting_prompt" keys:
        # main.py falls back to its own unmodified default prompts.
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
    env["PIPELINE_STAGE_TIMEOUT_SECONDS"] = str(PIPELINE_STAGE_TIMEOUT_SECONDS)
    env["BENCHMARK_STAGE_TIMEOUT_SECONDS"] = str(BENCHMARK_STAGE_TIMEOUT_SECONDS)

    log_path = os.path.join(LOG_DIR, f"{run_id}.log")
    with open(log_path, "w") as log_file:
        # Write the exact condition/task JSON at the top of this run's log,
        # so every run's parameters are individually auditable after the fact.
        log_file.write(f"BENCHMARK_CONDITION_JSON={json.dumps(condition)}\n")
        log_file.write(f"BENCHMARK_TASK_JSON={json.dumps(task)}\n")
        log_file.write("-" * 70 + "\n")
        log_file.flush()

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
    print_parameter_banner()
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
