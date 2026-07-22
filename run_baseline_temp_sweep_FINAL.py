"""
Baseline temperature sweep -- parameter compliance below matches the
project's Base Parameters table, with ONE deliberate deviation flagged:

  Model                        gpt-5.4                    -> MODEL
  Temperature                  0.1 / 0.5 / 0.7 / 0.9       -> TEMPERATURES (the manipulated variable)
  Number of Candidate Models   3                           -> NUMBER_OF_MODELS. Models are chosen by
                                                               the LIVE literature-search agent (Gemini
                                                               + Google Search) fresh on every one of
                                                               the 40 runs, NOT fixed to a hardcoded
                                                               list. This means candidate model identity
                                                               can vary run to run, including across
                                                               reps at the same temperature -- a
                                                               deliberate choice to also observe how
                                                               consistently the live literature stage
                                                               converges on similar candidates, at the
                                                               cost of temperature no longer being the
                                                               ONLY variable changing between runs.
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

No "model_names" key is passed in BENCHMARK_CONDITION_JSON, so main.py's
literature stage takes its normal live-search path (build_literature_agent
+ run_literature_review against Vertex AI/Gemini with Google Search
grounding) -- no patch is required for this behavior; it is main.py's
default when model_names is absent.

Analysis note for later: since candidate models may differ run to run,
downstream analysis should key results by the actual model_name found in
each run's report.json rather than assuming a fixed 3-model set, and
should treat "did the same models get selected repeatedly" as its own
finding worth reporting alongside the temperature-effect results.
"""

import os
import json
import subprocess
import time

TEMPERATURES = [0.1, 0.5, 0.7, 0.9]
N_REPS = 10

MODEL = "gpt-5.4"
NUMBER_OF_MODELS = 3
OUTCOME = "new_hyperlipidemia"

# Base LITERATURE_SYSTEM_PROMPT text (from src/settings/prompts.py), with one
# added constraint: exclude deep learning / neural network candidates so
# runs stay fast. This IS a deliberate deviation from "prompt wording
# unmodified" -- flagged explicitly here and in the parameter banner below,
# since every other stage's prompt (programming/benchmarking/reporting)
# remains genuinely unmodified.
LITERATURE_PROMPT_OVERRIDE = """
You are an expert scientist in the field of biostatistics who is working on a research project.
Your research group is tasked with benchmarking the performance of various new machine learning
models to predict patient outcomes based on clinical data, specifically data in the format of
EHRSHOT. Your task is to provide a list of candidate models to benchmark, along with a summary
of the documentation for each model.

Restrict candidates to lightweight, classical/traditional machine learning models that train
quickly on tabular data (for example: linear models, tree-based ensembles, simple kernel
methods). Do NOT propose deep learning, neural network, or transformer-based models -- these
are out of scope for this benchmark given time constraints on each run.
"""

# Applies to literature/programming/reporting stages.
PIPELINE_STAGE_TIMEOUT_SECONDS = 900
# Benchmarking now does substantially more work (cohort construction,
# feature caching, predictions.json/.csv validation, a written critique),
# so it gets its own longer, explicit timeout rather than relying on
# main.py's implicit 900s default.
BENCHMARK_STAGE_TIMEOUT_SECONDS = 1500

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
    print(f"  Candidate models:           {NUMBER_OF_MODELS} (live search, may vary per run)")
    print(f"  Search results per query:   not applicable (unset)")
    print(f"  Dataset:                    EHRSHOT")
    print(f"  Prompt wording/tone:        UNMODIFIED for programming/benchmarking/reporting;")
    print(f"                               literature prompt MODIFIED (lightweight models only,")
    print(f"                               deep learning excluded -- see LITERATURE_PROMPT_OVERRIDE)")
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
        "number_of_models": NUMBER_OF_MODELS,
        "literature_prompt": LITERATURE_PROMPT_OVERRIDE,
        # No "model_names" key: main.py takes its normal live literature-search
        # path (Gemini + Google Search grounding), fresh on every run --
        # constrained to lightweight/non-deep-learning candidates above.
        # No "programming_prompt" / "benchmarking_prompt" / "reporting_prompt" keys:
        # main.py falls back to its own unmodified default prompts for those stages.
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
            time.sleep(60)


if __name__ == "__main__":
    main()
