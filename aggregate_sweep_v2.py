"""
Aggregates the 10 repetition results per (temperature, outcome) combination
into averaged summary JSONs.

Reads:  /app/generated_code/temp<T>_<outcome>_rep<NN>/report.json
        (a BenchmarkReport: {"entries": [{model_name, accuracy, f1, auroc,
         precision, recall, brier, ...}, ...], "summary": ..., ...})

Writes: /app/sweep_results/results_temp_<T>_<outcome>_averaged.json
        one entry per model_name, with mean/stdev/n across however many
        reps actually succeeded for that combination.

Runs that failed (no report.json -- main.py writes an error report.md
instead when a stage fails) are simply skipped and don't count toward n.
"""

import os
import re
import json
import glob
import statistics

GENERATED_CODE_DIR = "/app/generated_code"
OUTPUT_DIR = "/app/sweep_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMPERATURES = [0.1, 0.5, 0.7, 0.9]
OUTCOMES = ["new_pancan", "new_lupus"]
N_REPS = 10

METRICS_TO_AVERAGE = ["accuracy", "f1", "precision", "recall", "auroc", "brier"]


def run_id_for(temperature: float, outcome: str, rep: int) -> str:
    return f"temp{temperature}_{outcome}_rep{rep:02d}"


def load_report(run_id: str) -> dict | None:
    path = os.path.join(GENERATED_CODE_DIR, run_id, "report.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"SKIPPING unreadable report for {run_id}: {e}")
        return None


def average_one_combination(temperature: float, outcome: str) -> dict:
    # collected[model_name][metric] = list of values across successful reps
    collected = {}
    n_succeeded = 0
    n_failed = 0

    for rep in range(1, N_REPS + 1):
        run_id = run_id_for(temperature, outcome, rep)
        report = load_report(run_id)
        if report is None:
            n_failed += 1
            continue
        n_succeeded += 1

        for entry in report.get("entries", []):
            model_name = entry.get("model_name")
            if model_name is None:
                continue
            if model_name not in collected:
                collected[model_name] = {m: [] for m in METRICS_TO_AVERAGE}

            for m in METRICS_TO_AVERAGE:
                val = entry.get(m)
                if val is not None:
                    collected[model_name][m].append(val)

    print(f"temp={temperature} outcome={outcome}: {n_succeeded} succeeded, {n_failed} failed out of {N_REPS} reps")

    averaged = {}
    for model_name, metric_lists in collected.items():
        entry = {"n_reps_succeeded": n_succeeded, "n_reps_failed": n_failed}
        for m in METRICS_TO_AVERAGE:
            values = metric_lists[m]
            if values:
                entry[f"{m}_mean"] = statistics.mean(values)
                entry[f"{m}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
                entry[f"{m}_n"] = len(values)
            else:
                entry[f"{m}_mean"] = None
                entry[f"{m}_stdev"] = None
                entry[f"{m}_n"] = 0
        averaged[model_name] = entry

    return averaged


def main():
    for temperature in TEMPERATURES:
        for outcome in OUTCOMES:
            averaged = average_one_combination(temperature, outcome)
            if not averaged:
                print(f"  -> no successful reps for temp={temperature} outcome={outcome}, skipping output file")
                continue
            out_path = os.path.join(
                OUTPUT_DIR, f"results_temp_{temperature}_{outcome}_averaged.json"
            )
            with open(out_path, "w") as f:
                json.dump(averaged, f, indent=2)
            print(f"  -> wrote {out_path} ({len(averaged)} models)")


if __name__ == "__main__":
    main()
