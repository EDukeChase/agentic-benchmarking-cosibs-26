"""
Aggregates the 10 repetition results per temperature into one summary JSON
per temperature -- updated for LIVE literature search, where the exact
candidate models (and their exact names) can differ run to run.

Because names vary (e.g. "random_forest" vs "random_forest_classifier" vs
"Random Forest"), results are grouped by a NORMALIZED model identity
(substring-matched against a list of known classical ML model families)
rather than by exact string match. Any run's raw model_name that doesn't
match a known family falls back to its own cleaned name, so nothing is
silently dropped or wrongly merged.

Also produces a model-selection audit file per temperature, showing:
  - which raw model_name strings were actually returned by the live search,
    grouped under each normalized category
  - how many of the 10 reps each category appeared in (a direct measure of
    how consistently the literature-search stage converges on similar
    candidates at that temperature)
  - any model whose name suggests deep learning / neural networks slipped
    through despite the "lightweight only" prompt constraint

Reads:  /app/generated_code/temp<T>_hyperlipidemia_rep<NN>/report.json
Writes: /app/baseline_temp_sweep_results/results_temp_<T>_hyperlipidemia_averaged.json
        /app/baseline_temp_sweep_results/model_selection_audit_temp_<T>.json
"""

import os
import re
import json
import statistics

GENERATED_CODE_DIR = "/app/generated_code"
OUTPUT_DIR = "/app/baseline_temp_sweep_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMPERATURES = [0.1, 0.5, 0.7, 0.9]
N_REPS = 10
OUTCOME = "new_hyperlipidemia"

METRICS_TO_AVERAGE = ["accuracy", "f1", "precision", "recall", "auroc", "brier"]

# Substring -> canonical category. Order matters: more specific patterns
# (e.g. "histgradientboosting") are listed before shorter ones that could
# also match a substring of them.
MODEL_FAMILY_PATTERNS = [
    ("logisticregression", "logistic_regression"),
    ("linearregression", "linear_regression"),
    ("randomforest", "random_forest"),
    ("extratrees", "extra_trees"),
    ("extremerandomtrees", "extra_trees"),
    ("xgboost", "xgboost"),
    ("histgradientboosting", "gradient_boosting"),
    ("gradientboosting", "gradient_boosting"),
    ("lightgbm", "lightgbm"),
    ("catboost", "catboost"),
    ("adaboost", "adaboost"),
    ("decisiontree", "decision_tree"),
    ("supportvectormachine", "svm"),
    ("svc", "svm"),
    ("svm", "svm"),
    ("kneighbors", "knn"),
    ("knearestneighbor", "knn"),
    ("knn", "knn"),
    ("naivebayes", "naive_bayes"),
    ("ridge", "ridge_regression"),
    ("lasso", "lasso_regression"),
    ("elasticnet", "elastic_net"),
]

# Substrings that would indicate the "lightweight, no deep learning"
# literature-prompt constraint was NOT respected -- flagged for audit,
# not silently dropped.
DEEP_LEARNING_FLAGS = [
    "neural", "deep", "mlp", "transformer", "cnn", "lstm", "gru",
    "attention", "perceptron", "bert", "gpt", "autoencoder",
]


def normalize_model_name(name: str) -> str:
    """Map a raw model_name string to a canonical family identifier.
    Falls back to a cleaned version of the raw name if no known family
    substring matches, so nothing is silently lost.
    """
    blob = re.sub(r"[^a-z0-9]", "", name.lower())
    for pattern, canonical in MODEL_FAMILY_PATTERNS:
        if pattern in blob:
            return canonical
    return blob or "unknown"


def is_flagged_deep_learning(name: str) -> bool:
    blob = name.lower()
    return any(flag in blob for flag in DEEP_LEARNING_FLAGS)


def run_id_for(temperature: float, rep: int) -> str:
    return f"temp{temperature}_hyperlipidemia_rep{rep:02d}"


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


def process_one_temperature(temperature: float):
    # metrics[canonical_category][metric] = list of values across reps
    metrics_by_category = {}
    # raw_names_seen[canonical_category] = {raw_name: count_of_reps_using_it}
    raw_names_seen = {}
    # reps_per_category[canonical_category] = set of rep numbers where this
    # category appeared at all (regardless of exact raw name)
    reps_per_category = {}
    deep_learning_hits = []  # list of (rep, raw_name) that got flagged

    n_succeeded = 0
    n_failed = 0

    for rep in range(1, N_REPS + 1):
        run_id = run_id_for(temperature, rep)
        report = load_report(run_id)
        if report is None:
            n_failed += 1
            continue
        n_succeeded += 1

        for entry in report.get("entries", []):
            raw_name = entry.get("model_name")
            if raw_name is None:
                continue

            if is_flagged_deep_learning(raw_name):
                deep_learning_hits.append({"rep": rep, "model_name": raw_name})

            category = normalize_model_name(raw_name)

            if category not in metrics_by_category:
                metrics_by_category[category] = {m: [] for m in METRICS_TO_AVERAGE}
                raw_names_seen[category] = {}
                reps_per_category[category] = set()

            raw_names_seen[category][raw_name] = raw_names_seen[category].get(raw_name, 0) + 1
            reps_per_category[category].add(rep)

            for m in METRICS_TO_AVERAGE:
                val = entry.get(m)
                if val is not None:
                    metrics_by_category[category][m].append(val)

    print(f"temperature={temperature}: {n_succeeded} succeeded, {n_failed} failed out of {N_REPS} reps")
    print(f"  categories found: {list(metrics_by_category.keys())}")
    if deep_learning_hits:
        print(f"  WARNING: {len(deep_learning_hits)} model(s) flagged as possible deep learning: {deep_learning_hits}")

    # Build averaged results, keyed by canonical category
    averaged = {}
    for category, metric_lists in metrics_by_category.items():
        entry = {
            "n_reps_this_category_appeared_in": len(reps_per_category[category]),
            "n_reps_succeeded_total": n_succeeded,
            "n_reps_failed_total": n_failed,
            "raw_model_names_seen": raw_names_seen[category],
        }
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
        averaged[category] = entry

    # Build the model-selection audit file
    audit = {
        "temperature": temperature,
        "n_reps_succeeded": n_succeeded,
        "n_reps_failed": n_failed,
        "categories": {
            category: {
                "n_reps_appeared_in": len(reps_per_category[category]),
                "raw_names_seen": raw_names_seen[category],
            }
            for category in metrics_by_category
        },
        "deep_learning_flagged": deep_learning_hits,
    }

    return averaged, audit


def main():
    for temperature in TEMPERATURES:
        averaged, audit = process_one_temperature(temperature)

        if averaged:
            out_path = os.path.join(
                OUTPUT_DIR, f"results_temp_{temperature}_{OUTCOME}_averaged.json"
            )
            with open(out_path, "w") as f:
                json.dump(averaged, f, indent=2)
            print(f"  -> wrote {out_path} ({len(averaged)} model categories)")
        else:
            print(f"  -> no successful reps for temperature={temperature}, skipping averaged output")

        audit_path = os.path.join(OUTPUT_DIR, f"model_selection_audit_temp_{temperature}.json")
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"  -> wrote {audit_path}")


if __name__ == "__main__":
    main()
