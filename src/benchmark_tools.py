from pathlib import Path
import json


REQUIRED_BENCHMARK_METRICS = {
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auroc",
    "brier",
}


def _normalize_metrics(model_name: str, metrics: dict) -> dict:
    """Normalize harmless LLM naming variation and validate one model result.

    The benchmarking agent occasionally uses scikit-learn's function-style name
    ``brier_score`` even though the repository schema calls the same observed
    value ``brier``. Renaming that key preserves the real calculated metric; no
    missing value is estimated or invented. All other required fields remain
    strict so malformed benchmark artifacts fail close to their source.
    """
    if not isinstance(metrics, dict):
        raise ValueError(
            f"Benchmark result for {model_name!r} must be a JSON object"
        )

    normalized = dict(metrics)
    if "brier" not in normalized and "brier_score" in normalized:
        normalized["brier"] = normalized.pop("brier_score")

    missing = sorted(REQUIRED_BENCHMARK_METRICS - normalized.keys())
    if missing:
        raise ValueError(
            f"Benchmark result for {model_name!r} is missing required metrics: "
            f"{', '.join(missing)}"
        )

    for metric in REQUIRED_BENCHMARK_METRICS | {"threshold"}:
        if metric in normalized and not isinstance(normalized[metric], (int, float)):
            raise ValueError(
                f"Benchmark metric {model_name!r}/{metric} must be numeric; "
                f"received {type(normalized[metric]).__name__}"
            )
    return normalized

def collect_benchmark_results(run_id: str) -> dict:
    """Reads the single benchmark_results.json the agent was told to produce for this run."""
    results_path = Path(f"/app/generated_code/{run_id}/benchmark_results.json")
    if not results_path.exists():
        raise FileNotFoundError(f"No benchmark_results.json found for run {run_id}")
    raw_results = json.loads(results_path.read_text())
    if not isinstance(raw_results, dict):
        raise ValueError("benchmark_results.json must contain a JSON object")
    return {
        model_name: _normalize_metrics(model_name, metrics)
        for model_name, metrics in raw_results.items()
    }

def collect_benchmark_scripts(run_id: str) -> dict[str, str]:
    """Reads test_<model_name>_benchmark.py files, scoped to this run's folder only."""
    run_dir = Path(f"/app/generated_code/{run_id}")
    scripts = {}
    for file in run_dir.glob("*/test_*_benchmark.py"):
        model_name = file.stem.removeprefix("test_").removesuffix("_benchmark")
        scripts[model_name] = file.read_text()
    return scripts
