from pathlib import Path
import json

def collect_benchmark_results(run_id: str) -> dict:
    """Reads the single benchmark_results.json the agent was told to produce for this run."""
    results_path = Path(f"/app/generated_code/{run_id}/benchmark_results.json")
    if not results_path.exists():
        raise FileNotFoundError(f"No benchmark_results.json found for run {run_id}")
    return json.loads(results_path.read_text())

def collect_benchmark_scripts(run_id: str) -> dict[str, str]:
    """Reads test_<model_name>_benchmark.py files, scoped to this run's folder only."""
    run_dir = Path(f"/app/generated_code/{run_id}")
    scripts = {}
    for file in run_dir.glob("*/test_*_benchmark.py"):
        model_name = file.stem.removeprefix("test_").removesuffix("_benchmark")
        scripts[model_name] = file.read_text()
    return scripts
