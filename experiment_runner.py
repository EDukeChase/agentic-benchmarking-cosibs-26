"""Run a manifest-defined experiment with independent full-pipeline replicates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from src.statistical_analysis import analyze


def _real(path: str) -> Path:
    if path.startswith("/app/"):
        return Path.cwd() / path.removeprefix("/app/")
    return Path(path)


def make_environment(manifest: dict, condition: dict, condition_id: str,
                     replicate: int, run_id: str) -> dict:
    """Create the environment variables read by main.py."""
    environment = os.environ.copy()
    environment["BENCHMARK_RUN_ID"] = run_id
    environment["BENCHMARK_EXPERIMENT_ID"] = manifest["experiment_id"]
    environment["BENCHMARK_CONDITION_ID"] = condition_id
    environment["BENCHMARK_REPLICATE"] = str(replicate)
    environment["BENCHMARK_CONDITION_JSON"] = json.dumps(condition)
    environment["BENCHMARK_TASK_JSON"] = json.dumps(manifest["task"])
    return environment


def read_successful_run(run_dir: Path, experiment_id: str,
                        condition_id: str, replicate: int,
                        run_id: str) -> list[dict]:
    """Turn one successful run into rows for the combined CSV file."""
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    benchmark_results = json.loads((run_dir / "benchmark_results.json").read_text())
    rows = []

    for model_name, metrics in benchmark_results.items():
        row = {
            "experiment_id": experiment_id,
            "condition_id": condition_id,
            "replicate": replicate,
            "run_id": run_id,
            "status": "success",
            "model_name": model_name,
            "runtime_seconds": run_manifest["runtime_seconds"],
            "leakage_passed": run_manifest["leakage_passed"],
        }
        row.update(metrics)
        row.update(run_manifest["token_usage"])
        rows.append(row)

    return rows


def run_experiment(manifest_path: str) -> int:
    manifest = json.loads(Path(manifest_path).read_text())
    experiment_id = manifest["experiment_id"]
    output_root = _real(f"/app/experiments/results/{experiment_id}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_manifest.json").write_text(json.dumps(manifest, indent=2))
    rows: list[dict] = []
    failures = 0
    for condition_spec in manifest["conditions"]:
        condition = {**manifest["baseline"], **condition_spec.get("overrides", {})}
        condition_id = condition_spec["condition_id"]
        for replicate in range(1, int(condition_spec.get("repetitions", 1)) + 1):
            run_id = uuid.uuid4().hex[:8]
            environment = make_environment(
                manifest, condition, condition_id, replicate, run_id
            )
            print(f"[{condition_id} {replicate}] starting run {run_id}", flush=True)
            completed = subprocess.run([sys.executable, "main.py"], cwd=Path.cwd(), env=environment)
            run_dir = _real(f"/app/generated_code/{run_id}")
            if completed.returncode != 0:
                failures += 1
                rows.append({
                    "condition_id": condition_id,
                    "replicate": replicate,
                    "run_id": run_id,
                    "status": "failed",
                })
                _write_csv(output_root / "results.csv", rows)
                continue

            new_rows = read_successful_run(
                run_dir, experiment_id, condition_id, replicate, run_id
            )
            rows.extend(new_rows)
            _write_csv(output_root / "results.csv", rows)
    _write_csv(output_root / "results.csv", rows)
    successful = [row for row in rows if row.get("status") == "success"]
    if successful:
        analyze(output_root / "results.csv", output_root / "statistical_analysis.json")
    summary = {"experiment_id": experiment_id, "failures": failures, "rows": len(rows)}
    (output_root / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    return 1 if failures else 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="experiments/manifest.json")
    args = parser.parse_args()
    raise SystemExit(run_experiment(args.manifest))
