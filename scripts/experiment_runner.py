"""Run every condition and replicate defined by an experiment manifest.

The manifest describes a baseline configuration, condition-specific overrides,
the prediction task, and the number of repetitions. Each repetition launches a
fresh ``main.py`` subprocess, giving it configuration through environment
variables. Successful model results are accumulated in one CSV, then passed to
the statistical-analysis module after all requested runs finish.

Results are written after every subprocess rather than only at the end. This
makes long experiments easier to inspect and preserves completed work if a later
run fails or the overall process is interrupted.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from src.evaluation.statistical_analysis import analyze


def _real(path: str) -> Path:
    """Translate a container-style /app path into a path usable on this host.

    Pipeline code commonly refers to files as if the repository were mounted at
    /app. During local execution, the repository root is the current directory,
    so only that prefix needs to be replaced. Ordinary host paths pass through
    unchanged.
    """
    if path.startswith("/app/"):
        return Path.cwd() / path.removeprefix("/app/")
    return Path(path)


def make_environment(manifest: dict, condition: dict, condition_id: str,
                     replicate: int, run_id: str) -> dict:
    """Build an isolated environment-variable mapping for one pipeline run.

    Starting with a copy of the current environment preserves credentials and
    normal runtime settings. The BENCHMARK variables then identify this exact
    experiment/condition/replicate and serialize its configuration as JSON for
    ``main.py``. Returning a new mapping avoids modifying the runner's own process
    environment between repetitions.
    """
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
    """Convert one completed run's artifacts into model-level CSV rows.

    ``benchmark_results.json`` contains one metric dictionary per candidate
    model, while ``run_manifest.json`` contains run-wide facts such as elapsed
    time, leakage checks, and token usage. These sources are merged so every row
    is self-contained and can be grouped later without reopening run folders.
    """
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    benchmark_results = json.loads((run_dir / "benchmark_results.json").read_text())
    rows = []

    # A run evaluating multiple models produces multiple output rows. Run-level
    # values repeat intentionally because each row represents one model result.
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
    """Execute the complete manifest and return a process-style status code.

    A return value of zero means every requested subprocess succeeded; one means
    at least one failed. Failures are recorded and do not stop later conditions or
    replicates, allowing the experiment to collect as much evidence as possible.
    """
    # Resolve the experiment-level output directory once. Saving a copy of the
    # manifest records the exact input used, even if the source file changes later.
    manifest = json.loads(Path(manifest_path).read_text())
    experiment_id = manifest["experiment_id"]
    output_root = _real(f"/app/experiments/results/{experiment_id}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_manifest.json").write_text(json.dumps(manifest, indent=2))
    rows: list[dict] = []
    failures = 0

    # A condition begins with every baseline setting, then replaces only keys
    # named in its overrides. This keeps manifests concise and ensures conditions
    # differ only where the author explicitly requested a change.
    for condition_spec in manifest["conditions"]:
        condition = {**manifest["baseline"], **condition_spec.get("overrides", {})}
        condition_id = condition_spec["condition_id"]

        # Replicates are numbered from one because the number is written into
        # manifests and result tables for people to read, not used as a list index.
        for replicate in range(1, int(condition_spec.get("repetitions", 1)) + 1):
            # Each run receives a short unique ID so its generated models and
            # reports are isolated from every other subprocess.
            run_id = uuid.uuid4().hex[:8]
            environment = make_environment(
                manifest, condition, condition_id, replicate, run_id
            )
            print(f"[{condition_id} {replicate}] starting run {run_id}", flush=True)

            # Do not capture stdout/stderr: allowing them to stream to the parent
            # terminal gives live progress and preserves pipeline diagnostics.
            completed = subprocess.run([sys.executable, "main.py"], cwd=Path.cwd(), env=environment)
            run_dir = _real(f"/app/generated_code/{run_id}")
            if completed.returncode != 0:
                # Record enough identity fields to locate and diagnose the failed
                # run. It has no model metrics because its artifacts may be partial.
                failures += 1
                rows.append({
                    "condition_id": condition_id,
                    "replicate": replicate,
                    "run_id": run_id,
                    "status": "failed",
                })
                _write_csv(output_root / "results.csv", rows)
                continue

            # Only a zero-exit run is trusted to have finalized manifests and
            # benchmark results. Merge those artifacts into the cumulative table.
            new_rows = read_successful_run(
                run_dir, experiment_id, condition_id, replicate, run_id
            )
            rows.extend(new_rows)
            _write_csv(output_root / "results.csv", rows)
    # The final write is intentionally redundant with checkpoint writes above: it
    # guarantees the on-disk CSV reflects the complete in-memory row collection.
    _write_csv(output_root / "results.csv", rows)
    successful = [row for row in rows if row.get("status") == "success"]
    if successful:
        # Statistical analysis requires at least one real model result. A manifest
        # whose every run failed still receives a summary but no misleading empty
        # analysis report.
        analyze(output_root / "results.csv", output_root / "statistical_analysis.json")
    summary = {"experiment_id": experiment_id, "failures": failures, "rows": len(rows)}
    (output_root / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    return 1 if failures else 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Rewrite the cumulative CSV using the union of all observed columns.

    Success rows contain metrics and token fields that failure rows do not. Taking
    the union creates a stable rectangular file; ``csv.DictWriter`` leaves absent
    values blank. Rewriting is safe here because ``rows`` always contains every
    result accumulated so far.
    """
    if not rows:
        return
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # The manifest argument is optional for convenience; running the file with no
    # arguments uses the repository's standard experiment definition.
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="experiments/manifest.json")
    args = parser.parse_args()
    raise SystemExit(run_experiment(args.manifest))
