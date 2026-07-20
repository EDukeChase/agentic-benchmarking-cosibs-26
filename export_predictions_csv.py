"""Convert each run's predictions.json under generated_code/ into a predictions.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _real(path: str) -> Path:
    if path.startswith("/app/"):
        return Path.cwd() / path.removeprefix("/app/")
    return Path(path)


def export_predictions_csv(run_dir: Path) -> Path | None:
    predictions_path = run_dir / "predictions.json"
    if not predictions_path.exists():
        return None
    data = json.loads(predictions_path.read_text())
    if not data:
        return None

    fieldnames = ["model"] + list(next(iter(data.values()))[0].keys())
    csv_path = run_dir / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, records in data.items():
            for record in records:
                writer.writerow({"model": model_name, **record})
    return csv_path


def export_all(generated_code_dir: Path) -> list[Path]:
    written = []
    for run_dir in sorted(generated_code_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = export_predictions_csv(run_dir)
        if csv_path is not None:
            written.append(csv_path)
    return written


if __name__ == "__main__":
    generated_code_dir = _real("/app/generated_code")
    for path in export_all(generated_code_dir):
        print(path)
