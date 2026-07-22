"""Bar plots comparing generated diagnosis vs true diagnosis for each run's models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def _real(path: str) -> Path:
    if path.startswith("/app/"):
        return Path.cwd() / path.removeprefix("/app/")
    return Path(path)


# predictions.json field names changed over time (truth -> true_diagnosis,
# diagnosed -> generated_diagnosis -> prediction/diagnosis, since each run's
# evaluator is now LLM-generated and free to invent new field names each time);
# accept any so every run's format still plots.
_TRUE_KEYS = ("true_diagnosis", "true_outcome", "true_binary_outcome", "truth")
_GENERATED_KEYS = ("generated_diagnosis", "predicted_diagnosis", "prediction", "diagnosis", "diagnosed")


def _predictions_by_model(data) -> dict[str, list[dict]]:
    # Older format: {model_name: [record, ...]}. Newer format: a flat list of
    # records each carrying its own "model" field.
    if isinstance(data, dict):
        return data
    by_model: dict[str, list[dict]] = {}
    for record in data:
        by_model.setdefault(record["model"], []).append(record)
    return by_model


def _first_present(record: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in record:
            return key
    return None


def _confusion_counts(records: list[dict]) -> tuple[int, int, int, int] | None:
    true_key = _first_present(records[0], _TRUE_KEYS)
    generated_key = _first_present(records[0], _GENERATED_KEYS)
    if true_key is None or generated_key is None:
        return None
    true_positive = sum(1 for r in records if r[true_key] and r[generated_key])
    false_negative = sum(1 for r in records if r[true_key] and not r[generated_key])
    false_positive = sum(1 for r in records if not r[true_key] and r[generated_key])
    true_negative = sum(1 for r in records if not r[true_key] and not r[generated_key])
    return true_positive, true_negative, false_positive, false_negative


def plot_run(run_dir: Path) -> Path | None:
    predictions_path = run_dir / "predictions.json"
    if not predictions_path.exists():
        return None
    data = json.loads(predictions_path.read_text())
    if not data:
        return None
    data = _predictions_by_model(data)

    model_names = []
    true_positive_counts = []
    true_negative_counts = []
    false_positive_counts = []
    false_negative_counts = []
    for model_name in sorted(data.keys()):
        counts = _confusion_counts(data[model_name])
        if counts is None:
            continue
        tp, tn, fp, fn = counts
        model_names.append(model_name)
        true_positive_counts.append(tp)
        true_negative_counts.append(tn)
        false_positive_counts.append(fp)
        false_negative_counts.append(fn)

    if not model_names:
        return None

    # Stack correct (green) below incorrect (red/orange) so right-vs-wrong
    # is immediately visible per model, with the breakdown still available.
    x = range(len(model_names))
    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.8), 5))
    bottom = [0] * len(model_names)
    segments = (
        (true_positive_counts, "Correct: caught the diagnosis", "#2ca02c"),
        (true_negative_counts, "Correct: correctly cleared", "#98df8a"),
        (false_positive_counts, "Wrong: false alarm", "#ff7f0e"),
        (false_negative_counts, "Wrong: missed diagnosis", "#d62728"),
    )
    for values, label, color in segments:
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]

    totals = [sum(counts) for counts in zip(true_positive_counts, true_negative_counts, false_positive_counts, false_negative_counts)]
    correct_totals = [tp + tn for tp, tn in zip(true_positive_counts, true_negative_counts)]
    max_total = max(totals) if totals else 0
    ax.set_ylim(0, max_total * 1.15)
    for i, (correct, total) in enumerate(zip(correct_totals, totals)):
        accuracy = correct / total if total else 0.0
        ax.text(i, total + max_total * 0.02, f"{accuracy:.0%} correct", ha="center", va="bottom")

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel("Patient count")
    ax.set_title(f"Diagnosis accuracy — run {run_dir.name}", pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()

    plot_path = run_dir / "diagnosis_accuracy.png"
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def plot_all(generated_code_dir: Path) -> list[Path]:
    written = []
    for run_dir in sorted(generated_code_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        plot_path = plot_run(run_dir)
        if plot_path is not None:
            written.append(plot_path)
    return written


if __name__ == "__main__":
    generated_code_dir = _real("/app/generated_code")
    for path in plot_all(generated_code_dir):
        print(path)
