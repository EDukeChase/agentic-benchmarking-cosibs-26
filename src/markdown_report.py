from pathlib import Path

from src.schemas import BenchmarkReport


def save_error_markdown(
    filename: str | Path,
    *,
    run_id: str,
    stage: str,
    error: BaseException,
    traceback_text: str,
) -> None:
    """Write a useful report when a run cannot produce a benchmark report."""
    error_name = type(error).__name__
    message = str(error).strip() or "No error message was provided."
    content = "\n".join([
        "# Benchmark Report",
        "",
        "## Run Status",
        "",
        "**Failed**",
        "",
        f"- Run ID: `{run_id}`",
        f"- Failed stage: `{stage}`",
        f"- Error type: `{error_name}`",
        "",
        "## Error",
        "",
        message,
        "",
        "## Traceback",
        "",
        "```text",
        traceback_text.rstrip(),
        "```",
        "",
        "Partial artifacts generated before the failure may still be present in this run directory.",
        "",
    ])
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _escape_cell(value: object) -> str:
    """Keep model names and statuses safe inside a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def render_markdown(report: BenchmarkReport) -> str:
    """Render a complete, human-readable benchmark report as Markdown."""
    lines = [
        "# Benchmark Report",
        "",
        "## Overall Summary",
        "",
        report.summary.strip(),
        "",
        "## Recommendations",
        "",
        report.recommendations.strip(),
        "",
        "## Model Results",
        "",
        "Primary metrics are shown first. Accuracy is secondary because the outcome is imbalanced.",
        "",
        "| Model | AUROC | F1 | Recall | Precision | Brier | Threshold | Accuracy (secondary) | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for model in report.entries:
        lines.append(
            "| "
            f"{_escape_cell(model.model_name)} | {model.auroc} | {model.f1} | "
            f"{model.recall} | {model.precision} | {model.brier} | {model.threshold} | "
            f"{model.accuracy} | "
            f"{_escape_cell(model.status)} |"
        )

    for model in report.entries:
        lines.extend([
            "",
            f"### {_escape_cell(model.model_name)}",
            "",
            "#### Rationale",
            "",
            model.rationale.strip(),
            "",
            "#### Implementation Notes",
            "",
            model.documentation.strip(),
            "",
        ])

    return "\n".join(lines)


def save_markdown(report: BenchmarkReport, filename: str | Path) -> None:
    """Write the canonical Markdown report using UTF-8 encoding."""
    Path(filename).write_text(render_markdown(report), encoding="utf-8")
