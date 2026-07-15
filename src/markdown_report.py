from pathlib import Path

from src.schemas import BenchmarkReport


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
        "| Model | Accuracy | F1 | Precision | Recall | AUROC | Brier | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for model in report.entries:
        lines.append(
            "| "
            f"{_escape_cell(model.model_name)} | {model.accuracy} | {model.f1} | "
            f"{model.precision} | {model.recall} | {model.auroc} | {model.brier} | "
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
