"""Per-candidate-method tracking, independent of whether a run finishes.

Every literature-search candidate is classified by how far it got through the
pipeline (benchmarked / implemented but not benchmarked / not implemented /
never reached because the run failed earlier) and by the source the literature
agent cited for it. ``build_method_entries`` computes this for one run and is
embedded directly in that run's ``run_manifest.json``; ``append_method_entries``
also appends the same rows to a single cross-run JSONL file so method
selection frequency, source variability, and success/failure rates can be
aggregated without re-opening every run directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.core.schemas import LiteratureReviewResult, ModelCode
from src.utils.naming import slugify

DEFAULT_LOG_PATH = Path("/app/experiments/method_log.jsonl")


def build_method_entries(
    literature_result: LiteratureReviewResult | None,
    model_code: list[ModelCode] | None,
    raw_results: dict | None,
) -> list[dict]:
    """Classify each searched candidate by how far it got through the pipeline.

    ``model_code`` is None when the programming stage never finished (or was
    never reached); ``raw_results`` is None when the benchmarking stage never
    finished. Distinguishing "never reached" from "reached but produced
    nothing" is what lets a partial/failed run still say which methods were
    tried and which of those actually produced results.
    """
    if literature_result is None:
        return []

    implemented_slugs = (
        {model.model_name for model in model_code} if model_code is not None else None
    )
    entries = []
    for candidate in literature_result.candidates:
        slug = slugify(candidate.model_name)
        metrics = raw_results.get(slug) if raw_results is not None else None
        if metrics is not None:
            status = "benchmarked"
        elif implemented_slugs is not None and slug in implemented_slugs:
            status = "implemented_not_benchmarked"
        elif implemented_slugs is not None:
            status = "not_implemented"
        else:
            status = "pipeline_failed_before_programming"

        entries.append({
            "model_name": candidate.model_name,
            "slug": slug,
            "resource_name": candidate.resource_name,
            "resource_link": candidate.resource_link,
            "status": status,
            "metrics": metrics,
        })
    return entries


def append_method_entries(
    entries: list[dict],
    *,
    run_id: str,
    experiment_id: str,
    condition_id: str,
    replicate: int,
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> None:
    """Append one JSON line per method entry to the shared cross-run log."""
    if not entries:
        return

    recorded_at = datetime.now(timezone.utc).isoformat()
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            row = {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "condition_id": condition_id,
                "replicate": replicate,
                "recorded_at": recorded_at,
                **entry,
            }
            f.write(json.dumps(row) + "\n")
