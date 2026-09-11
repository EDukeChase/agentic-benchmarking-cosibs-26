"""Repository-owned train/validation/test split, created before any agent runs.

The benchmarking stage's feature extraction and model code is entirely
agent-generated, so the only way to *guarantee* every run (and every
repeated selection of the same candidate model) scores against the same
patients is to create and freeze the split here, in code the LLM cannot
silently deviate from, and hand the agent an existing file to load rather
than a chance to compute its own.

This also backs the verification step in ``compare_split_usage``: the
benchmarking agent self-reports the patient IDs it actually used, and this
module compares that report against the frozen split we created. A used
patient missing from the frozen split's cohort entirely, or used in a
different group than assigned (e.g. a test patient appearing in training),
is a real correctness problem and should fail loudly. A model legitimately
using fewer patients than assigned (e.g. because it cannot use certain
patients) is not a violation, so that case is reported, not flagged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.deterministic import _real_path, create_splits
from src.settings.config import BenchmarkTaskConfig

SPLIT_PARTS = ("train", "validation", "test")


def build_cohort(task: BenchmarkTaskConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (patient_ids, y) for every patient with an available event file.

    This mirrors the cohort-inclusion rule described in the benchmarking
    prompt (sorted patient-ID order, skip only patients with no event file)
    without doing the full per-patient feature extraction — the split only
    needs patient identity and outcome label, not features.
    """
    root = _real_path(task.data_root)
    labels = pd.read_csv(root / "labels.csv")
    if task.outcome not in labels or task.patient_id_column not in labels:
        raise ValueError(f"Task columns missing: {task.patient_id_column}, {task.outcome}")

    patient_ids: list[int] = []
    targets: list[int] = []
    for record in labels.sort_values(task.patient_id_column).to_dict("records"):
        patient_id = int(record[task.patient_id_column])
        if not (root / "patient_data_all" / f"patient_{patient_id}.csv").exists():
            continue
        patient_ids.append(patient_id)
        targets.append(int(bool(record[task.outcome])))

    return np.asarray(patient_ids), np.asarray(targets)


def canonical_split_hash(split: dict[str, list[int]]) -> str:
    """A stable fingerprint for a split, independent of key/list ordering."""
    canonical = {part: sorted(int(pid) for pid in split[part]) for part in SPLIT_PARTS}
    encoded = json.dumps(canonical, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def ensure_split(task: BenchmarkTaskConfig) -> dict[str, list[int]]:
    """Load the frozen split for this dataset/outcome, creating it if absent.

    Called once per pipeline run, before the benchmarking agent starts, so
    the agent always finds an existing file — the "create a new split"
    branch belongs to this function alone, not to agent-generated code.
    """
    split_path = _real_path(task.split_file)
    patient_ids, y = build_cohort(task)
    cohort_ids = {int(pid) for pid in patient_ids.tolist()}

    if split_path.exists():
        raw = json.loads(split_path.read_text())
        y_by_patient = dict(zip(patient_ids.tolist(), y.tolist()))
        split: dict[str, list[int]] = {}
        for part in SPLIT_PARTS:
            ids = sorted(int(pid) for pid in raw.get(part, []) if int(pid) in cohort_ids)
            labels_here = [y_by_patient[pid] for pid in ids]
            if len(set(labels_here)) < 2:
                raise ValueError(
                    f"Frozen split '{part}' has fewer than two classes once "
                    "aligned to the current cohort"
                )
            split[part] = ids
    else:
        split = create_splits(patient_ids, y, task)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split, indent=2))

    groups = {part: set(split[part]) for part in SPLIT_PARTS}
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Patient leakage detected: frozen split has overlapping groups")

    return split


def _compare_ids(actual_ids: set[int], assigned_ids: set[int], all_frozen_ids: set[int]) -> dict:
    """Classify one set of actually-used IDs against its assigned group.

    ``foreign`` IDs (used but absent from the frozen cohort entirely) and
    ``misplaced`` IDs (used but assigned to a *different* group) are both
    correctness problems. Using a proper subset of the assigned group is not
    — there can be legitimate reasons a model uses fewer patients.
    """
    foreign = actual_ids - all_frozen_ids
    misplaced = (actual_ids & all_frozen_ids) - assigned_ids
    return {
        "n_actual": len(actual_ids),
        "n_assigned": len(assigned_ids),
        "exact_match": actual_ids == assigned_ids,
        "is_subset_of_assigned": actual_ids <= assigned_ids,
        "foreign_patient_ids": sorted(foreign),
        "misplaced_patient_ids": sorted(misplaced),
    }


def compare_split_usage(
    canonical_split: dict[str, list[int]], actual_split: dict[str, list[int]]
) -> dict[str, dict]:
    """Compare a reported train/validation/test usage against the frozen split."""
    canonical_sets = {part: set(int(pid) for pid in canonical_split[part]) for part in SPLIT_PARTS}
    all_frozen_ids = set().union(*canonical_sets.values())
    result = {}
    for part in SPLIT_PARTS:
        actual_ids = {int(pid) for pid in actual_split.get(part, [])}
        result[part] = _compare_ids(actual_ids, canonical_sets[part], all_frozen_ids)
    return result


def compare_test_set_usage(canonical_split: dict[str, list[int]], actual_test_ids: list[int]) -> dict:
    """Compare one model's actually-scored test patients against the frozen test group.

    Used for the per-model check derived from ``predictions.json`` (which
    already lists exactly the test patients each model was scored on), in
    addition to the run-level check derived from the agent's own report.
    """
    canonical_sets = {part: set(int(pid) for pid in canonical_split[part]) for part in SPLIT_PARTS}
    all_frozen_ids = set().union(*canonical_sets.values())
    return _compare_ids({int(pid) for pid in actual_test_ids}, canonical_sets["test"], all_frozen_ids)


def split_usage_is_clean(comparison: dict[str, dict]) -> bool:
    """True if no comparison entry found foreign or misplaced patient IDs."""
    return all(not c["foreign_patient_ids"] and not c["misplaced_patient_ids"] for c in comparison.values())
