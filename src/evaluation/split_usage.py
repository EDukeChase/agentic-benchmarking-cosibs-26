"""Record which patients each method says it used, against the study split.

This is a record, not a gate. Every method's split_usage.json entry (the patients
it fitted on, made choices on, and scored on) is compared with patient_split.csv
and the counts are written to split_check.json. A run is never failed because a
method used patients outside the group the study prompt allows. The one thing
that raises is a patient ID that is not in the split file at all, which means the
agent did not work from the study data.

Only counts are written, never patient IDs. The full audit (scripts/audit_run.py)
repeats these checks against the rebuilt cohort after the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ALLOWED_GROUPS = {
    "fit": {"train"},
    "choices": {"train", "validation"},
    "test": {"test"},
}


class UnknownPatientIdsError(RuntimeError):
    """A method reported patient IDs that are not in the split file at all."""


def load_split(data_dir: str | Path) -> dict[str, str]:
    """patient_id -> group, with patient IDs as text."""
    split = pd.read_csv(Path(data_dir) / "patient_split.csv", dtype=str)
    return dict(zip(split["patient_id"], split["split"]))


def compare_usage(usage: dict, group_of: dict[str, str]) -> dict:
    """Per method, how many reported patients fall outside the allowed groups."""
    test_group = {pid for pid, group in group_of.items() if group == "test"}
    methods = {}
    for method, entry in usage.items():
        record = {}
        unknown_total = 0
        for part, allowed in ALLOWED_GROUPS.items():
            reported = entry.get(part) if isinstance(entry, dict) else None
            ids = [str(pid) for pid in reported] if isinstance(reported, list) else []
            unknown = sum(1 for pid in ids if pid not in group_of)
            outside = sum(1 for pid in ids if pid in group_of and group_of[pid] not in allowed)
            record[part] = {
                "reported": len(ids),
                "distinct": len(set(ids)),
                "not_in_split_file": unknown,
                "outside_allowed_groups": outside,
            }
            unknown_total += unknown
        test_ids = {str(pid) for pid in entry.get("test", [])} if isinstance(entry, dict) and isinstance(entry.get("test"), list) else set()
        record["test"]["test_patients_not_scored"] = len(test_group - test_ids)
        record["unknown_total"] = unknown_total
        methods[method] = record
    return methods


def record_split_usage(out_dir: str | Path, data_dir: str | Path, records_dir: str | Path) -> dict:
    """Write split_check.json to the records folder and return its content.

    Raises UnknownPatientIdsError, after writing the file, only for IDs that are
    not in the split file at all.
    """
    out_dir, records_dir = Path(out_dir), Path(records_dir)
    records_dir.mkdir(parents=True, exist_ok=True)
    usage_path = out_dir / "split_usage.json"
    if not usage_path.exists():
        check = {"split_usage_json": "missing", "methods": {}}
    else:
        try:
            usage = json.loads(usage_path.read_text())
        except json.JSONDecodeError:
            usage = None
        if not isinstance(usage, dict):
            check = {"split_usage_json": "not a valid JSON object", "methods": {}}
        else:
            check = {"split_usage_json": "present", "methods": compare_usage(usage, load_split(data_dir))}
    (records_dir / "split_check.json").write_text(json.dumps(check, indent=2))

    unknown = sum(m["unknown_total"] for m in check["methods"].values())
    if unknown:
        raise UnknownPatientIdsError(
            f"split_usage.json lists {unknown} patient ID(s) that are not in patient_split.csv."
        )
    return check

