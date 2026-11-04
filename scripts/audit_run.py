"""Audit one run's deliverables against the raw data.

Checks the format of results.json, predictions.csv and splits.csv, then
recomputes the cohort, outcome, split and reported metrics from the data folder
and compares them with what the run reported. Exits 1 if any check fails.

The JSON summary holds check names and counts only, never patient IDs.

Usage:
    python scripts/audit_run.py --run-dir <run output folder> --data-dir <study data folder> [--json audit.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.audit import DEFAULT_TOLERANCE, audit_run

LABELS = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="folder holding results.json, predictions.csv, splits.csv")
    parser.add_argument("--data-dir", required=True, help="study data folder with the raw tables and patient_split.csv")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="allowed absolute difference for metrics and thresholds")
    parser.add_argument("--json", help="also write the full result to this file")
    parser.add_argument("--quiet", action="store_true", help="print only warnings, failures and the summary")
    args = parser.parse_args()

    report = audit_run(args.run_dir, args.data_dir, args.tolerance)
    for check in report.checks:
        if args.quiet and check.status == "pass":
            continue
        print(f"[{LABELS[check.status]}] {check.name}" + (f" -- {check.detail}" if check.detail else ""))
    counts = report.to_dict()["counts"]
    print(f"\n{counts['pass']} passed, {counts['warn']} warnings, {counts['fail']} failed -> {'AUDIT PASSED' if report.passed else 'AUDIT FAILED'}")
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
