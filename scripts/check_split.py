"""Check a three-group patient split against the study data.

Rebuilds the cohort and outcome from the raw tables and validates the split file:
valid labels, every cohort patient labeled, group sizes near 60/20/20, similar
event rates, and both outcome classes in every group. With --provided-split it
also checks the study design (the old validation patients are the new test set).
Exits 1 if any check fails. Output holds counts only, never patient IDs.

Usage:
    python scripts/check_split.py --data-dir <folder with the raw tables> [--split-file <csv>] [--provided-split <csv>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.split_check import DEFAULT_FRACTION_TOLERANCE, DEFAULT_RATE_TOLERANCE, check_split

LABELS = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-file", help="default: <data-dir>/patient_split.csv")
    parser.add_argument("--provided-split", help="a two-group split (train, validation) to check the study design against")
    parser.add_argument("--fraction-tolerance", type=float, default=DEFAULT_FRACTION_TOLERANCE)
    parser.add_argument("--rate-tolerance", type=float, default=DEFAULT_RATE_TOLERANCE)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split = pd.read_csv(args.split_file or data_dir / "patient_split.csv", dtype=str)
    provided = pd.read_csv(args.provided_split, dtype=str) if args.provided_split else None
    report = check_split(split, data_dir, provided, fraction_tolerance=args.fraction_tolerance, rate_tolerance=args.rate_tolerance)
    for check in report.checks:
        print(f"[{LABELS[check.status]}] {check.name}" + (f" -- {check.detail}" if check.detail else ""))
    counts = report.to_dict()["counts"]
    print(f"\n{counts['pass']} passed, {counts['warn']} warnings, {counts['fail']} failed -> {'SPLIT OK' if report.passed else 'SPLIT NOT OK'}")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
