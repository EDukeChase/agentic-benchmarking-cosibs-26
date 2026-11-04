"""Create a three-way study data folder from a folder with a two-group split.

Writes <out-dir>/patient_split.csv (train, validation, test) and links every
other file from <data-dir> into <out-dir>, so the source folder is never
modified and the large tables are not copied.

Usage:
    python scripts/make_study_split.py --data-dir <folder with a two-group patient_split.csv> --out-dir <new study folder>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.cohort import build_cohort
from src.evaluation.study_split import SPLIT_GROUPS, make_three_way_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.25, help="share of train carved into validation")
    parser.add_argument("--seed", type=int, default=5960)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir == data_dir:
        raise SystemExit("--out-dir must differ from --data-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = build_cohort(data_dir)
    provided = pd.read_csv(data_dir / "patient_split.csv")
    split = make_three_way_split(provided, cohort, args.validation_fraction, args.seed)
    split.to_csv(out_dir / "patient_split.csv", index=False)

    for source in data_dir.iterdir():
        target = out_dir / source.name
        if source.name == "patient_split.csv" or target.exists() or target.is_symlink():
            continue
        target.symlink_to(source)

    summary = cohort.merge(split, on="patient_id").groupby("split")["y"].agg(patients="size", events="sum")
    summary["rate"] = (summary["events"] / summary["patients"]).round(3)
    print(summary.reindex(list(SPLIT_GROUPS)).to_string())
    print(f"cohort: {len(cohort)} patients, {int(cohort['y'].sum())} events")
    print(f"wrote {out_dir / 'patient_split.csv'}")


if __name__ == "__main__":
    main()
