"""Validate a three-group patient split against the study data.

Works on any patient_split.csv, whoever or whatever produced it. The cohort and
outcome are rebuilt from the raw tables, so the check does not depend on how the
split was made. Reuses the Report type of the run audit.

Fails when: the file is malformed, labels are not train/validation/test, a
patient appears twice, a cohort patient has no label, a group is far from its
target share of the cohort, a group's event rate is far from the overall rate,
or a group has only one outcome class. With a provided two-group split it can
also check the study design: the old validation patients are the new test set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .audit import SPLITS, Report
from .cohort import build_cohort

DEFAULT_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}
DEFAULT_FRACTION_TOLERANCE = 0.02  # absolute, on each group's share of the cohort
DEFAULT_RATE_TOLERANCE = 0.03  # absolute, on each group's event rate against the overall rate


def check_split(split: pd.DataFrame, data_dir: str | Path, provided: pd.DataFrame | None = None,
                fractions: dict[str, float] = DEFAULT_FRACTIONS,
                fraction_tolerance: float = DEFAULT_FRACTION_TOLERANCE,
                rate_tolerance: float = DEFAULT_RATE_TOLERANCE) -> Report:
    report = Report()
    if set(split.columns) != {"patient_id", "split"}:
        report.add("split file: columns are patient_id and split", False, f"found {list(split.columns)}")
        return report
    report.add("split file: columns are patient_id and split", True)

    split = split.astype({"patient_id": str, "split": str})
    labels = sorted(split["split"].unique())
    report.add("split file: labels are train, validation and test", set(labels) <= set(SPLITS), f"found {labels}")
    duplicated = int(split["patient_id"].duplicated().sum())
    report.add("split file: each patient appears once", duplicated == 0, f"{duplicated} repeated patient IDs" if duplicated else "")

    data_dir = Path(data_dir)
    cohort = build_cohort(data_dir)
    known = set(pd.read_csv(data_dir / "patients.csv", usecols=["Id"], dtype=str)["Id"])
    unknown = int((~split["patient_id"].isin(known)).sum())
    report.add("split file: every listed patient exists in patients.csv", unknown == 0, f"{unknown} unknown patient IDs" if unknown else "", warn=True)

    merged = cohort.merge(split.drop_duplicates("patient_id"), on="patient_id", how="left")
    unlabeled = int(merged["split"].isna().sum())
    report.add("every cohort patient has a split label", unlabeled == 0, f"{unlabeled} of {len(cohort)} unlabeled" if unlabeled else f"{len(cohort)} cohort patients")
    merged = merged.dropna(subset=["split"])
    merged = merged[merged["split"].isin(SPLITS)]
    if merged.empty:
        return report

    overall_rate = float(merged["y"].mean())
    for group in SPLITS:
        part = merged[merged["split"] == group]
        share = len(part) / len(merged)
        report.add(f"{group}: share of the cohort is within {fraction_tolerance:.0%} of {fractions[group]:.0%}",
                   abs(share - fractions[group]) <= fraction_tolerance, f"{len(part)} patients, {share:.1%}")
        rate = float(part["y"].mean()) if len(part) else float("nan")
        report.add(f"{group}: event rate is within {rate_tolerance:.0%} of the overall {overall_rate:.1%}",
                   len(part) > 0 and abs(rate - overall_rate) <= rate_tolerance, f"{int(part['y'].sum())} events, {rate:.1%}")
        report.add(f"{group}: has both outcome classes", part["y"].nunique() == 2)

    if provided is not None:
        _check_against_provided(merged, provided, report)
    return report


def _check_against_provided(merged: pd.DataFrame, provided: pd.DataFrame, report: Report) -> None:
    provided = provided.astype(str).drop_duplicates("patient_id")
    both = merged.merge(provided.rename(columns={"split": "provided"}), on="patient_id", how="left")
    unlisted = int(both["provided"].isna().sum())
    report.add("provided split: every cohort patient is listed", unlisted == 0, f"{unlisted} cohort patients not listed" if unlisted else "")
    both = both.dropna(subset=["provided"])
    wrong_test = int(((both["split"] == "test") != (both["provided"] == "validation")).sum())
    report.add("design: the test patients are exactly the provided validation patients", wrong_test == 0,
               f"{wrong_test} cohort patients differ" if wrong_test else "")
    leaked = int(((both["provided"] == "train") & (both["split"] == "test")).sum())
    report.add("design: no provided train patient is in test", leaked == 0, f"{leaked} patients" if leaked else "")
    stray = int(((both["provided"] == "validation") & both["split"].isin(["train", "validation"])).sum())
    report.add("design: no provided validation patient is in train or validation", stray == 0, f"{stray} patients" if stray else "")
