"""Three-way patient split for the incident ischemic heart disease study.

The provided split has two groups (train, validation). The study reports on
the provided validation patients, renamed test, and carves a new validation
group out of the provided train patients, stratified on the outcome.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_GROUPS = ("train", "validation", "test")


def make_three_way_split(
    provided: pd.DataFrame,
    cohort: pd.DataFrame,
    validation_fraction_of_train: float = 0.25,
    seed: int = 5960,
) -> pd.DataFrame:
    """Return patient_id and split (train, validation or test) for every provided patient.

    Provided validation patients become test. Provided train patients who are
    in the cohort are divided into train and validation, stratified on y;
    provided train patients outside the cohort stay train, since they are
    never used.
    """
    unknown = set(provided["split"]) - {"train", "validation"}
    if unknown:
        raise ValueError(f"Unexpected split labels in the provided split: {sorted(unknown)}")
    missing = set(cohort["patient_id"]) - set(provided["patient_id"])
    if missing:
        raise ValueError(f"{len(missing)} cohort patients have no provided split label")

    result = provided[["patient_id", "split"]].copy()
    result["split"] = result["split"].replace({"validation": "test"})

    train_cohort = cohort[cohort["patient_id"].isin(set(provided.loc[provided["split"] == "train", "patient_id"]))]
    train_cohort = train_cohort.sort_values("patient_id")
    _, validation_ids = train_test_split(
        train_cohort["patient_id"],
        test_size=validation_fraction_of_train,
        stratify=train_cohort["y"],
        random_state=seed,
    )
    result.loc[result["patient_id"].isin(set(validation_ids)), "split"] = "validation"
    return result.sort_values("patient_id").reset_index(drop=True)
