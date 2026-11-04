"""Cohort and outcome definition for the incident ischemic heart disease task.

Builds the analysis cohort straight from the raw tables (patients.csv and
conditions.csv), one row per patient, so the cohort is defined in code the
agents cannot reinterpret. Patient IDs are strings (UUIDs) throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CohortDefinition:
    """The prediction task exactly as given to the agents."""

    index_date: str = "2015-12-31"
    end_date: str = "2025-12-31"
    min_age: int = 40
    max_age: int = 79
    outcome_code: str = "414545008"  # ischemic heart disease (SNOMED)
    exclusion_codes: tuple[str, ...] = ("414545008", "22298006")  # IHD, myocardial infarction


def age_on(birthdates: pd.Series, on: pd.Timestamp) -> pd.Series:
    """Whole years completed on the given date."""
    had_birthday = (birthdates.dt.month < on.month) | (
        (birthdates.dt.month == on.month) & (birthdates.dt.day <= on.day)
    )
    return on.year - birthdates.dt.year - (~had_birthday).astype(int)


def build_cohort(data_dir: str | Path, definition: CohortDefinition = CohortDefinition()) -> pd.DataFrame:
    """Return one row per eligible patient with columns patient_id and y.

    Eligible: alive on the index date (no death date, or a death date after
    it), aged min_age to max_age on the index date, and no condition from
    exclusion_codes starting on or before the index date. y is 1 if the first
    outcome_code condition starts after the index date and on or before
    end_date, else 0 (so a death inside the window without the condition is 0).
    """
    data_dir = Path(data_dir)
    index = pd.Timestamp(definition.index_date)
    end = pd.Timestamp(definition.end_date)

    patients = pd.read_csv(data_dir / "patients.csv", usecols=["Id", "BIRTHDATE", "DEATHDATE"])
    # explicit conversion: a folder where nobody has died has an all-empty DEATHDATE column
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
    patients["DEATHDATE"] = pd.to_datetime(patients["DEATHDATE"])
    conditions = pd.read_csv(
        data_dir / "conditions.csv",
        usecols=["START", "PATIENT", "CODE"],
        dtype={"CODE": str},
        parse_dates=["START"],
    )

    alive = patients["DEATHDATE"].isna() | (patients["DEATHDATE"] > index)
    in_age_range = age_on(patients["BIRTHDATE"], index).between(definition.min_age, definition.max_age)
    cohort = patients.loc[alive & in_age_range, ["Id"]].rename(columns={"Id": "patient_id"})

    prior = conditions[conditions["CODE"].isin(definition.exclusion_codes) & (conditions["START"] <= index)]
    cohort = cohort[~cohort["patient_id"].isin(set(prior["PATIENT"]))]

    outcome = conditions[conditions["CODE"] == definition.outcome_code]
    first_outcome = outcome.groupby("PATIENT")["START"].min()
    # explicit conversion: with no outcome rows at all the mapped column has no date type
    first = pd.to_datetime(cohort["patient_id"].map(first_outcome))
    cohort["y"] = ((first > index) & (first <= end)).astype(int)

    return cohort.sort_values("patient_id").reset_index(drop=True)
