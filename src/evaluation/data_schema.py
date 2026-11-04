"""Schema-only description of the benchmark task's data.

Column names and dtypes only, inferred from a few real rows read locally but
never included in the output themselves. This is meant to be sent to the
literature agent (an external API) to ground its model search in the real
data shape, so it must never carry an actual patient ID, outcome value, or
event value.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.deterministic import _real_path
from src.settings.config import BenchmarkTaskConfig


def describe_data_schema(task: BenchmarkTaskConfig) -> str:
    """Return a short, values-free description of the task's data shape."""
    root = _real_path(task.data_root)

    # A few rows are read locally so pandas can infer real dtypes, but only
    # column names and dtypes are ever placed in the returned text.
    labels = pd.read_csv(root / "labels.csv", nrows=5)
    label_columns = ", ".join(f"{col} ({labels[col].dtype})" for col in labels.columns)

    patient_dir = root / "patient_data_all"
    sample_file = next(patient_dir.glob("patient_*.csv"), None)
    if sample_file is None:
        event_columns = "(no sample patient event file found)"
    else:
        sample = pd.read_csv(sample_file, nrows=5)
        event_columns = ", ".join(f"{col} ({sample[col].dtype})" for col in sample.columns)

    return (
        f"Each patient, identified by '{task.patient_id_column}', "
        f"has a binary outcome to predict: '{task.outcome}' (0 or 1).\n"
        f"labels.csv columns and types: {label_columns}\n"
        "Each patient also has a longitudinal event file, one row per clinical event, "
        f"with these columns and types: {event_columns}\n"
        "This is a schema description only; no actual patient IDs, outcome values, or "
        "event values are included."
    )
