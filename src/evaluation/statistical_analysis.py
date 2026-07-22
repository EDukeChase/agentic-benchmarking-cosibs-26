"""Summarize completed benchmark runs and compare experimental conditions.

The experiment runner stores one CSV row per model evaluated in each pipeline
replicate. This module turns those raw rows into a JSON report containing:

1. descriptive statistics, which explain the center and spread of each metric;
2. one-way ANOVA tests, which ask whether any condition or factor mean differs;
3. Tukey HSD follow-up tests, which compare every pair after an ANOVA; and
4. Lin's concordance coefficients, which measure agreement across replicates.

This file reports statistical results but does not decide whether a result is
clinically meaningful. In particular, a small p-value describes evidence against
equal means; it does not describe the size or importance of the difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# Metrics are grouped by their role in the analysis plan. Clinical predictive
# performance is primary, while resource-use measurements are secondary.
# Uncertainty is handled separately because it is compared across agent factors
# instead of across model/disease conditions.
PRIMARY_METRICS = ("auroc", "f1", "recall", "precision", "brier")
SECONDARY_METRICS = ("accuracy", "runtime_seconds", "total_tokens")
TERTIARY_METRICS = ("uncertainty",)
METRICS = PRIMARY_METRICS + SECONDARY_METRICS + TERTIARY_METRICS


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Return Lin's concordance correlation coefficient for two matched arrays.

    Unlike ordinary correlation, concordance checks both whether values move
    together and whether they lie near the 45-degree equality line. Two runs can
    therefore be highly correlated but have poor concordance if one run is
    consistently larger than the other. Values near 1 indicate strong agreement,
    values near 0 indicate little agreement, and negative values indicate
    systematic disagreement.

    At least two matched observations are needed to estimate sample covariance.
    A zero denominator means both arrays are identical constants, so agreement is
    defined as perfect rather than allowing a division-by-zero error.
    """
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    covariance = np.cov(x, y, ddof=1)[0, 1]
    denominator = np.var(x, ddof=1) + np.var(y, ddof=1) + (np.mean(x) - np.mean(y)) ** 2
    return float(2 * covariance / denominator) if denominator else 1.0


def add_descriptive_stats(report: dict, data: pd.DataFrame, metric: str) -> None:
    """Add basic summaries for one metric to the in-progress report.

    Model-level data are summarized separately for every condition/model pair.
    Run-level data without a model_name column are summarized by condition only.
    Count is included so readers can see how many observations contributed to
    each mean, standard deviation, median, minimum, and maximum.
    """
    group_columns = ["condition_id"]
    if "model_name" in data.columns:
        group_columns.append("model_name")

    summary = data.groupby(group_columns)[metric].agg(
        ["count", "mean", "std", "median", "min", "max"]
    )
    records = summary.reset_index().to_dict(orient="records")
    report["descriptive"][metric] = records


def add_condition_tests(report: dict, data: pd.DataFrame, metric: str) -> None:
    """Compare condition means for one metric, separately for each model.

    A one-way ANOVA tests the overall null hypothesis that all included condition
    means are equal. If conditions differ, Tukey's honestly significant
    difference procedure supplies multiplicity-adjusted pairwise comparisons.
    Groups with fewer than two observations are excluded because a within-group
    variance cannot be estimated from a single value.
    """
    # With only one condition there is nothing to compare.
    if data["condition_id"].nunique() < 2:
        return

    if "model_name" in data.columns:
        model_groups = data.groupby("model_name")
    else:
        model_groups = [("all", data)]

    for model_name, model_data in model_groups:
        # SciPy expects one numeric array per condition. ``labels`` preserves the
        # corresponding condition names so matrix positions remain interpretable
        # when the numeric output is serialized to JSON.
        samples = []
        labels = []
        for condition_name, condition_data in model_data.groupby("condition_id"):
            if len(condition_data) >= 2:
                samples.append(condition_data[metric].to_numpy())
                labels.append(str(condition_name))

        if len(samples) < 2:
            continue

        f_value, p_value = stats.f_oneway(*samples)
        key = f"{model_name}/{metric}"
        report["anova_model-disease"][key] = {
            "f": float(f_value),
            "p": float(p_value),
            "groups": labels,
        }

        tukey = stats.tukey_hsd(*samples)
        report["tukey_hsd_model-disease"][key] = {
            "groups": labels,
            "statistic": np.asarray(tukey.statistic).tolist(),
            "pvalue": np.asarray(tukey.pvalue).tolist(),
        }

def add_factor_tests(report: dict, data: pd.DataFrame, metric: str) -> None:
    """Compare uncertainty across factor levels within each agent stage.

    Uncertainty experiments describe each observation using an agent stage, a
    factor name (the setting being varied), and a factor level (one possible
    value). The analysis keeps stages and factors separate, then performs the same
    ANOVA/Tukey sequence used for condition comparisons. Missing metadata causes
    this optional analysis to be skipped rather than making the full report fail.
    """
    required = {"agent_stage", "factor_name", "factor_level"}

    if not required.issubset(data.columns):
        return

    for stage, stage_data in data.groupby("agent_stage"):

        for factor, factor_data in stage_data.groupby("factor_name"):

            samples = []
            labels = []

            for level, level_data in factor_data.groupby("factor_level"):

                if len(level_data) >= 2:
                    samples.append(level_data[metric].to_numpy())
                    labels.append(str(level))

            if len(samples) < 2:
                continue

            f_value, p_value = stats.f_oneway(*samples)

            key = f"{stage}/{factor}/{metric}"

            report["anova_agent-uncertainty"][key] = {
                "f": float(f_value),
                "p": float(p_value),
                "levels": labels,
            }

            tukey = stats.tukey_hsd(*samples)

            report["tukey_hsd_agent-uncertainty"][key] = {
                "levels": labels,
                "statistic": np.asarray(tukey.statistic).tolist(),
                "pvalue": np.asarray(tukey.pvalue).tolist(),
            }


def add_self_consistency(report: dict, data: pd.DataFrame, metric: str) -> None:
    """Measure how consistently a condition ranks/scores models across replicates.

    For each pair of replicate numbers, models provide the matched observations:
    a model's value in the first replicate is paired with that same model's value
    in the second. Models missing either value are removed from that comparison.
    At least two matched models are required for concordance.
    """
    required = {"condition_id", "replicate", "model_name"}
    if not required.issubset(data.columns):
        return

    for condition_name, condition_data in data.groupby("condition_id"):
        # Pivoting aligns the same model across repetitions. Rows become models,
        # columns become replicate numbers, and cells contain the chosen metric.
        table = condition_data.pivot_table(
            index="model_name", columns="replicate", values=metric
        )
        replicate_numbers = list(table.columns)

        for index, first in enumerate(replicate_numbers):
            for second in replicate_numbers[index + 1:]:
                pair = table[[first, second]].dropna()
                if len(pair) < 2:
                    continue
                key = f"{condition_name}/replicate_{first}_vs_{second}/{metric}"
                report["self_consistency_ccc"][key] = lins_ccc(
                    pair[first].to_numpy(), pair[second].to_numpy()
                )


def analyze(results_csv: str | Path, output_json: str | Path) -> dict:
    """Analyze a combined results CSV and write the complete report as JSON.

    Metrics absent from the input or containing no usable numeric observations
    are skipped. Ordinary performance/resource metrics use condition comparisons;
    uncertainty uses factor comparisons. All eligible metrics also receive
    descriptive and replicate-agreement summaries.
    """
    frame = pd.read_csv(results_csv)
    report: dict = {"descriptive": {}, "anova_model-disease": {}, "tukey_hsd_model-disease": {},"anova_agent-uncertainty": {},  "tukey_hsd_agent-uncertainty": {}, "self_consistency_ccc": {}}
    for metric in METRICS:
        if metric not in frame:
            continue

        clean = frame.dropna(subset=[metric])
        if clean.empty:
            continue

        add_descriptive_stats(report, clean, metric)
        if metric != "uncertainty":
            add_condition_tests(report, clean, metric)
        if metric == "uncertainty":
            add_factor_tests(report, clean, metric)
        add_self_consistency(report, clean, metric)
    Path(output_json).write_text(json.dumps(report, indent=2, allow_nan=True))
    return report


if __name__ == "__main__":
    # Running this module directly provides a small command-line interface. The
    # first positional argument is the combined CSV to read and the second is the
    # JSON report path to create.
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv")
    parser.add_argument("output_json")
    args = parser.parse_args()
    analyze(args.results_csv, args.output_json)
