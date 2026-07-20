"""Aggregate experiment outputs and run the SAP's one-factor comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# Added Tertiary Metrics
PRIMARY_METRICS = ("auroc", "f1", "recall", "precision", "brier")
SECONDARY_METRICS = ("accuracy", "runtime_seconds", "total_tokens")
TERTIARY_METRICS = ("uncertainty",)
METRICS = PRIMARY_METRICS + SECONDARY_METRICS + TERTIARY_METRICS


# Lin's concordance correlation coefficient measures how closely two sets of
# numbers agree. A value near 1 means the two repetitions are very similar.
# This can measure cross-agent consistency for uncertainty
def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    covariance = np.cov(x, y, ddof=1)[0, 1]
    denominator = np.var(x, ddof=1) + np.var(y, ddof=1) + (np.mean(x) - np.mean(y)) ** 2
    return float(2 * covariance / denominator) if denominator else 1.0


# Add easy-to-read summaries such as the mean, standard deviation, and range for
# every experimental condition and model.
def add_descriptive_stats(report: dict, data: pd.DataFrame, metric: str) -> None:
    group_columns = ["condition_id"]
    if "model_name" in data.columns:
        group_columns.append("model_name")

    summary = data.groupby(group_columns)[metric].agg(
        ["count", "mean", "std", "median", "min", "max"]
    )
    records = summary.reset_index().to_dict(orient="records")
    report["descriptive"][metric] = records


# Compare experimental conditions for one metric. ANOVA first checks for an
# overall difference, and Tukey HSD shows which pairs of conditions differ.
# The ANOVA is for disease prediction accuracy for each model
def add_condition_tests(report: dict, data: pd.DataFrame, metric: str) -> None:
    if data["condition_id"].nunique() < 2:
        return

    if "model_name" in data.columns:
        model_groups = data.groupby("model_name")
    else:
        model_groups = [("all", data)]

    for model_name, model_data in model_groups:
        # Each item in samples contains all repetitions for one condition.
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

# Testing for uncertainty across different parameters
# Note that the metric should only be the uncertainty metric here
def add_factor_tests(report: dict, data: pd.DataFrame, metric: str) -> None:
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


# Compare every pair of repetitions within a condition. The metric values across
# models form the two lists used by Lin's concordance calculation.
def add_self_consistency(report: dict, data: pd.DataFrame, metric: str) -> None:
    required = {"condition_id", "replicate", "model_name"}
    if not required.issubset(data.columns):
        return

    for condition_name, condition_data in data.groupby("condition_id"):
        # Rows are models and columns are repetition numbers.
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


# Read the combined experiment CSV, run each analysis, and save one JSON report.
def analyze(results_csv: str | Path, output_json: str | Path) -> dict:
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
    # This block allows the file to be run directly from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv")
    parser.add_argument("output_json")
    args = parser.parse_args()
    analyze(args.results_csv, args.output_json)
