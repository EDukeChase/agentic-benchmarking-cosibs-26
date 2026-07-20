"""Aggregate experiment outputs and run the SAP's one-factor comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_METRICS = ("auroc", "f1", "recall", "precision", "brier")
SECONDARY_METRICS = ("accuracy", "runtime_seconds", "total_tokens")
METRICS = PRIMARY_METRICS + SECONDARY_METRICS


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    covariance = np.cov(x, y, ddof=1)[0, 1]
    denominator = np.var(x, ddof=1) + np.var(y, ddof=1) + (np.mean(x) - np.mean(y)) ** 2
    return float(2 * covariance / denominator) if denominator else 1.0


def add_descriptive_stats(report: dict, data: pd.DataFrame, metric: str) -> None:
    group_columns = ["condition_id"]
    if "model_name" in data.columns:
        group_columns.append("model_name")

    summary = data.groupby(group_columns)[metric].agg(
        ["count", "mean", "std", "median", "min", "max"]
    )
    records = summary.reset_index().to_dict(orient="records")
    report["descriptive"][metric] = records


def add_condition_tests(report: dict, data: pd.DataFrame, metric: str) -> None:
    if data["condition_id"].nunique() < 2:
        return

    if "model_name" in data.columns:
        model_groups = data.groupby("model_name")
    else:
        model_groups = [("all", data)]

    for model_name, model_data in model_groups:
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
        report["anova"][key] = {
            "f": float(f_value),
            "p": float(p_value),
            "groups": labels,
        }

        tukey = stats.tukey_hsd(*samples)
        report["tukey_hsd"][key] = {
            "groups": labels,
            "statistic": np.asarray(tukey.statistic).tolist(),
            "pvalue": np.asarray(tukey.pvalue).tolist(),
        }


def add_self_consistency(report: dict, data: pd.DataFrame, metric: str) -> None:
    required = {"condition_id", "replicate", "model_name"}
    if not required.issubset(data.columns):
        return

    for condition_name, condition_data in data.groupby("condition_id"):
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
    frame = pd.read_csv(results_csv)
    report: dict = {"descriptive": {}, "anova": {}, "tukey_hsd": {}, "self_consistency_ccc": {}}
    for metric in METRICS:
        if metric not in frame:
            continue

        clean = frame.dropna(subset=[metric])
        if clean.empty:
            continue

        add_descriptive_stats(report, clean, metric)
        add_condition_tests(report, clean, metric)
        add_self_consistency(report, clean, metric)
    Path(output_json).write_text(json.dumps(report, indent=2, allow_nan=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv")
    parser.add_argument("output_json")
    args = parser.parse_args()
    analyze(args.results_csv, args.output_json)
