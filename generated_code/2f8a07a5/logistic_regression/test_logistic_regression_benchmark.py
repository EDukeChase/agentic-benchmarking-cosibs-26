from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from logistic_regression.model import build_model

BASE_DIR = Path("/app/generated_code/2f8a07a5")
DATA_DIR = Path("/app/data")
MIMIC_DIR = DATA_DIR / "MIMIC_tabular"
LABELS_PATH = MIMIC_DIR / "diagnosis.csv"
INPUT_DIR = MIMIC_DIR / "inputs"

POSITIVE_LABEL = "pneumonia"


def _read_labels() -> Tuple[List[str], np.ndarray]:
    with LABELS_PATH.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    files = [row["file"] for row in rows]
    y = np.array([1 if POSITIVE_LABEL in [d.strip().lower() for d in row["diagnoses"].split("|")] else 0 for row in rows], dtype=int)
    return files, y


def _load_input_text(filename: str) -> str:
    path = INPUT_DIR / filename
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        pieces = []
        for row in reader:
            pieces.append(" ".join(str(v) for v in row.values() if v is not None))
    return " ".join(pieces)


def _build_features(files: List[str], targets: np.ndarray):
    texts = []
    kept_targets = []
    kept_files = []
    for fname, y_val in zip(files, targets):
        path = INPUT_DIR / fname
        if not path.exists():
            continue
        texts.append(_load_input_text(fname))
        kept_targets.append(y_val)
        kept_files.append(fname)
    vectorizer = TfidfVectorizer(min_df=1, max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X, np.asarray(kept_targets, dtype=int), kept_files


def _safe_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_score),
    }
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics


def run_benchmark() -> Dict[str, Dict[str, float]]:
    files, y = _read_labels()
    X, y, files = _build_features(files, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    model = build_model(max_iter=500)
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    return {POSITIVE_LABEL: _safe_binary_metrics(y_test, y_score, y_pred)}


if __name__ == "__main__":
    metrics = run_benchmark()
    print(json.dumps(metrics, indent=2, sort_keys=True))
