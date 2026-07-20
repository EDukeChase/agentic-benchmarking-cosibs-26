from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from linear_regression.model import build_model

BASE_DIR = Path("/app/generated_code/2f8a07a5")
DATA_DIR = Path("/app/data")
EHR_DIR = DATA_DIR / "EHR_SHOT"
LABELS_PATH = EHR_DIR / "labels.csv"
PATIENT_DIR = EHR_DIR / "patient_data_all"

TARGET_COLUMNS = [
    "new_acutemi",
    "new_celiac",
    "new_hyperlipidemia",
    "new_hypertension",
    "new_lupus",
    "new_pancan",
]


def _read_labels() -> Tuple[List[str], np.ndarray]:
    with LABELS_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    patient_ids = [row["patient_id"] for row in rows]
    y = np.array([[1 if row[col].strip().lower() == "true" else 0 for col in TARGET_COLUMNS] for row in rows], dtype=int)
    return patient_ids, y


def _load_patient_text(patient_id: str) -> str:
    path = PATIENT_DIR / f"patient_{patient_id}.csv"
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        pieces = []
        for row in reader:
            pieces.append(" ".join(str(v) for v in row.values() if v is not None))
    return " ".join(pieces)


def _build_features(patient_ids: List[str], targets: np.ndarray):
    texts = []
    kept_targets = []
    kept_ids = []
    for pid, y_row in zip(patient_ids, targets):
        path = PATIENT_DIR / f"patient_{pid}.csv"
        if not path.exists():
            continue
        texts.append(_load_patient_text(pid))
        kept_targets.append(y_row)
        kept_ids.append(pid)
    vectorizer = TfidfVectorizer(min_df=1, max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X, np.asarray(kept_targets, dtype=int), kept_ids


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
    patient_ids, y = _read_labels()
    X, y, patient_ids = _build_features(patient_ids, y)
    results: Dict[str, Dict[str, float]] = {}
    for i, target in enumerate(TARGET_COLUMNS):
        y_target = y[:, i]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_target, test_size=0.25, random_state=42, stratify=y_target if len(np.unique(y_target)) > 1 else None
        )
        model = build_model()
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        results[target] = _safe_binary_metrics(y_test, y_score, y_pred)
    return results


if __name__ == "__main__":
    metrics = run_benchmark()
    print(json.dumps(metrics, indent=2, sort_keys=True))
