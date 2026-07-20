"""Benchmark for binary celiac disease diagnosis using logistic regression.

Assumptions
-----------
- Each patient has one CSV file in ``/app/data/EHR_SHOT/patient_data_all`` named
  ``patient_<patient_id>.csv``.
- ``/app/data/EHR_SHOT/labels.csv`` contains one row per patient and a binary
  outcome column named ``new_celiac``.
- The task is binary classification with label encoding 0/1.

Split protocol
--------------
We use a reproducible stratified train/validation/test split with ratios
60/20/20 and random seed 42.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path("/app/data/EHR_SHOT")
PATIENT_DIR = BASE_DIR / "patient_data_all"
LABELS_PATH = BASE_DIR / "labels.csv"
RESULTS_PATH = Path("/app/generated_code/benchmark_results.json")
RANDOM_SEED = 42


def _find_id_and_label_columns(labels: pd.DataFrame) -> Tuple[str, str]:
    id_candidates = [c for c in labels.columns if "patient" in c.lower() or c.lower() == "id"]
    if not id_candidates:
        raise ValueError(f"Could not infer patient id column from labels columns: {list(labels.columns)}")
    id_col = id_candidates[0]
    label_candidates = [c for c in labels.columns if c != id_col and "celiac" in c.lower()]
    if not label_candidates:
        raise ValueError(f"Could not infer celiac label column from labels columns: {list(labels.columns)}")
    label_col = label_candidates[0]
    return id_col, label_col


def load_dataset() -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    labels = pd.read_csv(LABELS_PATH)
    id_col, label_col = _find_id_and_label_columns(labels)
    labels = labels[[id_col, label_col]].copy()
    labels[id_col] = labels[id_col].astype(str)
    labels[label_col] = labels[label_col].astype(int)

    records = []
    y = []
    patient_ids = []
    for _, row in labels.iterrows():
        pid = row[id_col]
        path = PATIENT_DIR / f"patient_{pid}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "TEXT" not in df.columns:
            raise ValueError(f"Expected TEXT column in {path}")
        text = " ".join(df["TEXT"].astype(str).tolist())
        records.append(text)
        y.append(int(row[label_col]))
        patient_ids.append(pid)
    return records, np.asarray(y, dtype=int), pd.DataFrame({"patient_id": patient_ids, "text": records, "label": y})


def make_splits(X: List[str], y: np.ndarray):
    # 60/20/20 stratified split with fixed seed for reproducibility.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=y_tmp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def featurize(train_texts, val_texts, test_texts):
    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test


def evaluate(y_true, y_pred, y_proba):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_proba))
        out["auprc"] = float(average_precision_score(y_true, y_proba))
        out["brier"] = float(brier_score_loss(y_true, y_proba))
    else:
        out["auroc"] = None
        out["auprc"] = None
        out["brier"] = None
    return out


def main():
    X, y, df = load_dataset()
    assert len(X) == len(y) and len(X) > 0, "No usable patient records were loaded"
    class_counts = Counter(y.tolist())
    assert len(class_counts) == 2, f"Expected binary labels, got: {class_counts}"

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
    X_train_f, X_val_f, X_test_f = featurize(X_train, X_val, X_test)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train_f, y_train)

    def predict(X):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return pred, proba

    val_pred, val_proba = predict(X_val_f)
    test_pred, test_proba = predict(X_test_f)
    val_metrics = evaluate(y_val, val_pred, val_proba)
    test_metrics = evaluate(y_test, test_pred, test_proba)

    print("Dataset size:", len(df))
    print("Label counts:", dict(class_counts))
    print("Validation metrics:", json.dumps(val_metrics, indent=2, sort_keys=True))
    print("Test metrics:", json.dumps(test_metrics, indent=2, sort_keys=True))

    assert 0.0 <= test_metrics["accuracy"] <= 1.0
    assert 0.0 <= test_metrics["f1"] <= 1.0

    results = {"logreg": {"validation": val_metrics, "test": test_metrics}}
    RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True))
    return results


if __name__ == "__main__":
    main()
