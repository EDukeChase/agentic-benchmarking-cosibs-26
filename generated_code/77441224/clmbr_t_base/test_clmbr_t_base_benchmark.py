from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

from model import CLMBRConfig, CLMBRTBase


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "EHR_SHOT"
LABELS_PATH = DATA_DIR / "labels.csv"
PATIENT_DIR = DATA_DIR / "patient_data_all"


def load_labels() -> Tuple[List[int], List[str], Dict[int, Dict[str, bool]]]:
    with open(LABELS_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        label_cols = [c for c in reader.fieldnames if c != "patient_id"]
        rows = []
        by_id = {}
        for row in reader:
            pid = int(row["patient_id"])
            parsed = {k: row[k].strip().lower() == "true" for k in label_cols}
            by_id[pid] = parsed
            rows.append(pid)
    return rows, label_cols, by_id


def read_patient_sequence(patient_id: int) -> List[int]:
    path = PATIENT_DIR / f"patient_{patient_id}.csv"
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        seq = []
        for i, row in enumerate(reader, start=1):
            if not row:
                continue
            code = row[0].strip()
            if not code:
                continue
            token = abs(hash(code)) % 50000 + 3
            seq.append(token)
    return seq


def build_batch(seqs: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = min(max(len(s) for s in seqs), 512)
    input_ids = torch.zeros((len(seqs), max_len), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(seqs):
        seq = seq[:max_len]
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1
    return input_ids, attention_mask


def main() -> Dict[str, float]:
    patient_ids, label_cols, labels_by_id = load_labels()
    target_col = "new_hypertension"
    y = np.array([int(labels_by_id[pid][target_col]) for pid in patient_ids], dtype=int)
    seqs = []
    kept_y = []
    kept_ids = []
    for pid, yi in zip(patient_ids, y):
        path = PATIENT_DIR / f"patient_{pid}.csv"
        if not path.exists():
            continue
        seq = read_patient_sequence(pid)
        if len(seq) == 0:
            seq = [1, 2]
        seqs.append(seq)
        kept_y.append(yi)
        kept_ids.append(pid)
    y = np.array(kept_y, dtype=int)
    patient_ids = kept_ids

    train_ids, test_ids, y_train, y_test, seq_train, seq_test = train_test_split(
        patient_ids, y, seqs, test_size=0.3, random_state=42, stratify=y
    )

    config = CLMBRConfig(vocab_size=50003, hidden_size=64, num_layers=2, num_heads=4, attention_window=16, max_position_embeddings=512)
    model = CLMBRTBase(config)
    model.eval()

    def embed_batch(batch_seqs: List[List[int]]) -> np.ndarray:
        embs = []
        with torch.no_grad():
            for start in range(0, len(batch_seqs), 8):
                chunk = batch_seqs[start:start+8]
                input_ids, attention_mask = build_batch(chunk)
                reps = model.encode(input_ids, attention_mask=attention_mask, pooling="last_valid")
                reps = torch.nan_to_num(reps, nan=0.0, posinf=0.0, neginf=0.0)
                embs.append(reps.cpu().numpy())
        return np.concatenate(embs, axis=0)

    X_train = embed_batch(seq_train)
    X_test = embed_batch(seq_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "auroc": float(roc_auc_score(y_test, probs)),
        "brier": float(brier_score_loss(y_test, probs)),
    }
    return metrics


if __name__ == "__main__":
    metrics = main()
    print(json.dumps(metrics, indent=2, sort_keys=True))
