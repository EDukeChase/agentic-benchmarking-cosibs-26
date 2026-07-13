"""Shared benchmarking utilities for EHRSHOT structured EHR models.

Assumptions explicitly stated:
- /data/EHR_SHOT/labels.csv contains one binary label per downstream task in
  columns other than patient_id.
- /data/EHR_SHOT/patient_data_all contains one CSV per patient with TIME and
  TEXT columns.
- We derive a simple structured sequence by splitting each TEXT field on commas
  and using the resulting event strings as tokens.
- A single multiclass-safe vocabulary is built from train/validation/test to keep
  the benchmark self-contained and deterministic.
"""

from __future__ import annotations

import csv
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from ehr_models import SequenceBatch, TimeSeriesBatch, CLMBRTBase, BEHRT, MedBERT, RETAIN, GRUD
from training import TrainConfig, BinaryClassificationTrainer


DATA_DIR = "/app/data/EHR_SHOT"
PATIENT_DIR = os.path.join(DATA_DIR, "patient_data_all")
LABELS_PATH = os.path.join(DATA_DIR, "labels.csv")


@dataclass
class PatientExample:
    patient_id: str
    tokens: List[str]
    label: float


def load_labels() -> Dict[str, Dict[str, int]]:
    with open(LABELS_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    out: Dict[str, Dict[str, int]] = {}
    for row in rows:
        pid = row["patient_id"]
        out[pid] = {k: int(row[k].strip().lower() == "true") for k in row if k != "patient_id"}
    return out


def split_patient_ids(label_name: str) -> Tuple[List[str], List[str], List[str]]:
    labels = load_labels()
    pids = sorted(labels.keys())
    n = len(pids)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    train = pids[:n_train]
    val = pids[n_train:n_train + n_val]
    test = pids[n_train + n_val:]
    if not test:
        test = val[-1:]
    return train, val, test


def read_patient_tokens(patient_id: str) -> List[str]:
    fp = os.path.join(PATIENT_DIR, f"patient_{patient_id}.csv")
    tokens: List[str] = []
    with open(fp, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("TEXT", "")
            parts = [p.strip() for p in text.split(",") if p.strip()]
            tokens.extend(parts)
    return tokens


def build_examples(label_name: str, split: str) -> List[PatientExample]:
    labels = load_labels()
    train, val, test = split_patient_ids(label_name)
    ids = {"train": train, "val": val, "test": test}[split]
    examples = []
    for pid in ids:
        if pid not in labels:
            continue
        fp = os.path.join(PATIENT_DIR, f"patient_{pid}.csv")
        if not os.path.exists(fp):
            continue
        examples.append(PatientExample(patient_id=pid, tokens=read_patient_tokens(pid), label=float(labels[pid][label_name])))
    return examples


def build_vocab(examples: Sequence[PatientExample]) -> Dict[str, int]:
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for ex in examples:
        for tok in ex.tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode_tokens(tokens: Sequence[str], vocab: Dict[str, int], max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = [vocab.get(tok, vocab["[UNK]"]) for tok in tokens[:max_len]]
    mask = [1] * len(ids)
    visits = list(range(len(ids)))
    ages = [min(i // 2, 127) for i in range(len(ids))]
    if len(ids) < max_len:
        pad = max_len - len(ids)
        ids += [vocab["[PAD]"]] * pad
        mask += [0] * pad
        visits += [0] * pad
        ages += [0] * pad
    return (torch.tensor(ids, dtype=torch.long), torch.tensor(visits, dtype=torch.long), torch.tensor(ages, dtype=torch.long), torch.tensor(mask, dtype=torch.bool))


class SequenceDataset(Dataset):
    def __init__(self, examples: Sequence[PatientExample], vocab: Dict[str, int], max_len: int):
        self.examples = list(examples)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        codes, visits, ages, mask = encode_tokens(ex.tokens, self.vocab, self.max_len)
        return SequenceBatch(codes=codes, visits=visits, ages=ages, mask=mask, labels=torch.tensor(ex.label, dtype=torch.float32))


def collate_identity(batch):
    return batch[0] if len(batch) == 1 else batch


def make_loader(dataset: Dataset, batch_size: int = 8, shuffle: bool = False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda xs: xs[0] if len(xs) == 1 else xs)


def batchify_sequence_batch(batch_list: List[SequenceBatch]) -> SequenceBatch:
    codes = torch.stack([b.codes for b in batch_list])
    visits = torch.stack([b.visits for b in batch_list])
    ages = torch.stack([b.ages for b in batch_list])
    mask = torch.stack([b.mask for b in batch_list])
    labels = torch.stack([b.labels for b in batch_list])
    return SequenceBatch(codes=codes, visits=visits, ages=ages, mask=mask, labels=labels)


class BinaryMetrics:
    @staticmethod
    def compute(y_true: torch.Tensor, y_score: torch.Tensor) -> Dict[str, float]:
        yt = y_true.detach().cpu().float()
        ys = y_score.detach().cpu().float()
        preds = (ys >= 0.5).float()
        acc = float((preds == yt).float().mean())
        tp = float(((preds == 1) & (yt == 1)).sum())
        fp = float(((preds == 1) & (yt == 0)).sum())
        fn = float(((preds == 0) & (yt == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
            auroc = float(roc_auc_score(yt.numpy(), ys.numpy()))
            auprc = float(average_precision_score(yt.numpy(), ys.numpy()))
            brier = float(brier_score_loss(yt.numpy(), ys.numpy()))
        except Exception:
            auroc = float("nan")
            auprc = float("nan")
            brier = float(((ys - yt) ** 2).mean())
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auroc": auroc, "auprc": auprc, "brier": brier}


def train_eval_model(model: nn.Module, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset, max_len: int, device: str = "cpu", epochs: int = 1, lr: float = 1e-3) -> Dict[str, float]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    def run_loader(ds, train: bool):
        ys, yhat = [], []
        loader = DataLoader(ds, batch_size=8, shuffle=train, collate_fn=batch_collate)
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch)
            logits = out["logits"].view(-1)
            labels = batch.labels.view(-1).float()
            if train:
                opt.zero_grad(); loss = loss_fn(logits, labels); loss.backward(); opt.step()
            ys.append(labels.detach().cpu())
            yhat.append(torch.sigmoid(logits.detach()).cpu())
        return torch.cat(ys), torch.cat(yhat)

    for _ in range(epochs):
        run_loader(train_ds, True)
    val_y, val_p = run_loader(val_ds, False)
    test_y, test_p = run_loader(test_ds, False)
    metrics = {"val_" + k: v for k, v in BinaryMetrics.compute(val_y, val_p).items()}
    metrics.update({"test_" + k: v for k, v in BinaryMetrics.compute(test_y, test_p).items()})
    return metrics


def batch_collate(batch_list):
    if isinstance(batch_list[0], SequenceBatch):
        return batchify_sequence_batch(batch_list)
    if isinstance(batch_list[0], TimeSeriesBatch):
        values = torch.stack([b.values for b in batch_list])
        masks = torch.stack([b.masks for b in batch_list])
        deltas = torch.stack([b.deltas for b in batch_list])
        labels = torch.stack([b.labels for b in batch_list])
        return TimeSeriesBatch(values=values, masks=masks, deltas=deltas, labels=labels)
    return batch_list


def move_batch(batch, device: str):
    if hasattr(batch, "__dataclass_fields__"):
        for f in batch.__dataclass_fields__:
            v = getattr(batch, f)
            if torch.is_tensor(v):
                setattr(batch, f, v.to(device))
    return batch


def benchmark(label_name: str, model_name: str) -> Dict[str, float]:
    train_ex = build_examples(label_name, "train")
    val_ex = build_examples(label_name, "val")
    test_ex = build_examples(label_name, "test")
    vocab = build_vocab(train_ex + val_ex + test_ex)
    max_len = max(max((len(ex.tokens) for ex in train_ex + val_ex + test_ex), default=1), 8)

    if model_name in {"Logistic Regression", "Random Forest", "XGBoost", "MLP"}:
        train_ds = SequenceDataset(train_ex, vocab, max_len)
        val_ds = SequenceDataset(val_ex, vocab, max_len)
        test_ds = SequenceDataset(test_ex, vocab, max_len)
        X_train, y_train = _logreg_features(train_ds, len(vocab))
        X_val, y_val = _logreg_features(val_ds, len(vocab))
        X_test, y_test = _logreg_features(test_ds, len(vocab))

        if model_name == "Logistic Regression":
            w = torch.zeros(X_train.shape[1])
            b = torch.tensor(0.0)
            Xtr = torch.tensor(X_train).float()
            ytr = torch.tensor(y_train).float()
            lr = 0.1
            for _ in range(200):
                logits = Xtr.matmul(w) + b
                probs = torch.sigmoid(logits)
                grad_w = Xtr.t().matmul(probs - ytr) / len(ytr)
                grad_b = (probs - ytr).mean()
                w -= lr * grad_w
                b -= lr * grad_b
            def predict(X):
                X = torch.tensor(X).float()
                return torch.sigmoid(X.matmul(w) + b)
        elif model_name == "Random Forest":
            # Lightweight stand-in when sklearn is unavailable: average of random stumps over selected features.
            import random
            random.seed(7)
            Xtr = torch.tensor(X_train).float()
            ytr = torch.tensor(y_train).float()
            idxs = [random.randrange(Xtr.shape[1]) for _ in range(100)]
            thresh = [float(torch.median(Xtr[:, i])) for i in idxs]
            pol = [1 if float((Xtr[:, i] > t).float().mean()) >= 0.5 else -1 for i, t in zip(idxs, thresh)]
            def predict(X):
                X = torch.tensor(X).float()
                votes = []
                for i, t, p in zip(idxs, thresh, pol):
                    votes.append(((X[:, i] > t).float() * p + (X[:, i] <= t).float() * (-p) + 1) / 2)
                return torch.stack(votes).mean(dim=0)
        elif model_name == "XGBoost":
            # Lightweight gradient boosting approximation using additive decision stumps.
            Xtr = torch.tensor(X_train).float()
            ytr = torch.tensor(y_train).float()
            scores = torch.zeros(len(ytr))
            stumps = []
            for _ in range(25):
                residual = ytr - torch.sigmoid(scores)
                best = None
                best_gain = -1e9
                for j in range(min(Xtr.shape[1], 100)):
                    thresh = torch.median(Xtr[:, j])
                    left = residual[Xtr[:, j] <= thresh].mean() if (Xtr[:, j] <= thresh).any() else torch.tensor(0.0)
                    right = residual[Xtr[:, j] > thresh].mean() if (Xtr[:, j] > thresh).any() else torch.tensor(0.0)
                    gain = float(left.abs() + right.abs())
                    if gain > best_gain:
                        best_gain = gain
                        best = (j, float(thresh), float(left), float(right))
                stumps.append(best)
                j, thresh, left, right = best
                scores += torch.where(Xtr[:, j] <= thresh, torch.tensor(left), torch.tensor(right))
            def predict(X):
                X = torch.tensor(X).float()
                s = torch.zeros(X.shape[0])
                for j, thresh, left, right in stumps:
                    s += torch.where(X[:, j] <= thresh, torch.tensor(left), torch.tensor(right))
                return torch.sigmoid(s)
        else:
            Xtr = torch.tensor(X_train).float()
            ytr = torch.tensor(y_train).float().view(-1, 1)
            model = nn.Sequential(nn.Linear(X_train.shape[1], 64), nn.ReLU(), nn.Linear(64, 1))
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.BCEWithLogitsLoss()
            for _ in range(25):
                opt.zero_grad(); loss = loss_fn(model(Xtr), ytr); loss.backward(); opt.step()
            def predict(X):
                with torch.no_grad():
                    return torch.sigmoid(model(torch.tensor(X).float()).view(-1))

        val_p = predict(X_val)
        test_p = predict(X_test)
        metrics = {"val_" + k: v for k, v in BinaryMetrics.compute(torch.tensor(y_val), val_p).items()}
        metrics.update({"test_" + k: v for k, v in BinaryMetrics.compute(torch.tensor(y_test), test_p).items()})
        return metrics

    seq_train = SequenceDataset(train_ex, vocab, max_len)
    seq_val = SequenceDataset(val_ex, vocab, max_len)
    seq_test = SequenceDataset(test_ex, vocab, max_len)
    device = "cpu"
    # Keep neural benchmarks intentionally small so the benchmark suite runs quickly.
    if model_name == "CLMBR-T-base":
        model = CLMBRTBase(vocab_size=len(vocab), d_model=16, nhead=2, num_layers=1, max_len=max_len)
    elif model_name == "BEHRT":
        model = BEHRT(vocab_size=len(vocab), d_model=16, nhead=2, num_layers=1, max_visits=max_len)
    elif model_name == "MedBERT":
        model = MedBERT(vocab_size=len(vocab), d_model=16, nhead=2, num_layers=1, max_visits=max_len)
    elif model_name == "RETAIN":
        model = RETAIN(vocab_size=len(vocab), embed_dim=8, hidden_dim=8)
    elif model_name == "GRUD":
        model = GRUD(input_dim=max(2, len(vocab)), hidden_dim=8)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name == "GRUD":
        train_ts = TimeSeriesDataset(train_ex, vocab, max_len)
        val_ts = TimeSeriesDataset(val_ex, vocab, max_len)
        test_ts = TimeSeriesDataset(test_ex, vocab, max_len)
        trainer = BinaryClassificationTrainer(model, TrainConfig(epochs=1, device=device, lr=1e-3))
        trainer.fit(DataLoader(train_ts, batch_size=16, shuffle=True, collate_fn=batch_collate), DataLoader(val_ts, batch_size=16, collate_fn=batch_collate))
        def predict(ds):
            ys, ps = [], []
            loader = DataLoader(ds, batch_size=16, collate_fn=batch_collate)
            model.eval()
            for batch in loader:
                batch = move_batch(batch, device)
                with torch.no_grad():
                    out = model(batch)
                    ys.append(batch.labels.cpu())
                    ps.append(torch.sigmoid(out["logits"].view(-1)).cpu())
            return torch.cat(ys), torch.cat(ps)
        val_y, val_p = predict(val_ts)
        test_y, test_p = predict(test_ts)
    else:
        trainer = BinaryClassificationTrainer(model, TrainConfig(epochs=1, device=device, lr=1e-3))
        trainer.fit(DataLoader(seq_train, batch_size=16, shuffle=True, collate_fn=batch_collate), DataLoader(seq_val, batch_size=16, collate_fn=batch_collate))
        def predict(ds):
            ys, ps = [], []
            loader = DataLoader(ds, batch_size=16, collate_fn=batch_collate)
            model.eval()
            for batch in loader:
                batch = move_batch(batch, device)
                with torch.no_grad():
                    out = model(batch)
                    ys.append(batch.labels.cpu())
                    ps.append(torch.sigmoid(out["logits"].view(-1)).cpu())
            return torch.cat(ys), torch.cat(ps)
        val_y, val_p = predict(seq_val)
        test_y, test_p = predict(seq_test)

    metrics = {"val_" + k: v for k, v in BinaryMetrics.compute(val_y, val_p).items()}
    metrics.update({"test_" + k: v for k, v in BinaryMetrics.compute(test_y, test_p).items()})
    return metrics


def _logreg_features(ds: SequenceDataset, vocab_size: int):
    xs, ys = [], []
    for i in range(len(ds)):
        b = ds[i]
        feat = torch.bincount(b.codes[b.mask], minlength=vocab_size).float()
        xs.append(feat)
        ys.append(b.labels)
    X = torch.stack(xs)
    return X.numpy(), torch.stack(ys).numpy()
