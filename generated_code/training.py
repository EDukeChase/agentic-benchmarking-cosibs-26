"""Training and evaluation utilities for EHR benchmark models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-2
    epochs: int = 10
    grad_clip_norm: float = 1.0
    device: str = "cpu"


class BinaryClassificationTrainer:
    def __init__(self, model: nn.Module, config: TrainConfig):
        self.model = model.to(config.device)
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def _move_batch(self, batch):
        if hasattr(batch, "__dataclass_fields__"):
            for field in batch.__dataclass_fields__:
                val = getattr(batch, field)
                if val is not None and torch.is_tensor(val):
                    setattr(batch, field, val.to(self.config.device))
        elif isinstance(batch, dict):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.config.device)
        else:
            for attr in ("codes", "visits", "ages", "positions", "mask", "labels", "values", "masks", "deltas"):
                if hasattr(batch, attr):
                    val = getattr(batch, attr)
                    if val is not None and torch.is_tensor(val):
                        setattr(batch, attr, val.to(self.config.device))
        return batch

    def _forward(self, batch):
        out = self.model(batch)
        logits = out["logits"]
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        return logits

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        history = {}
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0
            n = 0
            for batch in train_loader:
                batch = self._move_batch(batch)
                labels = batch.labels if hasattr(batch, "labels") else batch["labels"]
                labels = labels.float().to(self.config.device)
                self.optim.zero_grad()
                logits = self._forward(batch)
                loss = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optim.step()
                total_loss += loss.item() * labels.size(0)
                n += labels.size(0)
            history[f"train_loss_epoch_{epoch}"] = total_loss / max(n, 1)
            if val_loader is not None:
                history[f"val_loss_epoch_{epoch}"] = self.evaluate_loss(val_loader)
        return history

    @torch.no_grad()
    def evaluate_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in loader:
            batch = self._move_batch(batch)
            labels = batch.labels if hasattr(batch, "labels") else batch["labels"]
            labels = labels.float().to(self.config.device)
            logits = self._forward(batch)
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
        return total_loss / max(n, 1)
