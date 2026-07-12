from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluation import evaluate_binary_classification


def train_torch_model(model, train_loader, val_loader, config):
    device = torch.device(config.device)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_state = None
    best_val = float("inf")
    patience = 0
    for _ in range(config.epochs):
        model.train()
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            lengths = batch["lengths"].to(device)
            numeric = batch["numeric_features"]
            if numeric is not None:
                numeric = numeric.to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(tokens, lengths=lengths, numeric_features=numeric)
            loss = criterion(logits.view(-1), labels.view(-1))
            loss.backward()
            optimizer.step()
        val_loss = _evaluate_loss(model, val_loader, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _evaluate_loss(model, loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            lengths = batch["lengths"].to(device)
            numeric = batch["numeric_features"]
            if numeric is not None:
                numeric = numeric.to(device)
            labels = batch["labels"].to(device)
            logits = model(tokens, lengths=lengths, numeric_features=numeric)
            losses.append(criterion(logits.view(-1), labels.view(-1)).item())
    return float(np.mean(losses)) if losses else float("inf")


def predict_torch_model(model, loader, device="cpu"):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            lengths = batch["lengths"].to(device)
            numeric = batch["numeric_features"]
            if numeric is not None:
                numeric = numeric.to(device)
            batch_labels = batch["labels"]
            logits = model(tokens, lengths=lengths, numeric_features=numeric)
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            labels.extend(batch_labels.numpy().tolist())
    return np.asarray(labels), np.asarray(probs)
