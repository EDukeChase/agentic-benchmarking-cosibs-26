from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PatientExample:
    tokens: Sequence[int]
    numeric_features: Optional[Sequence[float]] = None
    label: Optional[float] = None


class SequenceDataset(Dataset):
    def __init__(self, examples: Sequence[PatientExample], max_seq_len: int = 512):
        self.examples = list(examples)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = list(ex.tokens)[-self.max_seq_len :]
        numeric = ex.numeric_features
        label = ex.label
        return {"tokens": tokens, "numeric_features": numeric, "label": label}


def collate_sequence_batch(batch, pad_token_id: int = 0):
    token_lists = [item["tokens"] for item in batch]
    max_len = max(len(t) for t in token_lists) if token_lists else 0
    tokens = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    lengths = torch.tensor([len(t) for t in token_lists], dtype=torch.long)
    for i, seq in enumerate(token_lists):
        if seq:
            tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    numeric = None
    if batch[0]["numeric_features"] is not None:
        numeric = torch.tensor(np.asarray([b["numeric_features"] for b in batch]), dtype=torch.float32)
    labels = None
    if batch[0]["label"] is not None:
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    return {"tokens": tokens, "lengths": lengths, "numeric_features": numeric, "labels": labels}
