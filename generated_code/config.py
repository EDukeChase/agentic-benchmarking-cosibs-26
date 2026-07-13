"""Configuration helpers for EHR benchmark models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ModelName = Literal["clmbr_t_base", "behrt", "med_bert", "retain", "grud"]


@dataclass
class ModelConfig:
    model_name: ModelName
    vocab_size: int = 1000
    input_dim: int = 32
    num_labels: int = 1
    d_model: int = 256
    hidden_dim: int = 128
    max_len: int = 512
    max_visits: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-2
    epochs: int = 10
