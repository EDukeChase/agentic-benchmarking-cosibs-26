from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int
    num_numeric_features: int = 0
    num_classes: int = 1
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    max_seq_len: int = 512
    model_name: str = "gru"


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 10
    patience: int = 3
    device: str = "cpu"
    task_type: str = "binary"
    seed: int = 42
