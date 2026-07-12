import argparse

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import PatientExample, SequenceDataset, collate_sequence_batch
from .models import GRUPatientModel, LSTMPatientModel, TransformerCLMBRLikeModel
from .training import train_torch_model, predict_torch_model
from .evaluation import evaluate_binary_classification


def build_model(model_config: ModelConfig):
    if model_config.model_name == "gru":
        return GRUPatientModel(**model_config.__dict__)
    if model_config.model_name == "lstm":
        return LSTMPatientModel(**model_config.__dict__)
    if model_config.model_name in {"clmbr_t_base", "transformer"}:
        return TransformerCLMBRLikeModel(**model_config.__dict__)
    raise ValueError(f"Unknown model name: {model_config.model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gru")
    args = parser.parse_args()
    print(f"Selected model: {args.model_name}")


if __name__ == "__main__":
    main()
