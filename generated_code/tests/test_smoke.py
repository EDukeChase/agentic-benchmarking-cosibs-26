import torch
from torch.utils.data import DataLoader

from ehrshot_ml.config import ModelConfig, TrainConfig
from ehrshot_ml.data import PatientExample, SequenceDataset, collate_sequence_batch
from ehrshot_ml.models import GRUPatientModel, LSTMPatientModel, TransformerCLMBRLikeModel
from ehrshot_ml.training import train_torch_model, predict_torch_model
from ehrshot_ml.evaluation import evaluate_binary_classification


def _make_data():
    examples = [
        PatientExample(tokens=[1, 2, 3, 4], numeric_features=[0.1, 1.0], label=0),
        PatientExample(tokens=[1, 3, 5], numeric_features=[0.2, 0.9], label=1),
        PatientExample(tokens=[2, 4, 6, 7, 8], numeric_features=[0.3, 0.8], label=0),
        PatientExample(tokens=[1, 2], numeric_features=[0.4, 0.7], label=1),
    ]
    return examples


def test_gru_forward():
    batch = collate_sequence_batch([SequenceDataset(_make_data())[0], SequenceDataset(_make_data())[1]])
    model = GRUPatientModel(vocab_size=20, num_numeric_features=2)
    out = model(batch["tokens"], lengths=batch["lengths"], numeric_features=batch["numeric_features"])
    assert out.shape == (2,)


def test_lstm_forward():
    batch = collate_sequence_batch([SequenceDataset(_make_data())[0], SequenceDataset(_make_data())[1]])
    model = LSTMPatientModel(vocab_size=20, num_numeric_features=2)
    out = model(batch["tokens"], lengths=batch["lengths"], numeric_features=batch["numeric_features"])
    assert out.shape == (2,)


def test_transformer_forward():
    batch = collate_sequence_batch([SequenceDataset(_make_data())[0], SequenceDataset(_make_data())[1]])
    model = TransformerCLMBRLikeModel(vocab_size=20, embedding_dim=32, hidden_dim=32, num_layers=2, nhead=4, max_seq_len=16, num_numeric_features=2)
    out = model(batch["tokens"], lengths=batch["lengths"], numeric_features=batch["numeric_features"])
    assert out.shape == (2,)


def test_train_and_eval_smoke():
    examples = _make_data()
    ds = SequenceDataset(examples)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_sequence_batch)
    model = GRUPatientModel(vocab_size=20, num_numeric_features=2)
    cfg = TrainConfig(epochs=1, patience=1, lr=1e-2)
    model = train_torch_model(model, loader, loader, cfg)
    y_true, y_prob = predict_torch_model(model, loader)
    metrics = evaluate_binary_classification(y_true, y_prob)
    assert "auroc" in metrics and "auprc" in metrics
