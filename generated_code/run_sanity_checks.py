"""Sanity checks for the EHR model implementations."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ehr_models import SequenceBatch, TimeSeriesBatch, CLMBRTBase, BEHRT, MedBERT, RETAIN, GRUD
from training import TrainConfig, BinaryClassificationTrainer


def main():
    torch.manual_seed(7)
    batch_size = 4
    seq_len = 12
    vocab_size = 50

    seq_codes = torch.randint(0, vocab_size, (batch_size, seq_len))
    seq_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    seq_visits = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    seq_ages = torch.randint(0, 20, (batch_size, seq_len))
    labels = torch.randint(0, 2, (batch_size,)).float()

    seq_batch = SequenceBatch(codes=seq_codes, visits=seq_visits, ages=seq_ages, mask=seq_mask, labels=labels)

    clmbr = CLMBRTBase(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2, max_len=seq_len)
    behrt = BEHRT(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2, max_visits=seq_len)
    medbert = MedBERT(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2, max_visits=seq_len)
    retain = RETAIN(vocab_size=vocab_size, embed_dim=32, hidden_dim=32)

    for name, model in [("clmbr", clmbr), ("behrt", behrt), ("medbert", medbert), ("retain", retain)]:
        out = model(seq_batch)
        assert out["logits"].shape[0] == batch_size, f"{name} logits shape wrong"

    ts_values = torch.randn(batch_size, seq_len, 8)
    ts_masks = torch.randint(0, 2, (batch_size, seq_len, 8)).float()
    ts_deltas = torch.rand(batch_size, seq_len, 8)
    ts_batch = TimeSeriesBatch(values=ts_values, masks=ts_masks, deltas=ts_deltas, labels=labels)
    grud = GRUD(input_dim=8, hidden_dim=16)
    out = grud(ts_batch)
    assert out["logits"].shape[0] == batch_size, "grud logits shape wrong"

    class SequenceDataset:
        def __len__(self):
            return seq_codes.size(0)

        def __getitem__(self, idx):
            return SequenceBatch(
                codes=seq_codes[idx],
                visits=seq_visits[idx],
                ages=seq_ages[idx],
                mask=seq_mask[idx],
                labels=labels[idx],
            )

    trainer = BinaryClassificationTrainer(retain, TrainConfig(epochs=1, device="cpu"))
    trainer.fit(DataLoader(SequenceDataset(), batch_size=2, shuffle=False))
    print("sanity checks passed")


if __name__ == "__main__":
    main()
