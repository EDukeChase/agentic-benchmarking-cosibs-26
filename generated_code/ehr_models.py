"""Model architectures for structured EHR benchmarking.

This module provides clean, modular PyTorch implementations of five models
commonly used for structured longitudinal EHR prediction tasks:

- CLMBR-T-Base: causal transformer for next-code prediction / patient embeddings
- BEHRT: bidirectional transformer with visit/code/age embeddings
- MedBERT: BERT-style transformer with code/visit/serialization embeddings
- RETAIN: reverse-time attention model for interpretable visit prediction
- GRU-D: decay-aware recurrent model for irregular multivariate time series

The implementations are designed for benchmarking and research prototyping.
They intentionally expose a shared interface but preserve the distinct data
assumptions of each model family.

Important assumptions that are not fully specified in the literature review:
- CLMBR-T-Base is implemented as a causal Transformer encoder approximating the
  documented autoregressive next-code objective.
- BEHRT and MedBERT are implemented as visit-level token transformers with
  different embedding compositions.
- RETAIN is implemented on multi-hot visit vectors.
- GRU-D expects dense tensors with masks and delta times.

All models return logits for downstream prediction and can optionally return
intermediate representations for downstream embedding extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
from torch import nn


@dataclass
class SequenceBatch:
    """Container for code-sequence style models.

    Attributes:
        codes: LongTensor of shape [batch, seq_len]
        visits: Optional LongTensor of shape [batch, seq_len] visit ids.
        ages: Optional LongTensor of shape [batch, seq_len] age bucket ids.
        positions: Optional LongTensor of shape [batch, seq_len] positional ids.
        mask: Optional BoolTensor of shape [batch, seq_len] where True marks valid tokens.
        labels: Optional LongTensor of shape [batch] or [batch, seq_len].
    """

    codes: torch.Tensor
    visits: Optional[torch.Tensor] = None
    ages: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


@dataclass
class TimeSeriesBatch:
    """Container for GRU-D style irregular multivariate time-series data.

    Attributes:
        values: FloatTensor [batch, time, features]
        masks: FloatTensor or BoolTensor [batch, time, features], 1 if observed
        deltas: FloatTensor [batch, time, features], time since last observation
        labels: Optional targets [batch]
    """

    values: torch.Tensor
    masks: torch.Tensor
    deltas: torch.Tensor
    labels: Optional[torch.Tensor] = None


class TransformerEncoderWithPooling(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is None:
            pooled = h[:, -1]
        else:
            valid_lens = (~src_key_padding_mask).sum(dim=1).clamp(min=1) - 1
            pooled = h[torch.arange(h.size(0), device=h.device), valid_lens]
        return h, self.pool(pooled)


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, nhead: int = 12, num_layers: int = 12,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, codes: torch.Tensor, mask: Optional[torch.Tensor] = None, return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        bsz, seq_len = codes.shape
        pos = torch.arange(seq_len, device=codes.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(codes) + self.pos_emb(pos)
        x = self.dropout(x)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=codes.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = None if mask is None else ~mask.bool()
        h = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        h = self.ln(h)
        logits = self.head(h)
        out = {"logits": logits}
        if return_hidden:
            out["hidden_states"] = h
            out["pooled"] = h[:, -1]
        return out


class CLMBRTBase(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, nhead: int = 12, num_layers: int = 12,
                 max_len: int = 496, dropout: float = 0.1):
        super().__init__()
        self.backbone = CausalTransformer(vocab_size, d_model, nhead, num_layers, max_len=max_len, dropout=dropout)

    def forward(self, batch: SequenceBatch, return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        return self.backbone(batch.codes, mask=batch.mask, return_hidden=return_hidden)


class _VisitTransformerBase(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 4,
                 max_visits: int = 256, dropout: float = 0.1):
        super().__init__()
        self.code_emb = nn.Embedding(vocab_size, d_model)
        self.visit_emb = nn.Embedding(max_visits, d_model)
        self.pos_emb = nn.Embedding(max_visits, d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def _encode(self, codes: torch.Tensor, visits: Optional[torch.Tensor], ages: Optional[torch.Tensor], mask: Optional[torch.Tensor], extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len = codes.shape
        if visits is None:
            visits = torch.arange(seq_len, device=codes.device).unsqueeze(0).expand(bsz, -1)
        pos = torch.arange(seq_len, device=codes.device).unsqueeze(0).expand(bsz, -1)
        x = self.code_emb(codes) + self.visit_emb(visits.clamp(min=0, max=self.visit_emb.num_embeddings - 1)) + self.pos_emb(pos)
        if extra is not None:
            x = x + extra
        x = self.dropout(x)
        key_padding_mask = None if mask is None else ~mask.bool()
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.ln(h)


class BEHRT(_VisitTransformerBase):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 4,
                 max_visits: int = 256, max_age: int = 128, dropout: float = 0.1):
        super().__init__(vocab_size, d_model, nhead, num_layers, max_visits, dropout)
        self.age_emb = nn.Embedding(max_age, d_model)

    def forward(self, batch: SequenceBatch, return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        age = batch.ages if batch.ages is not None else None
        extra = None if age is None else self.age_emb(age.clamp(min=0, max=self.age_emb.num_embeddings - 1))
        h = self._encode(batch.codes, batch.visits, batch.ages, batch.mask, extra=extra)
        if batch.mask is None:
            pooled = h[:, -1]
        else:
            valid_lens = batch.mask.long().sum(dim=1).clamp(min=1) - 1
            pooled = h[torch.arange(h.size(0), device=h.device), valid_lens]
        logits = self.head(pooled)
        out = {"logits": logits.squeeze(-1), "pooled": pooled}
        if return_hidden:
            out["hidden_states"] = h
        return out


class MedBERT(_VisitTransformerBase):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 4,
                 max_visits: int = 256, max_serialization: int = 64, dropout: float = 0.1):
        super().__init__(vocab_size, d_model, nhead, num_layers, max_visits, dropout)
        self.serialization_emb = nn.Embedding(max_serialization, d_model)

    def forward(self, batch: SequenceBatch, serialization_ids: Optional[torch.Tensor] = None, return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        extra = None
        if serialization_ids is not None:
            extra = self.serialization_emb(serialization_ids.clamp(min=0, max=self.serialization_emb.num_embeddings - 1))
        h = self._encode(batch.codes, batch.visits, batch.ages, batch.mask, extra=extra)
        if batch.mask is None:
            pooled = h[:, -1]
        else:
            valid_lens = batch.mask.long().sum(dim=1).clamp(min=1) - 1
            pooled = h[torch.arange(h.size(0), device=h.device), valid_lens]
        logits = self.head(pooled)
        out = {"logits": logits.squeeze(-1), "pooled": pooled}
        if return_hidden:
            out["hidden_states"] = h
        return out


class RETAIN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.alpha_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.beta_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.alpha_fc = nn.Linear(hidden_dim, 1)
        self.beta_fc = nn.Linear(hidden_dim, embed_dim)
        self.out = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: SequenceBatch, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        x = self.dropout(self.embed(batch.codes))
        x_rev = torch.flip(x, dims=[1])
        h_alpha, _ = self.alpha_rnn(x_rev)
        h_beta, _ = self.beta_rnn(x_rev)
        logits_mask = None if batch.mask is None else torch.flip(batch.mask.float(), dims=[1])
        alpha_scores = self.alpha_fc(h_alpha).squeeze(-1)
        if logits_mask is not None:
            alpha_scores = alpha_scores.masked_fill(logits_mask <= 0, float("-inf"))
        alpha = torch.softmax(alpha_scores, dim=1)
        beta = torch.tanh(self.beta_fc(h_beta))
        c = torch.sum(alpha.unsqueeze(-1) * beta * x_rev, dim=1)
        logits = self.out(c).squeeze(-1)
        out = {"logits": logits, "pooled": c}
        if return_attention:
            out["visit_attention"] = alpha
        return out


class GRUD(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.gamma_x = nn.Linear(input_dim, input_dim)
        self.gamma_h = nn.Linear(input_dim, hidden_dim)
        self.x_mean = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self.gru_cell = nn.GRUCell(input_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, batch: TimeSeriesBatch, return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        x, m, d = batch.values, batch.masks.float(), batch.deltas
        bsz, time_steps, feat = x.shape
        h = torch.zeros(bsz, self.hidden_dim, device=x.device)
        x_last = self.x_mean.unsqueeze(0).expand(bsz, -1)
        hs = []
        for t in range(time_steps):
            gamma_x = torch.exp(-torch.relu(self.gamma_x(d[:, t])))
            gamma_h = torch.exp(-torch.relu(self.gamma_h(d[:, t])))
            x_hat = m[:, t] * x[:, t] + (1 - m[:, t]) * (gamma_x * x_last + (1 - gamma_x) * self.x_mean)
            h = gamma_h * h
            inp = torch.cat([x_hat, m[:, t]], dim=-1)
            h = self.gru_cell(inp, h)
            h = self.dropout(h)
            x_last = torch.where(m[:, t].bool(), x[:, t], x_last)
            hs.append(h.unsqueeze(1))
        h_seq = torch.cat(hs, dim=1)
        logits = self.out(h).squeeze(-1)
        out = {"logits": logits, "pooled": h}
        if return_hidden:
            out["hidden_states"] = h_seq
        return out
