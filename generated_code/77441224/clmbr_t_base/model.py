"""CLMBR-T-base implementation for EHRSHOT-style benchmarking.

This module provides a clean, self-contained implementation of a decoder-only
Transformer for structured EHR code sequences with a local causal attention
window. It is designed to be importable and usable for benchmarking, while
explicitly documenting assumptions where the public sources do not fully specify
exact preprocessing/tokenization details.

Key assumptions:
- Inputs are sequences of integer token ids representing standardized EHR codes.
- Tokenization/vocabulary construction is external to this module.
- The model uses causal local self-attention with a fixed attention radius.
- Patient representations are extracted from the final valid token hidden state.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CLMBRConfig:
    vocab_size: int
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: Optional[int] = None
    dropout: float = 0.0
    attention_window: int = 496
    max_position_embeddings: int = 5952
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.attention_window < 1:
            raise ValueError("attention_window must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")


class LocalCausalSelfAttention(nn.Module):
    def __init__(self, config: CLMBRConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        idx = torch.arange(seq_len, device=x.device)
        causal = idx[None, :] <= idx[:, None]
        if self.config.attention_window is not None:
            local = (idx[None, :] >= (idx[:, None] - (self.config.attention_window - 1)))
            causal = causal & local
        attn_scores = attn_scores.masked_fill(~causal[None, None, :, :], float("-inf"))
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out(context)


class TransformerBlock(nn.Module):
    def __init__(self, config: CLMBRConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = LocalCausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class CLMBRTBase(nn.Module):
    def __init__(self, config: CLMBRConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, seq_len)")
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")

        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, key_padding_mask=(attention_mask == 0))
        x = self.final_ln(x)

        logits = self.lm_head(x)
        out: Dict[str, torch.Tensor] = {"logits": logits}
        if return_hidden_states:
            out["hidden_states"] = x
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have same shape as input_ids")
            loss = self.next_token_loss(logits, labels, attention_mask)
            out["loss"] = loss
        return out

    @staticmethod
    def next_token_loss(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        loss = loss * shift_mask.view(-1)
        denom = shift_mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "last_valid",
    ) -> torch.Tensor:
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, return_hidden_states=True)
        hidden = outputs["hidden_states"]
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        if pooling == "last_valid":
            valid = attention_mask.sum(dim=1) > 0
            lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
            reps = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
            if not bool(valid.all()):
                reps = reps.clone()
                reps[~valid] = 0
            return reps
        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        raise ValueError(f"Unknown pooling: {pooling}")


def build_optimizer(model: nn.Module, learning_rate: float = 1e-5, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def save_checkpoint(model: CLMBRTBase, path: str) -> None:
    payload = {"config": asdict(model.config), "state_dict": model.state_dict()}
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> CLMBRTBase:
    payload = torch.load(path, map_location=map_location)
    config = CLMBRConfig(**payload["config"])
    model = CLMBRTBase(config)
    model.load_state_dict(payload["state_dict"])
    return model


class SequenceClassificationHead(nn.Module):
    def __init__(self, encoder: CLMBRTBase, num_labels: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        reps = self.encoder.encode(input_ids, attention_mask=attention_mask, pooling="last_valid")
        logits = self.classifier(reps)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out
