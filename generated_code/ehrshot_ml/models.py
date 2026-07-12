from typing import Optional

import torch
import torch.nn as nn


class PatientEncoderBase(nn.Module):
    def forward(self, tokens, lengths=None, numeric_features=None):
        raise NotImplementedError


class GRUPatientModel(PatientEncoderBase):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=1, dropout=0.1, num_numeric_features=0, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim + num_numeric_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, tokens, lengths=None, numeric_features=None):
        x = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        rep = h[-1]
        if numeric_features is not None:
            rep = torch.cat([rep, numeric_features], dim=-1)
        return self.head(rep).squeeze(-1)


class LSTMPatientModel(PatientEncoderBase):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=1, dropout=0.1, num_numeric_features=0, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim + num_numeric_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, tokens, lengths=None, numeric_features=None):
        x = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        rep = h[-1]
        if numeric_features is not None:
            rep = torch.cat([rep, numeric_features], dim=-1)
        return self.head(rep).squeeze(-1)


class TransformerCLMBRLikeModel(PatientEncoderBase):
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=768, num_layers=12, dropout=0.0, num_numeric_features=0, num_classes=1, nhead=8, max_seq_len=496):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=4 * embedding_dim, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Identity() if hidden_dim == embedding_dim else nn.Linear(embedding_dim, hidden_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim + num_numeric_features, num_classes))
        self.max_seq_len = max_seq_len

    def forward(self, tokens, lengths=None, numeric_features=None):
        bsz, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_embedding(tokens) + self.pos_embedding(positions.clamp(max=self.max_seq_len - 1))
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=tokens.device), diagonal=1).bool()
        key_padding_mask = tokens.eq(0)
        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        if lengths is None:
            rep = x[:, -1]
        else:
            idx = (lengths - 1).clamp(min=0)
            rep = x[torch.arange(bsz, device=tokens.device), idx]
        rep = self.proj(rep)
        if numeric_features is not None:
            rep = torch.cat([rep, numeric_features], dim=-1)
        return self.head(rep).squeeze(-1)
