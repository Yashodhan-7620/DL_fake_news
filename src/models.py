"""PyTorch text classifiers: BiLSTM, GRU, TextCNN. Each outputs ONE logit (P(real) = sigmoid)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .preprocessing import PAD_ID


def _masked_mean_max(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool over time ignoring padding. h: (B,T,H) mask: (B,T) bool."""
    m = mask.unsqueeze(-1)
    mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
    mx = h.masked_fill(~m, -1e9).max(1).values
    return torch.cat([mean, mx], dim=1)


class RecurrentClassifier(nn.Module):
    """Embedding -> (Bi)LSTM / GRU -> masked mean+max pooling -> dropout -> linear."""

    def __init__(self, vocab_size, embed_dim, hidden, kind="lstm", bidirectional=True, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(embed_dim, hidden, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1) * 2
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        mask = x != PAD_ID
        # an all-pad row would break packing; guarantee length >= 1
        lengths = mask.sum(1).clamp(min=1).cpu()
        e = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.fc(self.drop(_masked_mean_max(h, mask))).squeeze(-1)


class TextCNN(nn.Module):
    """Embedding -> parallel Conv1d (k=3,4,5) -> global max pool -> dropout -> linear."""

    def __init__(self, vocab_size, embed_dim, filters=128, kernel_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, filters, k, padding=k // 2) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernel_sizes), 1)

    def forward(self, x):
        mask = (x != PAD_ID).unsqueeze(1)                       # (B,1,T)
        e = self.drop(self.emb(x)).transpose(1, 2)              # (B,E,T)
        feats = []
        for conv in self.convs:
            c = torch.relu(conv(e))[:, :, : x.size(1)]          # even kernels add 1 extra step
            feats.append(c.masked_fill(~mask, -1e9).max(2).values)
        return self.fc(self.drop(torch.cat(feats, 1))).squeeze(-1)


def build_model(name: str, vocab_size: int, ncfg: dict) -> nn.Module:
    d, e = ncfg["dropout"], ncfg["embed_dim"]
    if name == "lstm":
        return RecurrentClassifier(vocab_size, e, ncfg["lstm"]["hidden"], "lstm", True, d)
    if name == "gru":
        return RecurrentClassifier(vocab_size, e, ncfg["gru"]["hidden"], "gru", True, d)
    if name == "cnn":
        c = ncfg["cnn"]
        return TextCNN(vocab_size, e, c["filters"], tuple(c["kernel_sizes"]), d)
    raise ValueError(f"Unknown model '{name}' (choose lstm, gru or cnn)")
