"""Encodeurs de la trajectoire clinique : une seule visite (MLP) ou toute la trajectoire (GRU)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import mlp


class ClinicalEncoder(nn.Module):
    """mlp : dernière visite seulement | gru : récurrence sur les visites passées."""

    def __init__(self, kind: str, d_step: int, d_static: int, d: int = 64, dropout: float = 0.2):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.head = mlp(d_step + d_static, 2 * d, d, dropout)
        elif kind == "gru":
            self.rnn = nn.GRU(d_step, d, num_layers=2, batch_first=True, dropout=dropout)
            self.head = mlp(d + d_static, 2 * d, d, dropout)
        else:
            raise ValueError(f"Encodeur inconnu : {kind} (ADR-0015 : mlp ou gru)")

    def forward(self, seq, static):
        if self.kind == "mlp":                       # séquences alignées à droite : [:, -1] = visite courante
            return self.head(torch.cat([seq[:, -1], static], dim=1))
        out, _ = self.rnn(seq)
        return self.head(torch.cat([out[:, -1], static], dim=1))
