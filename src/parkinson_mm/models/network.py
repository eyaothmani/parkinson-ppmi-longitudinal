"""Réseau de prédiction : encodeur de la trajectoire clinique (MLP ou GRU) + tête linéaire.

Il prédit la VARIATION normalisée du MDS-UPDRS III jusqu'à la visite suivante, à partir des variables cliniques
M1 (ADR-0015). Les fusions multimodales de la v1.0.0 sont consultables au tag `v1.0.0`.
"""
from __future__ import annotations

import torch.nn as nn

from .encoders import ClinicalEncoder


class LongitudinalNet(nn.Module):
    def __init__(self, kind: str, d_step: int, d_static: int, d: int = 64, dropout: float = 0.2):
        super().__init__()
        self.clinical = ClinicalEncoder(kind, d_step, d_static, d, dropout)
        self.head = nn.Linear(d, 1)

    def forward(self, b: dict):
        return self.head(self.clinical(b["seq"], b["static"])).squeeze(-1)
