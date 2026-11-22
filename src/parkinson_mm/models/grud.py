"""Récurrence GRU-D (Che et al., 2018) : un GRU qui sait qu'une mesure manque et depuis combien de temps.

À chaque visite : x̂ = m·x + (1 − m)·γx·x_dernier (la moyenne vaut 0 après standardisation) et h ← γh·h avant la mise à
jour, avec γ = exp(−max(0, W·δ + b)) : plus la dernière mesure est ancienne, moins elle pèse. Les visites de
remplissage (séquences alignées à droite) ne modifient pas l'état caché.
Ablations (ADR-0018) : `masks=False` retire le masque des entrées ; `decay=False` retire la décroissance (γ = 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUD(nn.Module):
    def __init__(self, n_features: int, d: int = 64, masks: bool = True, decay: bool = True):
        super().__init__()
        self.d, self.masks, self.decay = d, masks, decay
        # poids > 0 au départ : avec 0, max(0, ·) aurait un gradient nul et la décroissance ne s'apprendrait jamais
        self.w_x = nn.Parameter(torch.empty(n_features).uniform_(0.05, 0.5))
        self.b_x = nn.Parameter(torch.zeros(n_features))
        self.decay_h = nn.Linear(n_features, d)
        self.cell = nn.GRUCell(2 * n_features if masks else n_features, d)

    def forward(self, b: dict) -> torch.Tensor:
        x, m, delta, last, visit = b["x"], b["m"], b["delta"], b["x_last"], b["visit"]
        h = x.new_zeros(x.shape[0], self.d)
        for k in range(x.shape[1]):
            gamma_x, gamma_h = 1.0, 1.0
            if self.decay:
                gamma_x = torch.exp(-torch.relu(delta[:, k] * self.w_x + self.b_x))
                gamma_h = torch.exp(-torch.relu(self.decay_h(delta[:, k])))
            x_hat = m[:, k] * x[:, k] + (1 - m[:, k]) * gamma_x * last[:, k]
            step = torch.cat([x_hat, m[:, k]], dim=1) if self.masks else x_hat
            h = torch.where(visit[:, k, None] > 0, self.cell(step, gamma_h * h), h)
        return h
