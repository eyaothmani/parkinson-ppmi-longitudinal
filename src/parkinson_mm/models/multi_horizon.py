"""Réseau multi-horizon (ADR-0018) : un encodeur, puis des têtes partagées par les horizons 12, 24 et 36 mois.

- encodeur `grud` : toute la trajectoire (GRU-D) ; encodeur `mlp` : la dernière visite seulement (valeurs + masque).
  Têtes et perte identiques : la comparaison MLP / GRU-D isole l'apport de la trajectoire.
- sortie principale : quantiles 10/50/90 % de la VARIATION normalisée du MDS-UPDRS III, par horizon. Les bornes sont
  construites autour de la médiane (médiane ∓ softplus) : les quantiles ne se croisent jamais.
- sortie auxiliaire (multitâche) : UPDRS II et MoCA à chaque horizon, standardisés.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grud import GRUD
from .layers import mlp


class MultiHorizonNet(nn.Module):
    def __init__(self, kind: str, n_features: int, d_static: int, n_horizons: int, n_aux: int, d: int = 64,
                 dropout: float = 0.2, masks: bool = True, decay: bool = True):
        super().__init__()
        if kind not in ("grud", "mlp"):
            raise ValueError(f"Encodeur inconnu : {kind} (ADR-0018 : grud ou mlp)")
        self.kind, self.n_horizons, self.n_aux = kind, n_horizons, max(n_aux, 1)
        self.encoder = GRUD(n_features, d, masks, decay) if kind == "grud" else None
        d_encoded = d if kind == "grud" else 2 * n_features
        self.body = mlp(d_encoded + d_static, 2 * d, d, dropout)
        self.quantiles = nn.Linear(d, 3 * n_horizons)
        self.auxiliary = nn.Linear(d, self.n_aux * n_horizons)

    def forward(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        # mlp : séquences alignées à droite, [:, -1] = visite d'origine (valeurs + masque)
        encoded = self.encoder(b) if self.kind == "grud" else torch.cat([b["x"][:, -1], b["m"][:, -1]], dim=1)
        z = self.body(torch.cat([encoded, b["static"]], dim=1))
        raw = self.quantiles(z).view(-1, self.n_horizons, 3)
        median = raw[..., 0]
        q = torch.stack([median - F.softplus(raw[..., 1]), median, median + F.softplus(raw[..., 2])], dim=-1)
        return q, self.auxiliary(z).view(-1, self.n_horizons, self.n_aux)
