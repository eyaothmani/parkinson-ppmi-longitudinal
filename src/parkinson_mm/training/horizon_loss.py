"""Pertes du multi-horizon : perte quantile (pinball) sur l'UPDRS III + Huber sur les cibles auxiliaires.

Les cibles absentes (aucune mesure dans la fenêtre d'un horizon) sont exclues : un patient suivi 24 mois apprend
quand même ses horizons 12 et 24 mois. La perte quantile rend l'intervalle 10–90 % directement interprétable (H4).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def pinball(pred: torch.Tensor, y: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """pred (n, horizons, quantiles), y (n, horizons) avec NaN pour les cibles absentes."""
    mask = (~torch.isnan(y)).float()
    diff = torch.nan_to_num(y)[..., None] - pred
    loss = torch.maximum(quantiles * diff, (quantiles - 1) * diff).mean(dim=-1)
    return masked_mean(loss, mask)


def auxiliary(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (~torch.isnan(y)).float()
    return masked_mean(F.smooth_l1_loss(pred, torch.nan_to_num(y), reduction="none", beta=0.5), mask)


def total(outputs, y: torch.Tensor, y_aux: torch.Tensor, quantiles: torch.Tensor, weight: float) -> torch.Tensor:
    q, aux = outputs
    loss = pinball(q, y, quantiles)
    return loss + weight * auxiliary(aux, y_aux) if weight > 0 else loss
