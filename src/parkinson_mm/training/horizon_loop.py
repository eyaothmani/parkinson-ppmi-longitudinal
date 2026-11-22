"""Boucle d'entraînement multi-horizon : AdamW, perte quantile + multitâche, arrêt anticipé sur la MAE de validation
(médiane prédite, toutes les cibles présentes). Mêmes réglages que l'objectif 2 (`params.yaml > training`)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from . import horizon_loss
from .loop import to_tensors


def _as_tensor(values, device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def train_multi_horizon(net, train: dict, val: dict, targets: tuple, cfg, quantiles, weight: float):
    """`targets` = (y_train, y_val, aux_train) : variations normalisées (n, horizons) et auxiliaires standardisées."""
    torch.manual_seed(cfg.seed)
    net.to(cfg.device)
    tr, q = to_tensors(train, cfg.device), _as_tensor(quantiles, cfg.device)
    y, y_val, aux = _as_tensor(targets[0], cfg.device), targets[1], _as_tensor(targets[2], cfg.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    generator = torch.Generator().manual_seed(cfg.seed)
    best, best_state, wait, epochs_run = np.inf, None, 0, 0
    for epochs_run in range(1, cfg.epochs + 1):  # noqa: B007 (valeur utilisée après la boucle)
        net.train()
        for idx in torch.randperm(len(y), generator=generator).split(cfg.batch_size):
            loss = horizon_loss.total(net({k: v[idx] for k, v in tr.items()}), y[idx], aux[idx], q, weight)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        val_mae = float(np.nanmean(np.abs(predict_quantiles(net, val)[..., 1] - y_val)))
        scheduler.step(val_mae)
        if val_mae < best - 1e-4:
            best, best_state, wait = val_mae, copy.deepcopy(net.state_dict()), 0
        elif (wait := wait + 1) >= cfg.patience:
            break
    net.load_state_dict(best_state)
    return net, {"epochs": epochs_run, "best_val_mae_scaled": best}


@torch.no_grad()
def predict_quantiles(net, arrays: dict, batch_size: int = 2048) -> np.ndarray:
    """(n, horizons, 3) dans l'échelle normalisée de la cible."""
    net.eval()
    data = to_tensors(arrays, next(net.parameters()).device.type)
    out = [net({k: v[idx] for k, v in data.items()})[0].cpu().numpy()
           for idx in torch.arange(len(data["x"])).split(batch_size)]
    return np.concatenate(out)
