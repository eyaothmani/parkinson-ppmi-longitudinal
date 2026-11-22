"""Boucle d'entraînement d'un réseau : AdamW, perte de Huber, arrêt anticipé sur la MAE de validation."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn


def to_tensors(arrays: dict, device: str) -> dict:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in arrays.items()}


def train_network(net, train: dict, val: dict, y_train, y_val, y_scale: float, cfg) -> tuple[nn.Module, dict]:
    torch.manual_seed(cfg.seed)
    net.to(cfg.device)
    tr, target = to_tensors(train, cfg.device), torch.as_tensor(y_train / y_scale, dtype=torch.float32,
                                                                 device=cfg.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    huber = nn.SmoothL1Loss(beta=0.5)
    generator = torch.Generator().manual_seed(cfg.seed)
    best, best_state, wait, epochs_run = np.inf, None, 0, 0
    for epochs_run in range(1, cfg.epochs + 1):  # noqa: B007 (valeur utilisée après la boucle)
        net.train()
        for idx in torch.randperm(len(target), generator=generator).split(cfg.batch_size):
            batch = {k: v[idx] for k, v in tr.items()}
            loss = huber(net(batch), target[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        val_mae = float(np.mean(np.abs(predict(net, val, y_scale) - y_val)))
        scheduler.step(val_mae)
        if val_mae < best - 1e-4:
            best, best_state, wait = val_mae, copy.deepcopy(net.state_dict()), 0
        elif (wait := wait + 1) >= cfg.patience:
            break
    net.load_state_dict(best_state)
    return net, {"epochs": epochs_run, "best_val_mae_delta": best}


@torch.no_grad()
def predict(net, arrays: dict, y_scale: float, batch_size: int = 2048) -> np.ndarray:
    net.eval()
    device = next(net.parameters()).device.type
    data = to_tensors(arrays, device)
    out = [net({k: v[idx] for k, v in data.items()}).cpu().numpy()
           for idx in torch.arange(len(data["seq"])).split(batch_size)]
    return np.concatenate(out) * y_scale
