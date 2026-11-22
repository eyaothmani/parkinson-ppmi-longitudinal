"""Hyperparamètres d'entraînement des réseaux (valeurs réelles lues dans params.yaml > training)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 80
    patience: int = 12          # arrêt anticipé si la validation ne s'améliore plus
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    d: int = 64                 # taille des représentations internes
    dropout: float = 0.2
    seed: int = 42
    device: str = "cpu"


def from_params(cfg, device: str, fast: bool = False) -> TrainConfig:
    t, quick = cfg.training, cfg.fast_mode
    return TrainConfig(epochs=quick.epochs if fast else t.epochs, patience=quick.patience if fast else t.patience,
                       batch_size=t.batch_size, lr=t.lr, weight_decay=t.weight_decay, d=t.d, dropout=t.dropout,
                       seed=cfg.project.seed, device=device)
