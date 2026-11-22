"""Entraîne un modèle sur un ensemble de patients (prétraitement appris sur ce seul ensemble)."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from ..features.preprocessing import TabularPreprocessor
from ..models.baselines import make_tabular
from .inputs import TABULAR, NetworkInputs
from .spec import FittedModel, ModelSpec


def fit(spec: ModelSpec, samples, X, mask, train_idx, cfg=None, seed: int = 42, n_seeds: int = 1) -> FittedModel:
    start = time.time()
    if spec.family == "baseline":
        return FittedModel(spec, info={"fit_seconds": 0.0})
    delta = samples["delta"].to_numpy()
    if spec.family == "ml":
        pre = TabularPreprocessor(TABULAR).fit(samples.iloc[train_idx])
        model = make_tabular(spec.algo, seed).fit(pre.transform(samples.iloc[train_idx]), delta[train_idx])
        return FittedModel(spec, [model], pre, info={"fit_seconds": round(time.time() - start, 2)})
    return _fit_network(spec, samples, X, mask, train_idx, cfg, seed, n_seeds, start)


def _fit_network(spec, samples, X, mask, train_idx, cfg, seed, n_seeds, start) -> FittedModel:
    import torch

    from ..models.network import LongitudinalNet
    from .loop import train_network

    # 15 % des patients du train servent à l'arrêt anticipé (jamais les patients évalués)
    patients = samples["PATNO"].to_numpy()
    rng = np.random.default_rng(seed)
    pool = np.unique(patients[train_idx])
    held = np.isin(patients[train_idx], rng.choice(pool, max(1, int(0.15 * len(pool))), replace=False))
    fit_idx, val_idx = train_idx[~held], train_idx[held]
    inputs = NetworkInputs().fit(samples.iloc[fit_idx], X[fit_idx], mask[fit_idx])
    tr = inputs.transform(samples.iloc[fit_idx], X[fit_idx], mask[fit_idx])
    va = inputs.transform(samples.iloc[val_idx], X[val_idx], mask[val_idx])
    delta, dims = samples["delta"].to_numpy(), inputs.dims(tr)
    y_scale = float(np.std(delta[fit_idx])) or 1.0     # la cible est normalisée pour stabiliser l'apprentissage
    nets, epochs = [], []
    for k in range(n_seeds):
        torch.manual_seed(seed + k)
        net = LongitudinalNet(spec.algo, dims["step"], dims["static"], cfg.d, cfg.dropout)
        net, info = train_network(net, tr, va, delta[fit_idx], delta[val_idx], y_scale, replace(cfg, seed=seed + k))
        nets.append(net.cpu())
        epochs.append(info["epochs"])
    return FittedModel(spec, nets, inputs, y_scale, {"fit_seconds": round(time.time() - start, 2), "epochs": epochs})
