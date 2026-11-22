"""Réseaux multi-horizon (MLP dernière visite, GRU-D) : prétraitement appris sur le train, arrêt anticipé sur 15 %
des patients du train (jamais les patients évalués), un réseau par graine."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from ..models.multi_horizon import MultiHorizonNet
from .horizon_inputs import HorizonInputs, variation
from .horizon_loop import train_multi_horizon
from .horizon_model import HorizonModel

TUNABLE = ("lr", "d", "dropout", "weight_decay", "batch_size")


def fit_networks(spec, samples, X, visit, train_idx, settings, hcfg, seed, n_seeds) -> HorizonModel:
    settings = replace(settings, **{k: v for k, v in spec.options if k in TUNABLE})    # réglages (ADR-0018)
    months, patients = list(hcfg.months), samples["PATNO"].to_numpy()
    pool = np.unique(patients[train_idx])        # 15 % des patients du train : arrêt anticipé
    held = np.isin(patients[train_idx], np.random.default_rng(seed).choice(pool, max(1, int(0.15 * len(pool))),
                                                                            replace=False))
    fit_idx, val_idx = train_idx[~held], train_idx[held]
    inputs = HorizonInputs(months, hcfg.auxiliary, spec.option("modalites", ())).fit(
        samples.iloc[fit_idx], X[fit_idx], visit[fit_idx])
    tr, va = (inputs.transform(samples.iloc[i], X[i], visit[i]) for i in (fit_idx, val_idx))
    y = variation(samples, months)
    y_scale = float(np.nanstd(y[fit_idx])) or 1.0
    targets = (y[fit_idx] / y_scale, y[val_idx] / y_scale, inputs.auxiliary_targets(samples.iloc[fit_idx]))
    weight = spec.option("auxiliary_weight", hcfg.auxiliary_weight)
    nets, epochs = [], []
    for k in range(n_seeds):
        torch.manual_seed(seed + k)
        net = MultiHorizonNet(spec.algo, tr["x"].shape[-1], tr["static"].shape[1], len(months), len(hcfg.auxiliary),
                              settings.d, settings.dropout, spec.option("masks", True), spec.option("decay", True))
        net, info = train_multi_horizon(net, tr, va, targets, replace(settings, seed=seed + k), hcfg.quantiles, weight)
        nets.append(net.cpu())
        epochs.append(info["epochs"])
    return HorizonModel(spec, months, nets, inputs, y_scale, {"epochs": epochs})
