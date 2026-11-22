"""Entraîne un modèle multi-horizon sur un ensemble de patients (prétraitement appris sur ce seul ensemble)."""
from __future__ import annotations

import time

import numpy as np

from ..features.preprocessing import TabularPreprocessor
from ..models.baselines import make_tabular
from .horizon_inputs import horizon_view, tabular_columns, variation
from .horizon_model import HorizonModel, HorizonSpec
from .horizon_networks import fit_networks


def fit_horizon(spec: HorizonSpec, samples, X, visit, train_idx, settings, hcfg, seed: int = 42,
                n_seeds: int = 1) -> HorizonModel:
    start, months = time.time(), list(hcfg.months)
    if spec.family == "baseline":
        model = HorizonModel(spec, months)
    elif spec.family == "ml":
        model = _fit_lgbm(spec, samples, train_idx, months, seed, n_seeds)
    else:
        model = fit_networks(spec, samples, X, visit, train_idx, settings, hcfg, seed, n_seeds)
    model.info.update(fit_seconds=round(time.time() - start, 2), seeds=n_seeds)
    return model


def _fit_lgbm(spec, samples, train_idx, months, seed, n_seeds) -> HorizonModel:
    """Un LightGBM par horizon, appris sur les seules visites d'origine qui ont une cible à cet horizon."""
    y = variation(samples, months)
    rows = [train_idx[~np.isnan(y[train_idx, j])] for j in range(len(months))]
    views = [horizon_view(samples.iloc[r], h) for r, h in zip(rows, months, strict=True)]
    pres = [TabularPreprocessor(tabular_columns(spec.option("modalites", ()))).fit(v) for v in views]
    seeds = [[make_tabular("lgbm", seed + k, **dict(spec.options)).fit(pre.transform(v), y[r, j])
              for j, (pre, v, r) in enumerate(zip(pres, views, rows, strict=True))] for k in range(n_seeds)]
    return HorizonModel(spec, months, seeds, pres)
