"""Validation croisée multi-horizon groupée par patient (mêmes 5 plis que l'objectif 2) : prédictions hors-pli."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.horizon_metrics import long_format
from .horizon_fit import fit_horizon
from .horizon_model import HorizonSpec


def horizon_specs(cfg, only=None, ablations: bool = True) -> list[HorizonSpec]:
    """Les 4 modèles comparés (ADR-0018), puis les ablations du GRU-D (validation croisée seulement)."""
    entries = [*cfg.horizons.experiments, *(cfg.horizons.ablations if ablations else [])]
    return [HorizonSpec.from_params(e) for e in entries if not only or e["name"] in only]


def cross_validate_horizon(spec: HorizonSpec, samples: pd.DataFrame, X, visit, settings, hcfg, folds, seed: int,
                           on_fold=None, n_seeds: int = 1) -> pd.DataFrame:
    """Le test n'est JAMAIS utilisé ici. `on_fold(pli, prédictions, info)` trace chaque pli (run imbriqué)."""
    dev = (samples["split"] == "dev").to_numpy()
    parts = []
    for fold in folds:
        held_out = np.where(dev & (samples["fold"] == fold).to_numpy())[0]
        train = np.where(dev & (samples["fold"] != fold).to_numpy())[0]
        model = fit_horizon(spec, samples, X, visit, train, settings, hcfg, seed + int(fold), n_seeds)
        quantiles = model.predict(samples.iloc[held_out], X[held_out], visit[held_out])
        part = long_format(samples, held_out, quantiles, hcfg.months, spec.name, family=spec.family, fold=int(fold))
        parts.append(part)
        if on_fold:
            on_fold(int(fold), part, model.info)
    return pd.concat(parts, ignore_index=True)
