"""Réglage à budget égal (ADR-0018) : recherche aléatoire, même nombre d'essais pour LightGBM, le MLP et le GRU-D.

Chaque essai est évalué sur les plis de réglage (0 et 1) par la MAE moyenne des trois horizons ; l'essai 0 reprend les
réglages de l'objectif 2. La recherche aléatoire (Bergstra et Bengio, 2012) suffit à ce budget et n'ajoute aucune
dépendance. Les plis de test ne sont jamais lus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.horizon_metrics import summarize
from .horizon_cv import cross_validate_horizon


def sample(space: dict, rng: np.random.Generator) -> dict:
    """Une liste = choix parmi des valeurs ; {low, high[, log]} = valeur continue (échelle log si demandé)."""
    out = {}
    for key, rule in space.items():
        if isinstance(rule, list):
            out[key] = rule[int(rng.integers(len(rule)))]
        elif rule.get("log"):
            out[key] = float(np.exp(rng.uniform(np.log(rule["low"]), np.log(rule["high"]))))
        else:
            out[key] = float(rng.uniform(rule["low"], rule["high"]))
    return out


def tune(spec, samples, X, visit, settings, hcfg, trials: int, folds, seed: int) -> tuple[dict, pd.DataFrame]:
    """Renvoie (meilleurs réglages, tableau de tous les essais)."""
    rng, rows = np.random.default_rng(seed), []
    space = dict(hcfg.tuning.spaces[spec.algo])
    for trial in range(trials):
        params = sample(space, rng) if trial else {}
        oof = cross_validate_horizon(spec.with_options(**params), samples, X, visit, settings, hcfg, folds, seed)
        rows.append({"essai": trial, **params, "MAE_moyenne": float(summarize(oof, hcfg.mcid_points)["MAE"].mean())})
    table = pd.DataFrame(rows)
    best = rows[int(table["MAE_moyenne"].idxmin())]
    return {k: v for k, v in best.items() if k in space}, table
