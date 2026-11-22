"""Sensibilité de l'objectif 2 : mêmes modèles, mêmes plis, mêmes graines, certaines variables neutralisées.

Neutraliser = mettre la variable à « manquant » pour tous les échantillons : le prétraitement (appris sur le train)
la remplace par une constante, elle ne porte donc plus aucune information. Le code d'entraînement n'est pas modifié.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.groups import STEP_CLIN
from ..training.spec import ModelSpec

CHANNELS = [*STEP_CLIN, "dt_prev"]          # canaux des séquences (features/sequences.py)


def neutralize(samples: pd.DataFrame, X: np.ndarray, columns) -> tuple[pd.DataFrame, np.ndarray]:
    """Copies des échantillons et des séquences où les `columns` sont manquantes partout."""
    s, X = samples.copy(), X.copy()
    s[[c for c in columns if c in s]] = np.nan
    X[..., [i for i, c in enumerate(CHANNELS) if c in columns]] = np.nan
    return s, X


def variant_specs(cfg, label: str, models) -> list[ModelSpec]:
    """Un ModelSpec par modèle de l'objectif 2, renommé « <modèle> – <variante> »."""
    base = {e["name"]: e for e in cfg.experiments}
    return [ModelSpec(f"{name} – {label}", base[name]["family"], base[name]["algo"], cfg.sensibilite.question)
            for name in models]


def planned_pairs(cfg) -> list[tuple[str, str]]:
    """Comparaisons fixées à l'avance : chaque variante face à son modèle d'origine ; « sans état cible » face à la
    persistance brute (aucun des deux ne connaît l'état de l'examen à prédire)."""
    pairs = []
    for label, entry in cfg.sensibilite.variantes.items():
        for name in entry["modeles"]:
            pairs.append((f"{name} – {label}", name))
            if entry.get("face_persistance"):
                pairs.append((f"{name} – {label}", "Persistance"))
    return pairs
