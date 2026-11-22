"""Persistance ajustée à l'état pour les horizons 12, 24 et 36 mois (validation croisée seulement).

Même idée que pour la visite suivante : à chaque horizon, dernier score + variation médiane pour la même transition
d'état (état de l'examen d'origine -> état prévu à l'horizon), apprise sur les autres plis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .reference import NAME, shifted_persistence

STATES = {"off": "OFF", "on": "ON", "untreated": "untreated"}   # colonnes off_h, on_h, untreated_h des échantillons


def horizon_states(samples: pd.DataFrame, months) -> pd.DataFrame:
    """État prévu de l'examen cible à chaque horizon ; « unknown » si aucun indicateur n'est à 1."""
    parts = []
    for h in months:
        state = pd.Series("unknown", index=samples.index, dtype=object)
        for column, label in STATES.items():
            state = state.mask(samples[f"{column}_{h}"] == 1, label)
        parts.append(pd.DataFrame({"sample_id": samples.index, "horizon": h, "target_state": state.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def adjusted_horizons(oof: pd.DataFrame, samples: pd.DataFrame, months, statistic: str = "median") -> pd.DataFrame:
    """Prédictions hors-pli de la persistance ajustée, par horizon (mêmes lignes que la persistance brute)."""
    p = oof[oof["model"] == "Persistance"].merge(horizon_states(samples, months), on=["sample_id", "horizon"])
    origin = samples["med_state"].astype(str).reindex(p["sample_id"]).to_numpy()
    p["transition"] = origin + " -> " + p["target_state"]
    pred = shifted_persistence(p, ("horizon", "transition"), statistic, within=("horizon",))
    return p.drop(columns=["target_state", "transition"]).assign(model=NAME, family="baseline", pred=pred,
                                                                 q_low=np.nan, q_high=np.nan)
