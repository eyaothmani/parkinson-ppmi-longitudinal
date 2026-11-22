"""Échantillons « visite i -> visite i+1 » pour l'entraînement, et dernière visite pour l'inférence."""
from __future__ import annotations

import pandas as pd

from .history import encode_target_conditions, with_history


def make_training_samples(long: pd.DataFrame, patients: pd.DataFrame, min_gap: int = 3,
                          max_gap: int = 24) -> pd.DataFrame:
    """Une ligne par paire de visites consécutives dont l'écart est entre `min_gap` et `max_gap` mois."""
    v = with_history(long)
    following = v.groupby("PATNO")[["np3", "months_from_bl", "med_state"]].shift(-1)
    v["y_next"] = following["np3"]
    v["gap_months"] = following["months_from_bl"] - v["months_from_bl"]
    v["target_state"] = following["med_state"]
    v = encode_target_conditions(v)
    s = v[v["y_next"].notna() & v["gap_months"].between(min_gap, max_gap)].copy()
    s["delta"] = s["y_next"] - s["np3"]          # ce que les modèles apprennent : la VARIATION
    return s.merge(patients, on="PATNO", how="left", suffixes=("", "_patient")).reset_index(drop=True)


def make_inference_samples(long: pd.DataFrame, patients: pd.DataFrame, gap_months, target_state):
    """Dernière visite connue de chaque patient + conditions de la visite à prédire.
    Retourne aussi l'historique trié, nécessaire pour construire les séquences."""
    v = with_history(long)
    last = v.groupby("PATNO").tail(1).copy()
    last["gap_months"] = last["PATNO"].map(gap_months) if isinstance(gap_months, pd.Series) else gap_months
    last["target_state"] = last["PATNO"].map(target_state) if isinstance(target_state, pd.Series) else target_state
    last = encode_target_conditions(last)
    return last.merge(patients, on="PATNO", how="left", suffixes=("", "_patient")).reset_index(drop=True), v
