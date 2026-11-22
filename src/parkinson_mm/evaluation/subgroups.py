"""Sous-groupes de patients (sprint 6) : chaque exemple « visite i → visite i+1 » reçoit son groupe pour chaque
critère. Une donnée inconnue laisse le groupe vide : l'exemple ne compte pas pour ce critère."""
from __future__ import annotations

import numpy as np
import pandas as pd

STATES = {"untreated": "sans traitement", "OFF": "OFF", "ON": "ON"}


def genetics(s: pd.DataFrame) -> pd.Series:
    """LRRK2, GBA ou non porteur ; vide si le statut génétique est inconnu."""
    known = s[["lrrk2", "gba"]].notna().any(axis=1)
    label = np.select([s["lrrk2"].eq(1), s["gba"].eq(1), known], ["LRRK2", "GBA", "non porteur"], default="")
    return pd.Series(label, index=s.index).replace("", np.nan)


def assign_groups(s: pd.DataFrame) -> pd.DataFrame:
    """Une colonne par critère, même index que les échantillons."""
    year = pd.to_datetime(s["bl_date"]).dt.year
    return pd.DataFrame({
        "Stade (Hoehn & Yahr)": pd.cut(s["nhy"], [-0.5, 1.5, 2.5, 5], labels=["H&Y 0–1", "H&Y 2", "H&Y ≥ 3"]),
        "État de l'examen à prédire": s["target_state"].map(STATES),
        "Durée de maladie": pd.cut(s["disease_dur_years"], [-1, 2, 5, 100], labels=["< 2 ans", "2–5 ans", "> 5 ans"]),
        "Génétique": genetics(s),
        "Sexe": s["male"].map({1: "homme", 0: "femme"}),
        "Âge": pd.cut(s["age"], [0, 60, 70, 120], labels=["< 60 ans", "60–70 ans", "> 70 ans"]),
        "Période d'inclusion": pd.cut(year, [0, 2014, 2019, 2100], labels=["≤ 2014", "2015–2019", "≥ 2020"]),
    }, index=s.index)
