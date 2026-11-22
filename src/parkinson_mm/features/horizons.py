"""Échantillons multi-horizon (ADR-0018) : visite d'origine -> MDS-UPDRS III le plus proche de 12, 24 et 36 mois.

Une ligne par visite d'origine. Pour chaque horizon h : cible `y_h`, cibles auxiliaires `<variable>_h`, délai réel
`gap_h` (années) et état de mesure prévu (`off_h`, `on_h`, `untreated_h`). Seule une visite FUTURE du même patient,
dans la fenêtre h ± tolérance, sert de cible ; à égalité de distance, la plus précoce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .history import with_history

STATES = ("off", "on", "untreated")
SUMMARY = ["np3_prev", "np3_slope_hist", "np3_mean_hist", "n_prior"]   # résumé de trajectoire (visites passées)


def condition_columns(months) -> list[str]:
    """Conditions connues à l'avance pour chaque horizon : délai et état de mesure prévus."""
    return [f"{name}_{h}" for h in months for name in ("gap", *STATES)]


def add_conditions(s: pd.DataFrame, h: int, gap_months, state) -> pd.DataFrame:
    """Délai en années (nominal si inconnu) et état prévu en indicateurs (tous à 0 si inconnu)."""
    gap = pd.Series(gap_months, index=s.index, dtype=float)
    s[f"gap_{h}"] = gap.fillna(h) / 12
    state = pd.Series(state, index=s.index, dtype=object).str.lower()
    for name in STATES:
        s[f"{name}_{h}"] = (state == name).astype(float)
    return s


def pair_horizons(v: pd.DataFrame, months, tolerance: int, auxiliary) -> pd.DataFrame:
    """Ajoute à chaque visite de `v` (index = position) ses cibles à chaque horizon."""
    target = v.loc[v["np3"].notna(), ["PATNO", "months_from_bl", "np3", "med_state", *auxiliary]]
    target = target.add_prefix("t_").rename(columns={"t_PATNO": "PATNO"}).reset_index(names="origin")
    pairs = v[["PATNO", "months_from_bl"]].reset_index(names="row").merge(target, on="PATNO")
    pairs["gap"] = pairs["t_months_from_bl"] - pairs["months_from_bl"]
    out = v.copy()
    for h in months:
        near = pairs[(pairs["gap"] - h).abs() <= tolerance].assign(dist=lambda d, h=h: (d["gap"] - h).abs())
        near = near.sort_values(["row", "dist", "gap"]).drop_duplicates("row").set_index("row")
        out[f"y_{h}"] = near["t_np3"]
        for name in auxiliary:
            out[f"{name}_{h}"] = near[f"t_{name}"]
        out = add_conditions(out, h, near["gap"], near["t_med_state"])
    return out


def make_horizon_samples(long: pd.DataFrame, patients: pd.DataFrame, months, tolerance: int,
                         auxiliary) -> pd.DataFrame:
    """Visites d'origine ayant au moins une cible ; historique calculé sur les visites passées uniquement."""
    v = pair_horizons(with_history(long), months, tolerance, auxiliary)
    has_target = np.column_stack([v[f"y_{h}"].notna() for h in months]).any(axis=1)
    s = v[has_target & v["np3"].notna().to_numpy()]
    return s.merge(patients, on="PATNO", how="left", suffixes=("", "_patient")).reset_index(drop=True)
