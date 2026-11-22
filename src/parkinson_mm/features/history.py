"""Variables d'historique d'un patient, calculées avec les visites PASSÉES uniquement (règle anti-fuite)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_step_encodings(v: pd.DataFrame) -> pd.DataFrame:
    return v.assign(med_off=(v["med_state"] == "OFF").astype(float), med_on=(v["med_state"] == "ON").astype(float))


def expanding_slope(t: pd.Series, y: pd.Series, patient: pd.Series) -> pd.Series:
    """Pente des moindres carrés de y ~ t sur les visites <= i (sommes cumulées) ; NaN avant 2 visites."""
    sums = pd.DataFrame({"n": 1.0, "t": t, "y": y, "tt": t * t, "ty": t * y}).groupby(patient.to_numpy()).cumsum()
    denominator = sums["n"] * sums["tt"] - sums["t"] ** 2
    slope = (sums["n"] * sums["ty"] - sums["t"] * sums["y"]) / denominator.where(denominator > 1e-9)
    return slope.where(sums["n"] >= 2)


def with_history(v: pd.DataFrame) -> pd.DataFrame:
    """Trie par patient et date, puis ajoute score précédent, pente, moyenne et rang de la visite."""
    v = add_step_encodings(v).sort_values(["PATNO", "visit_date"]).reset_index(drop=True)
    by_patient = v.groupby("PATNO")
    v["np3_prev"] = by_patient["np3"].shift(1)
    v["np3_slope_hist"] = expanding_slope(v["time_years"], v["np3"], v["PATNO"])
    v["np3_mean_hist"] = by_patient["np3"].transform(lambda s: s.expanding().mean())
    v["n_prior"] = by_patient.cumcount()
    v["row_in_long"] = np.arange(len(v))   # position, utilisée pour retrouver la séquence de chaque échantillon
    return v


def encode_target_conditions(v: pd.DataFrame) -> pd.DataFrame:
    """Écart jusqu'à la visite à prédire et état médicamenteux prévu à cette visite."""
    v["gap_years"] = v["gap_months"] / 12
    for state in ("OFF", "ON", "untreated"):
        v[f"target_{state.lower()}"] = (v["target_state"] == state).astype(float)
    return v
