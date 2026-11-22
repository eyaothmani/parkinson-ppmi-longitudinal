"""Autres échelles cliniques par visite : UPDRS I, II, IV, Schwab & England, âge."""
from __future__ import annotations

import pandas as pd

# table -> {colonne: (min, max)} ; une valeur hors bornes devient manquante
SCALES = {
    "updrs1": {"NP1RTOT": (0, 24)},
    "updrs1p": {"NP1PTOT": (0, 28)},
    "updrs2": {"NP2PTOT": (0, 52), "NP2TRMR": (0, 4), "NP2WALK": (0, 4), "NP2FREZ": (0, 4)},
    "updrs4": {"NP4TOT": (0, 24)},
    "schwab": {"MSEADLG": (0, 100)},
}


def per_visit(table: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    out = table[["PATNO", "EVENT_ID", *bounds]].copy()
    for col, (low, high) in bounds.items():
        values = pd.to_numeric(out[col], errors="coerce")
        out[col] = values.where(values.between(low, high))
    return out.groupby(["PATNO", "EVENT_ID"], as_index=False).mean()


def add_clinical_scales(v: pd.DataFrame, tables: dict) -> pd.DataFrame:
    for key, bounds in SCALES.items():
        if key in tables:
            v = v.merge(per_visit(tables[key], bounds), on=["PATNO", "EVENT_ID"], how="left")
    v["updrs1"] = v["NP1RTOT"] + v["NP1PTOT"]
    v = v.rename(columns={"NP2PTOT": "updrs2", "NP4TOT": "updrs4", "MSEADLG": "schwab"})
    # Scores tremblement / PIGD complets (parties II + III, Stebbins 2013)
    v["tremor_score"] = (v["tremor3"] + v["NP2TRMR"]) / 11
    v["pigd_score"] = (v["pigd3"] + v["NP2WALK"] + v["NP2FREZ"]) / 5
    return add_age(v, tables["age"])


def add_age(v: pd.DataFrame, age: pd.DataFrame) -> pd.DataFrame:
    age = age.groupby(["PATNO", "EVENT_ID"], as_index=False)["AGE_AT_VISIT"].mean()
    v = v.merge(age, on=["PATNO", "EVENT_ID"], how="left").rename(columns={"AGE_AT_VISIT": "age"})
    # Âge manquant : âge à BL + temps écoulé
    age_at_bl = (v["age"] - v["time_years"]).groupby(v["PATNO"]).median()
    v["age"] = v["age"].fillna(v["PATNO"].map(age_at_bl) + v["time_years"])
    return v
