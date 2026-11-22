"""Durée de la maladie et dose de traitement (LEDD) à chaque visite."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import months_between, parse_month

LEVODOPA = r"levodopa|sinemet|stalevo|rytary|duopa|madopar|carbidopa|dhivy|crexont|inbrija"


def add_disease_duration(v: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
    if history is None:
        return v.assign(disease_dur_years=np.nan)
    h = history.drop_duplicates("PATNO")
    dx = v["PATNO"].map(h.set_index("PATNO")["PDDXDT"].pipe(parse_month))
    v["disease_dur_years"] = (months_between(dx, v["visit_date"]) / 12).where(lambda s: s > -1)
    return v


def add_ledd(v: pd.DataFrame, log: pd.DataFrame | None, ledd_max: float = 4000) -> pd.DataFrame:
    """LEDD (mg/j) = somme des traitements actifs à la date de visite.
    Les inhibiteurs de COMT sont codés « LD x 0.33 » : facteur appliqué à la lévodopa active (Tomlinson 2010)."""
    if log is None:
        return v.assign(ledd=np.nan)
    meds = log.assign(start=parse_month(log["STARTDT"]), stop=parse_month(log["STOPDT"]),
                      dose=pd.to_numeric(log["LEDD"], errors="coerce"),
                      factor=pd.to_numeric(log["LEDD"].astype("string").str.extract(r"LD\s*x\s*([0-9.]+)")[0],
                                           errors="coerce"),
                      is_levodopa=log["LEDTRT"].astype("string").str.contains(LEVODOPA, case=False).fillna(False))
    pairs = v[["PATNO", "visit_date"]].reset_index().merge(meds.dropna(subset=["start"]), on="PATNO")
    active = pairs[(pairs["start"] <= pairs["visit_date"])
                   & (pairs["stop"].isna() | (pairs["stop"] >= pairs["visit_date"]))]
    base = active.groupby("index")["dose"].sum()
    levodopa = active[active["is_levodopa"]].groupby("index")["dose"].sum()
    comt = active.groupby("index")["factor"].max()
    total = base.add((comt * levodopa).fillna(0), fill_value=0).reindex(v.index).fillna(0.0)
    untracked = ~v["PATNO"].isin(log["PATNO"]) & (v["on_treatment"] == 1)  # traité mais absent du journal
    v["ledd"] = total.mask(untracked).where(lambda s: s <= ledd_max)      # > ledd_max : erreur de saisie
    return v
