"""Une ligne par patient et par visite, avec un axe de temps continu depuis la baseline (BL)."""
from __future__ import annotations

import pandas as pd

from .io import months_between
from .journal import CleaningLog

STATE_PRIORITY = {"untreated": 0, "OFF": 1, "ON": 2, "unknown": 3}
VISIT_PRIORITY = {"BL": 0, "V": 1, "SC": 2, "ST": 3, "U": 4, "P": 5}


def one_exam_per_visit(u: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Garde l'examen non traité, sinon OFF, sinon ON : OFF reflète mieux la maladie que l'effet du traitement.
    Les scores ON et OFF bruts sont conservés pour les analyses de sensibilité."""
    on_off = (u[u["med_state"].isin(["ON", "OFF"])]
              .pivot_table(index=["PATNO", "EVENT_ID"], columns="med_state", values="np3", aggfunc="mean")
              .rename(columns={"ON": "np3_on", "OFF": "np3_off"}).reset_index())
    ranked = u.assign(_rank=u["med_state"].map(STATE_PRIORITY))
    one = (ranked.sort_values(["PATNO", "EVENT_ID", "_rank", "n_items_valid"], ascending=[True, True, True, False])
           .drop_duplicates(["PATNO", "EVENT_ID"]).drop(columns="_rank").merge(on_off, how="left"))
    log.add("1 examen par visite", u, one, "Doublons ON/OFF : écart ~10 pts ; OFF prioritaire, état en covariable.")
    # Deux examens le même mois (SC + BL, ST + V04) ne sont pas deux points de trajectoire.
    one["_vp"] = one["EVENT_ID"].astype(str).map(lambda e: VISIT_PRIORITY.get(e if e in ("BL", "SC") else e[0], 9))
    monthly = one.sort_values(["PATNO", "visit_date", "_vp"]).drop_duplicates(["PATNO", "visit_date"])
    log.add("1 visite par mois", one, monthly, "Visite programmée prioritaire.")
    return monthly.drop(columns="_vp")


def add_time_axis(v: pd.DataFrame, log: CleaningLog, jump_points: float = 25, jump_months: int = 12) -> pd.DataFrame:
    v = v.sort_values(["PATNO", "visit_date"]).copy()
    bl = v[v["EVENT_ID"] == "BL"].groupby("PATNO")["visit_date"].min()
    v["bl_date"] = v["PATNO"].map(bl).fillna(v["PATNO"].map(v.groupby("PATNO")["visit_date"].min()))
    v["months_from_bl"] = months_between(v["bl_date"], v["visit_date"])
    before = v
    v = v[v["months_from_bl"] >= 0].copy()
    log.add("Mesures avant BL exclues", before, v, "Le temps 0 est la baseline.")
    group = v.groupby("PATNO")
    v["time_years"] = v["months_from_bl"] / 12
    v["n_visits"] = group["PATNO"].transform("size")
    # Saut implausible : on le SIGNALE sans le supprimer (une aggravation brutale peut être réelle).
    jump = (v["np3"] - group["np3"].shift()).abs() > jump_points
    recent = (v["months_from_bl"] - group["months_from_bl"].shift()) <= jump_months
    v["jump_flag"] = jump & recent
    return v
