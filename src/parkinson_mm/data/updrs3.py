"""Nettoyage de la cible : MDS-UPDRS partie III (score total NP3, 0 à 132)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import parse_month
from .journal import CleaningLog
from .updrs3_items import ALL_ITEMS, SUBSCORES, UNRATABLE


def clean_updrs3(u: pd.DataFrame, patients: pd.Series, log: CleaningLog, max_missing: int = 3) -> pd.DataFrame:
    u = u[u["PATNO"].isin(patients)]
    remote = u["EVENT_ID"].astype(str).str.startswith("R")
    log.add("Visites à distance exclues", u, u[~remote], "Visites R## : pas d'examen moteur possible.")
    u = u[~remote].copy()

    items = u[ALL_ITEMS].apply(pd.to_numeric, errors="coerce")
    u.attrs["n_code_101"] = int((items == UNRATABLE).sum().sum())
    items = items.where((items >= 0) & (items <= 4))   # 101 et valeurs hors échelle -> manquant
    n_valid = items.notna().sum(axis=1)
    total = pd.to_numeric(u["NP3TOT"], errors="coerce")
    # Total fourni > recalculé (33 items) > proratisé (<= max_missing items manquants) > rejeté
    n_items = len(ALL_ITEMS)
    prorated = (items.sum(axis=1) * n_items / n_valid).round().where(n_valid >= n_items - max_missing)
    u["np3"] = total.fillna(prorated)
    u["np3_imputed"] = (total.isna() & u["np3"].notna()).astype(int)
    u[ALL_ITEMS], u["n_items_valid"] = items, n_valid
    before = u
    u = u[u["np3"].between(0, 132)].copy()
    log.add("NP3 valide", before, u, f"101 = non évaluable ; proratisation si <= {max_missing} items manquants.")

    for name, cols in SUBSCORES.items():                # sous-scores seulement si complets
        u[name] = u[cols].sum(axis=1, min_count=len(cols))
    u["asym3"] = (u["right3"] - u["left3"]).abs()
    u["nhy"] = pd.to_numeric(u["NHY"], errors="coerce").where(lambda s: s.between(0, 5))
    return add_medication_state(u)


def add_medication_state(u: pd.DataFrame) -> pd.DataFrame:
    """État de l'examen : non traité, OFF (avant la dose) ou ON (après la dose)."""
    treated = pd.to_numeric(u["PDTRTMNT"], errors="coerce")
    state = u["PDSTATE"].astype("string").fillna("")
    u["med_state"] = np.select([(treated == 0).to_numpy(), (state == "OFF").to_numpy(), (state == "ON").to_numpy()],
                               ["untreated", "OFF", "ON"], default="unknown")
    u["on_treatment"] = (treated == 1).astype(float).where(treated.notna())
    u["visit_date"] = parse_month(u["INFODT"])
    return u
