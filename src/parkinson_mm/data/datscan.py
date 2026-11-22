"""Imagerie DaTSCAN : ratios de liaison striatale (SBR) par visite et à la baseline."""
from __future__ import annotations

import pandas as pd

from .io import months_between, parse_month
from .journal import CleaningLog

FEATURES = ["sbr_striatum", "sbr_putamen", "sbr_putamen_min", "sbr_caudate", "sbr_put_caud_ratio", "sbr_put_asym"]


def clean_scans(d: pd.DataFrame, patients: pd.Series, log: CleaningLog) -> pd.DataFrame:
    d = d[d["PATNO"].isin(patients)]
    ok = d[(d["DATSCAN_ANALYZED"] == "Yes") & (d["DATSCAN_LIGAND"] == "123I-DaTscan")].copy()
    log.add("DaTSCAN analysés, ligand 123I", d, ok, "Le TRODAT n'est pas sur la même échelle SBR.")
    ok["scan_date"] = parse_month(ok["DATSCAN_DATE"])
    put = ok[["PUTAMEN_L_REF_CWM", "PUTAMEN_R_REF_CWM"]]
    ok["sbr_striatum"] = ok["STRIATUM_REF_CWM"]
    ok["sbr_putamen"] = put.mean(axis=1)
    ok["sbr_putamen_min"] = put.min(axis=1)                   # côté le plus atteint
    ok["sbr_caudate"] = ok[["CAUDATE_L_REF_CWM", "CAUDATE_R_REF_CWM"]].mean(axis=1)
    ok["sbr_put_caud_ratio"] = ok["sbr_putamen"] / ok["sbr_caudate"].where(ok["sbr_caudate"] > 0.05)
    ok["sbr_put_asym"] = (put.iloc[:, 0] - put.iloc[:, 1]).abs() / ok["sbr_putamen"].where(ok["sbr_putamen"] > 0.05)
    ok = ok.dropna(subset=["scan_date", "sbr_putamen"])
    ok = ok[["PATNO", "scan_date", *FEATURES]].sort_values(["PATNO", "scan_date"])
    return ok.drop_duplicates(["PATNO", "scan_date"])


def merge_last_scan(v: pd.DataFrame, scans: pd.DataFrame) -> pd.DataFrame:
    """À chaque visite, le DERNIER scan déjà réalisé : jamais un scan futur (aucune fuite d'information)."""
    merged = pd.merge_asof(v.sort_values("visit_date"), scans.sort_values("scan_date"), left_on="visit_date",
                           right_on="scan_date", by="PATNO", direction="backward")
    merged["sbr_age_months"] = months_between(merged["scan_date"], merged["visit_date"])
    return merged.drop(columns="scan_date").sort_values(["PATNO", "visit_date"]).reset_index(drop=True)


def baseline_scan(scans: pd.DataFrame, bl_dates: pd.Series, window_months: int = 6) -> pd.DataFrame:
    """SBR de référence : premier scan à +/- `window_months` de la baseline."""
    s = scans.merge(bl_dates.rename("bl_date").reset_index(), on="PATNO")
    s = s[months_between(s["bl_date"], s["scan_date"]).abs() <= window_months]
    first = s.sort_values("scan_date").drop_duplicates("PATNO")[["PATNO", *FEATURES]]
    return first.rename(columns={c: f"{c}_bl" for c in FEATURES})
