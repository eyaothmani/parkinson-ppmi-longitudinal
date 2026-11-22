"""Variables fixes au niveau patient : démographie, antécédents familiaux, génétique."""
from __future__ import annotations

import numpy as np
import pandas as pd


def carrier(values: pd.Series) -> pd.Series:
    """1 = porteur d'un variant, 0 = non porteur ('0' dans PPMI), NaN = non testé."""
    s = values.astype("string").str.strip()
    return pd.Series(np.where(s.isna(), np.nan, np.where(s.isin(["0", "0.0"]), 0.0, 1.0)), index=values.index)


def patient_table(tables: dict, cohort: pd.DataFrame) -> pd.DataFrame:
    p = cohort[["PATNO", "ENROLL_AGE"]].merge(
        tables["demographics"].drop_duplicates("PATNO")[["PATNO", "SEX"]], on="PATNO", how="left")
    p = p.rename(columns={"SEX": "male"})  # PPMI : 1 = homme, 0 = femme
    education = tables["socio"].groupby("PATNO", as_index=False)["EDUCYRS"].max()
    p = p.merge(education, on="PATNO", how="left")
    p["EDUCYRS"] = p["EDUCYRS"].where(p["EDUCYRS"].between(0, 30))
    p = p.merge(tables["family"].groupby("PATNO", as_index=False)["ANYFAMPD"].max(), on="PATNO", how="left")
    return p.merge(genetics(tables), on="PATNO", how="left")


def genetics(tables: dict) -> pd.DataFrame:
    g = tables["genetics"].drop_duplicates("PATNO")
    out = pd.DataFrame({"PATNO": g["PATNO"], "lrrk2": carrier(g["LRRK2"]), "gba": carrier(g["GBA"]),
                        "snca": carrier(g["SNCA"]), "PATHVAR_COUNT": g["PATHVAR_COUNT"]})
    apoe = g["APOE"].astype("string")
    out["apoe4"] = np.where(apoe.isna(), np.nan, apoe.str.contains("E4").astype(float))
    prs = tables["prs"][["PATNO", "GP2_PGS"]].drop_duplicates("PATNO").rename(columns={"GP2_PGS": "prs"})
    return out.merge(prs, on="PATNO", how="left")
