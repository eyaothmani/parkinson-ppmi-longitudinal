"""Biomarqueurs à la baseline. UN seul projet de laboratoire par marqueur : mélanger des dosages
de laboratoires différents reviendrait à mélanger des échelles différentes."""
from __future__ import annotations

import numpy as np
import pandas as pd

# (nom du test, projet PPMI, filtre d'unité, variable, passage au log)
MARKERS = [("ABeta42", 283, None, "csf_abeta42", False), ("ABeta40", 283, None, "csf_abeta40", False),
           ("pTau181", 283, None, "csf_ptau181", True), ("CSF Alpha-synuclein", 124, None, "csf_asyn", False),
           ("NFL", 283, "Average", "plasma_nfl", True), ("GFAP", 283, "Average", "plasma_gfap", True)]


def censored_to_float(values: pd.Series) -> pd.Series:
    """'<8' (sous la limite de détection) -> 4 ; '>1997' (au-dessus) -> 1997."""
    s = values.astype("string").str.strip()
    below = pd.to_numeric(s.str.extract(r"^<\s*([0-9.]+)")[0], errors="coerce") / 2
    above = pd.to_numeric(s.str.extract(r"^>\s*([0-9.]+)")[0], errors="coerce")
    return pd.to_numeric(s, errors="coerce").fillna(below).fillna(above)


def baseline_markers(bio: pd.DataFrame | None) -> pd.DataFrame | None:
    if bio is None:
        return None
    b = bio[bio["CLINICAL_EVENT"].isin(["BL", "SC"])].assign(value=lambda d: censored_to_float(d["TESTVALUE"]))
    columns = []
    for test, project, unit, name, use_log in MARKERS:
        sel = b[(b["TESTNAME"] == test) & (pd.to_numeric(b["PROJECTID"], errors="coerce") == project)]
        if unit:
            sel = sel[sel["UNITS"].astype(str).str.contains(unit, case=False)]
        median = sel[sel["value"] > 0].groupby("PATNO")["value"].median()   # réplicats -> médiane
        columns.append((np.log(median) if use_log else median).rename(name))
    out = pd.concat(columns, axis=1)
    out["csf_ab42_40"] = out["csf_abeta42"] / out.pop("csf_abeta40")   # le ratio est plus robuste que Aβ42 seul
    return out.reset_index()
