"""Contrat sur les données brutes : les colonnes dont dépend le code sont présentes et lisibles."""
from __future__ import annotations

import pandas as pd

from ..tables import REQUIRED
from .checks import Check

COLUMNS = {
    "status": ["PATNO", "COHORT_DEFINITION", "ENROLL_STATUS"],
    "updrs3": ["PATNO", "EVENT_ID", "INFODT", "PDTRTMNT", "PDSTATE", "NP3TOT", "NHY"],
    "datscan": ["PATNO", "DATSCAN_DATE", "DATSCAN_ANALYZED", "DATSCAN_LIGAND", "PUTAMEN_L_REF_CWM"],
    "genetics": ["PATNO", "LRRK2", "GBA", "APOE"],
    "ledd": ["PATNO", "LEDTRT", "LEDD", "STARTDT", "STOPDT"],
    "biospecimen": ["PATNO", "CLINICAL_EVENT", "TESTNAME", "TESTVALUE", "PROJECTID", "UNITS"],
}


def raw_checks(tables: dict[str, pd.DataFrame]) -> list[Check]:
    checks = [Check(f"table présente : {key}", lambda k=key: k in tables) for key in REQUIRED]
    for key, cols in COLUMNS.items():
        if key in tables:
            checks.append(Check(f"colonnes de {key}", lambda k=key, c=cols: set(c) <= set(tables[k].columns)))
    if "updrs3" in tables:
        dates = tables["updrs3"]["INFODT"]
        checks.append(Check("dates MM/YYYY lisibles (>= 99 %)",
                            lambda: pd.to_datetime(dates, format="%m/%Y", errors="coerce").notna().mean() >= 0.99))
    return checks
