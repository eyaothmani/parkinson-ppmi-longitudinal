"""Score d'évolution individuel = pente de l'UPDRS III (points/an) estimée par le modèle mixte.

Pente = effet fixe + effet aléatoire du patient (BLUP). Les BLUP sont « rétrécis » vers la moyenne :
un patient avec peu de visites bruitées reçoit une pente prudente, proche de celle de la cohorte.
"""
from __future__ import annotations

import pandas as pd


def individual_slopes(result, d: pd.DataFrame) -> pd.DataFrame:
    random = pd.DataFrame(result.random_effects).T
    slope_col = next(c for c in random.columns if "time" in c)
    out = random.rename(columns={slope_col: "b_slope", "Group": "b_intercept"})
    out.index.name = "PATNO"
    out = out.reset_index()
    profile = d.drop_duplicates("PATNO").set_index("PATNO")
    fixed = pd.Series(result.params.get("time_years", 0.0), index=profile.index)
    for term, value in result.params.items():          # interactions temps × covariable du patient
        variable = term.split(":", 1)[1] if term.startswith("time_years:") else None
        if variable in profile:
            fixed = fixed + value * profile[variable].astype(float)
    out["score_evolution"] = out["PATNO"].map(fixed) + out["b_slope"]
    out["n_visits"] = out["PATNO"].map(d.groupby("PATNO").size())
    out["followup_years"] = out["PATNO"].map(d.groupby("PATNO")["time_years"].max())
    return out
