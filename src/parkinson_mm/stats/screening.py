"""Criblage des facteurs associés à la vitesse de progression, avec correction des tests multiples.

Stratégie fixée à l'avance dans params.yaml (stats.factors) : un modèle par facteur, ajusté de la même façon,
un seul test par facteur (interaction temps × facteur), correction de Benjamini-Hochberg sur l'ensemble.
"""
from __future__ import annotations

import pandas as pd
from statsmodels.stats.multitest import multipletests

from .lme import BASE_ADJUSTMENT, fit_lme, tidy


def adjustment_for(variable: str) -> str:
    """Ajustement de base sans le facteur testé lui-même (ni sa version recodée, ex. age10 / age_c)."""
    terms = [t.strip() for t in BASE_ADJUSTMENT.split("+")]
    prefix = variable.split("_")[0][:3]
    return " + ".join(t for t in terms if t != variable and not (prefix in ("age", "dur") and t.startswith(prefix)))


def screen_factors(d: pd.DataFrame, factors: dict, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for label, variable in factors.items():
        if variable not in d or d[variable].nunique() < 2:   # facteur absent de ces données : ignoré
            continue
        data = d.dropna(subset=[variable])
        result = fit_lme(f"np3 ~ time_years * {variable} + {adjustment_for(variable)}", data)
        slope = tidy(result).loc[f"time_years:{variable}"]
        rows.append({"facteur": label, "variable": variable, "n_patients": result.n_patients_,
                     "effet_pente_pts_an": slope["coef"], "IC95_bas": slope["ci_low"],
                     "IC95_haut": slope["ci_high"], "p": slope["p"], "converge": bool(result.converged)})
    table = pd.DataFrame(rows)
    table["p_FDR"] = multipletests(table["p"], method="fdr_bh")[1]
    table["significatif_FDR"] = table["p_FDR"] < alpha
    return table.sort_values("p").reset_index(drop=True)
