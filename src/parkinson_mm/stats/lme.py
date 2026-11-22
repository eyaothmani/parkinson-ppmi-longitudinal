"""Modèles linéaires à effets mixtes (LME) pour mesures répétées.

NP3_it = β0 + β1·t + β2·X_i + β3·(t × X_i) + ajustements + b0i + b1i·t + ε_it
β3 est le coefficient d'intérêt : association entre le facteur X et la VITESSE de progression (points/an).
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Ajustement systématique : âge, sexe, durée de maladie à BL (stade) et état médicamenteux de l'examen
BASE_ADJUSTMENT = "age_c + male + dur_c + C(med, Treatment('untreated'))"


def prepare(long: pd.DataFrame, patients: pd.DataFrame, min_visits: int = 2) -> pd.DataFrame:
    d = long.merge(patients.drop(columns=["n_visits"], errors="ignore"), on="PATNO", how="left")
    d = d[d["n_visits"] >= min_visits].copy()
    d["med"] = d["med_state"].replace({"unknown": "ON"})
    d["age_c"] = d["age_bl"] - 62             # intercept = patient de 62 ans à BL
    d["age10"] = d["age_c"] / 10              # effet de l'âge par tranche de 10 ans
    duration = d.get("disease_dur_bl", pd.Series(np.nan, index=d.index))
    d["dur_c"] = duration.fillna(duration.median()).fillna(1.5) - 1.5
    d["ledd100"] = d.get("ledd", np.nan) / 100
    return d


def fit_lme(formula: str, d: pd.DataFrame, re_formula: str = "~time_years", reml: bool = True):
    """Ajuste le modèle sur les lignes complètes pour les variables de la formule ; essaie 3 optimiseurs."""
    used = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)) & set(d.columns)
    data = d.dropna(subset=sorted(used | {"PATNO", "time_years"}))
    result = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("lbfgs", "powell", "nm"):
            try:
                result = smf.mixedlm(formula, data, groups=data["PATNO"], re_formula=re_formula).fit(
                    reml=reml, method=[method])
            except (np.linalg.LinAlgError, ValueError):
                continue
            if result.converged:
                break
    if result is None:
        raise RuntimeError(f"Modèle mixte non ajusté : {formula}")
    if not result.converged:   # aucun optimiseur n'a convergé : on le signale au lieu de le taire
        warnings.warn(f"Modèle mixte non convergé : {formula}", RuntimeWarning, stacklevel=2)
    result.n_obs_, result.n_patients_ = len(data), data["PATNO"].nunique()
    return result


def tidy(result) -> pd.DataFrame:
    """Coefficients fixes avec IC 95 % et p-values (sans les composantes de variance)."""
    ci = result.conf_int()
    table = pd.DataFrame({"coef": result.params, "se": result.bse, "ci_low": ci[0], "ci_high": ci[1],
                          "p": result.pvalues})
    return table.drop(index=[i for i in table.index if "Var" in i or "Cov" in i])
