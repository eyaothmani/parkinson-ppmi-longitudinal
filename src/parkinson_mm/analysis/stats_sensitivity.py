"""Sensibilité de l'objectif 3 (toute la cohorte, pas de prédiction) : forme du temps, résultat inattendu H&Y ≥ 2,
désaccord GBA avec la littérature, et vitesse des patients de novo (comparable à Holden et al. 2018)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from ..stats.lme import BASE_ADJUSTMENT, fit_lme, tidy
from ..stats.screening import adjustment_for

UNMEDICATED = ["untreated", "OFF"]        # examens sans effet symptomatique du traitement


def time_shape(d: pd.DataFrame) -> dict:
    """Le temps linéaire suffit-il ? Comparaison au modèle quadratique (maximum de vraisemblance, test du rapport)."""
    linear = fit_lme("np3 ~ time_years + " + BASE_ADJUSTMENT, d, reml=False)
    quad = fit_lme("np3 ~ time_years + I(time_years ** 2) + " + BASE_ADJUSTMENT, d, reml=False)
    b1, b2 = quad.params["time_years"], quad.params["I(time_years ** 2)"]
    return {"AIC_lineaire": float(linear.aic), "AIC_quadratique": float(quad.aic), "terme_quadratique": float(b2),
            "LRT_p": float(chi2.sf(2 * (quad.llf - linear.llf), 1)), "pente_an_1": float(b1 + 2 * b2),
            "pente_an_5": float(b1 + 10 * b2), "pente_lineaire": float(linear.params["time_years"])}


def summary(label: str, result, term: str) -> dict:
    row = tidy(result).loc[term]
    return {"analyse": label, "effet_pts_an": float(row["coef"]), "IC95_bas": float(row["ci_low"]),
            "IC95_haut": float(row["ci_high"]), "p": float(row["p"]), "n_patients": int(result.n_patients_),
            "converge": bool(result.converged)}


def factor_effect(d: pd.DataFrame, label: str, variable: str, extra: str = "") -> dict:
    """Effet du facteur sur la vitesse (interaction temps × facteur), ajusté comme le criblage (+ `extra`)."""
    formula = f"np3 ~ time_years * {variable} + {adjustment_for(variable)}{extra}"
    return summary(label, fit_lme(formula, d.dropna(subset=[variable])), f"time_years:{variable}")


def rows(d: pd.DataFrame) -> list[dict]:
    first = d.drop_duplicates("PATNO").set_index("PATNO")["np3_bl"]
    d = d.assign(z_np3_bl=d["PATNO"].map((first - first.mean()) / first.std()))
    off = d[d["med"].isin(UNMEDICATED)]
    novo_ids = d.sort_values("time_years").drop_duplicates("PATNO").query("med == 'untreated'")["PATNO"]
    novo = d[d["PATNO"].isin(novo_ids) & (d["time_years"] <= 5)]
    plan = [("H&Y ≥ 2 : analyse principale", d, "hy2_bl", ""),
            ("H&Y ≥ 2 : + vitesse ajustée sur le score initial", d, "hy2_bl", " + time_years * z_np3_bl"),
            ("H&Y ≥ 2 : examens non traités ou OFF seulement", off, "hy2_bl", ""),
            ("GBA : analyse principale", d, "gba", ""), ("GBA : examens non traités ou OFF seulement", off, "gba", "")]
    out = [factor_effect(data, label, var, extra) for label, data, var, extra in plan if data[var].nunique() > 1]
    slopes = {"De novo, 5 ans : vitesse ajustée (âge, sexe, durée, état)": "np3 ~ time_years + " + BASE_ADJUSTMENT,
              "De novo, 5 ans : vitesse brute": "np3 ~ time_years"}
    out += [summary(label, fit_lme(formula, novo), "time_years") for label, formula in slopes.items()]
    return out + [{"analyse": "Holden et al. 2018 (référence publiée, de novo PPMI)", "effet_pts_an": 2.4,
                   "IC95_bas": np.nan, "IC95_haut": np.nan, "p": np.nan, "n_patients": 362, "converge": True}]
