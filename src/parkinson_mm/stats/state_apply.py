"""Application du score d'état : état actuel de chaque patient, et prévision de la visite suivante.

La prévision sert à VALIDER le score d'état (objectif 1) et de base statistique à l'objectif 2 :
si le niveau et la vitesse estimés décrivent bien le patient, ils doivent prédire sa prochaine mesure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lme import BASE_ADJUSTMENT, fit_lme
from .state import StateModel

COVARIATES = ["age_c", "male", "dur_c"]


def fit_state_model(d_dev: pd.DataFrame) -> StateModel:
    """Modèle mixte de base ajusté sur les patients du DEV uniquement (le test reste intact)."""
    result = fit_lme("np3 ~ time_years + " + BASE_ADJUSTMENT, d_dev)
    return StateModel(result, d_dev[COVARIATES].median().to_dict())


def histories(d: pd.DataFrame) -> dict[int, pd.DataFrame]:
    usable = d.dropna(subset=["np3", "time_years", "med"])
    return {patno: g.sort_values("time_years") for patno, g in usable.groupby("PATNO")}


def current_states(model: StateModel, d: pd.DataFrame) -> pd.DataFrame:
    """Score d'état de chaque patient à sa dernière visite (niveau, vitesse et IC 95 %)."""
    rows = [{"PATNO": patno, "n_visits": len(h), **model.state(h)} for patno, h in histories(d).items()]
    out = pd.DataFrame(rows)
    for name, value in (("niveau", "niveau_actuel"), ("vitesse", "vitesse")):
        half_width = 1.96 * out[f"{name}_sd"]
        out[f"{name}_ic_bas"], out[f"{name}_ic_haut"] = out[value] - half_width, out[value] + half_width
    return out


def forecast_samples(model: StateModel, d: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Visite i -> i+1 : historique jusqu'à i (inclus), état médicamenteux prévu à i+1."""
    groups, rows = histories(d), []
    for sample_id, s in samples.iterrows():
        h = groups[s["PATNO"]]
        h = h[h["time_years"] <= s["time_years"] + 1e-9]
        target = h.tail(1).assign(time_years=s["time_years"] + s["gap_years"],
                                  med="ON" if s["target_state"] == "unknown" else s["target_state"])
        mean, sd = model.predict(h, target, noise=True)
        rows.append({"sample_id": sample_id, "PATNO": s["PATNO"], "y": s["y_next"], "y_last": s["np3"],
                     "gap_months": s["gap_months"], "pred": float(np.clip(mean[0], 0, 132)), "pred_sd": float(sd[0])})
    return pd.DataFrame(rows)
