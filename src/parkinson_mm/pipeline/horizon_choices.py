"""Choix et données annexes de l'évaluation multi-horizon, fixés AVANT de lire le test (ADR-0018)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..training.horizon_cv import horizon_specs
from .common import tables_dir


def cv_choices(cfg) -> dict:
    """Sur la seule validation croisée, à l'horizon principal : meilleur modèle tabulaire (LightGBM ou MLP, pour H1)
    et modèle retenu pour le registry (meilleur modèle appris, ADR-0004)."""
    cv = pd.read_csv(tables_dir(cfg, "horizons") / "cv_summary.csv")
    at = cv[cv["horizon"] == cfg.horizons.primary].set_index("model")
    specs = horizon_specs(cfg, ablations=False)
    tabular = min((s.name for s in specs if s.algo in ("lgbm", "mlp")), key=lambda n: at.loc[n, "MAE"])
    selected = min((s.name for s in specs if s.family != "baseline"), key=lambda n: at.loc[n, "MAE"])
    return {"meilleur_tabulaire": tabular, "retenu": selected, "retenu_cv_MAE": float(at.loc[selected, "MAE"]),
            "retenu_cv_MAE_sd": float(at.loc[selected, "MAE_sd"]),
            "dl": next(s.name for s in specs if s.algo == "grud"),
            "persistance": next(s.name for s in specs if s.family == "baseline")}


def speeds(cfg, samples: pd.DataFrame, test, quantiles: np.ndarray) -> pd.DataFrame:
    """H5 : vitesse du GRU-D depuis la première visite de chaque patient de test, (médiane au plus long horizon −
    médiane au plus court) / écart en années, face à la pente individuelle du modèle mixte (objectif 1)."""
    months = list(cfg.horizons.months)
    first = (samples.iloc[test][["PATNO", "time_years"]].reset_index(drop=True).rename_axis("i").reset_index()
             .sort_values(["PATNO", "time_years"]).drop_duplicates("PATNO"))
    longest, shortest = months.index(max(months)), months.index(min(months))
    years = (max(months) - min(months)) / 12
    speed = (quantiles[first["i"], longest, 1] - quantiles[first["i"], shortest, 1]) / years
    slopes = pd.read_csv(cfg.path("processed") / "pd_score_evolution.csv")[["PATNO", "score_evolution"]]
    return (pd.DataFrame({"PATNO": first["PATNO"].to_numpy(), "vitesse_dl": speed})
            .merge(slopes.rename(columns={"score_evolution": "pente_modele_mixte"}), on="PATNO"))
