"""Métriques multi-horizon (ADR-0018) : par modèle et par horizon, sur les seules visites ayant une cible à cet horizon.

En plus de MAE, RMSE et R² : couverture de l'intervalle 10–90 % (modèles à quantiles) et part des prédictions à moins
d'une différence cliniquement importante (±3,25 points de MDS-UPDRS III).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import regression_metrics


def long_format(samples: pd.DataFrame, rows, quantiles: np.ndarray, months, model: str, **extra) -> pd.DataFrame:
    """Une ligne par (visite d'origine, horizon) avec une cible : y, score d'origine, médiane et bornes prédites."""
    rows, parts = np.asarray(rows), []
    for j, h in enumerate(months):
        y = samples[f"y_{h}"].to_numpy(dtype=float)[rows]
        keep = ~np.isnan(y)
        parts.append(pd.DataFrame({
            "PATNO": samples["PATNO"].to_numpy()[rows][keep], "sample_id": rows[keep], "horizon": h, "y": y[keep],
            "y_last": samples["np3"].to_numpy(dtype=float)[rows][keep], "pred": quantiles[keep, j, 1],
            "q_low": quantiles[keep, j, 0], "q_high": quantiles[keep, j, 2], "model": model, **extra}))
    return pd.concat(parts, ignore_index=True)


def horizon_metrics(group: pd.DataFrame, mcid: float) -> pd.Series:
    out = regression_metrics(group["y"], group["y_last"], group["pred"])
    inside = (group["y"] >= group["q_low"]) & (group["y"] <= group["q_high"])
    has_interval = group["q_low"].notna().all()
    out["couverture_10_90"] = float(inside.mean()) if has_interval else np.nan
    out["largeur_10_90"] = float((group["q_high"] - group["q_low"]).mean()) if has_interval else np.nan
    out["part_mcid"] = float(((group["pred"] - group["y"]).abs() <= mcid).mean())
    return pd.Series(out)


def summarize(predictions: pd.DataFrame, mcid: float, by=("model", "horizon")) -> pd.DataFrame:
    return predictions.groupby(list(by)).apply(horizon_metrics, mcid, include_groups=False).reset_index()


def summarize_cv(oof: pd.DataFrame, mcid: float) -> pd.DataFrame:
    """Métriques par pli, puis moyenne et écart-type entre plis, par modèle et par horizon."""
    per_fold = summarize(oof, mcid, ("model", "family", "horizon", "fold"))
    table = per_fold.groupby(["model", "family", "horizon"]).agg(
        MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), RMSE=("RMSE", "mean"), R2=("R2", "mean"),
        R2_delta=("R2_delta", "mean"), couverture_10_90=("couverture_10_90", "mean"), part_mcid=("part_mcid", "mean"))
    return table.reset_index().sort_values(["horizon", "MAE"]).round(4)
