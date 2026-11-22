"""Métriques de régression. La MAE est la métrique principale (en points UPDRS III, lisible par un clinicien).

On rapporte aussi le R² sur la VARIATION : le R² sur le score est artificiellement élevé, car le score
d'aujourd'hui explique déjà l'essentiel du score de demain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HORIZONS = ([0, 6, 12, 24], ["0-6m", "6-12m", "12-24m"])


def regression_metrics(y, y_last, pred) -> dict:
    y, y_last, pred = (np.asarray(a, dtype=float) for a in (y, y_last, pred))
    err, change, change_pred = pred - y, y - y_last, pred - y_last
    mae, mae_persistence = np.abs(err).mean(), np.abs(change).mean()
    spread = ((change - change.mean()) ** 2).sum() or np.nan   # variation constante : R² sur Δ non défini
    return {"MAE": float(mae), "RMSE": float(np.sqrt((err ** 2).mean())),
            "R2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "R2_delta": float(1 - ((change - change_pred) ** 2).sum() / spread),
            "spearman": float(spearmanr(y, pred).statistic) if len(y) > 2 else np.nan,
            "skill_vs_persistance_%": float(100 * (1 - mae / mae_persistence)), "n": int(len(y))}


def _metrics_frame(group: pd.DataFrame) -> pd.Series:
    return pd.Series(regression_metrics(group["y"], group["y_last"], group["pred"]))


def summarize_cv(oof: pd.DataFrame) -> pd.DataFrame:
    """Métriques par pli, puis moyenne et écart-type entre plis (variabilité de l'estimation)."""
    per_fold = oof.groupby(["model", "family", "fold"]).apply(_metrics_frame, include_groups=False).reset_index()
    summary = per_fold.groupby(["model", "family"]).agg(
        MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), RMSE=("RMSE", "mean"), R2=("R2", "mean"),
        R2_delta=("R2_delta", "mean"), spearman=("spearman", "mean"), skill_pct=("skill_vs_persistance_%", "mean"))
    return summary.reset_index().sort_values("MAE").round(4)


def by_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE selon l'écart entre la visite courante et la visite prédite."""
    edges, labels = HORIZONS
    tagged = predictions.assign(horizon=pd.cut(predictions["gap_months"], edges, labels=labels))
    return tagged.groupby(["model", "horizon"], observed=True).apply(_metrics_frame, include_groups=False).reset_index()
