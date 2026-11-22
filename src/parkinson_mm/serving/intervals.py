"""Intervalle de prédiction empirique PAR HORIZON : l'erreur grandit avec le délai jusqu'à la visite prédite.

Quantiles des résidus de VALIDATION CROISÉE (jamais du test), calculés séparément pour chaque tranche de délai
(0-6, 6-12, 12-24 mois) ; repli sur l'intervalle global si une tranche a trop peu de résidus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.metrics import HORIZONS

MIN_RESIDUALS = 50


def horizon_of(gap_months) -> pd.Series:
    edges, labels = HORIZONS
    return pd.cut(pd.Series(np.asarray(gap_months, dtype=float)), edges, labels=labels).astype(str)


def fit_intervals(residuals, gap_months, coverage: float = 0.8) -> dict:
    alpha, res = (1 - coverage) / 2, pd.Series(np.asarray(residuals, dtype=float))
    def band(r):
        return float(np.quantile(r, alpha)), float(np.quantile(r, 1 - alpha))
    out = {"global": band(res)}
    for label, group in res.groupby(horizon_of(gap_months).to_numpy()):
        if label != "nan" and len(group) >= MIN_RESIDUALS:
            out[label] = band(group)
    return out


def bounds(intervals, gap_months) -> tuple[np.ndarray, np.ndarray]:
    """Bornes (basse, haute) à ajouter à la prédiction ; ancien format (un seul couple) accepté."""
    if not isinstance(intervals, dict):
        intervals = {"global": tuple(intervals)}
    pairs = [intervals.get(h, intervals["global"]) for h in horizon_of(gap_months)]
    return np.array([p[0] for p in pairs], dtype=float), np.array([p[1] for p in pairs], dtype=float)


def coverage_by_horizon(y, pred, gap_months, intervals) -> pd.DataFrame:
    """Part des vraies valeurs dans l'intervalle, par tranche de délai et au total (à mesurer sur le test)."""
    low, high = bounds(intervals, gap_months)
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    inside = (y >= pred + low) & (y <= pred + high)
    frame = pd.DataFrame({"horizon": horizon_of(gap_months).to_numpy(), "dedans": inside, "largeur": high - low})
    table = frame.groupby("horizon").agg(n=("dedans", "size"), couverture=("dedans", "mean"),
                                         largeur=("largeur", "mean"))
    total = pd.DataFrame({"n": [len(frame)], "couverture": [frame["dedans"].mean()],
                          "largeur": [frame["largeur"].mean()]}, index=["global"])
    return pd.concat([table, total]).rename_axis("horizon").reset_index()
