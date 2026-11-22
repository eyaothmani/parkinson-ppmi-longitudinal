"""Dérive des entrées : Population Stability Index (PSI) entre production et entraînement.
Repères usuels : < 0,1 stable ; 0,1–0,25 à surveiller ; > 0,25 dérive importante (ADR-0009)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def reference_quantiles(train: pd.DataFrame, columns: list[str], bins: int = 10) -> dict:
    """Déciles d'entraînement, stockés avec le modèle pour comparer plus tard."""
    out = {}
    for col in columns:
        values = pd.to_numeric(train.get(col), errors="coerce").dropna() if col in train else pd.Series(dtype=float)
        if len(values) > bins:
            out[col] = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1))).tolist()
    return out


def psi(values, edges: list[float], eps: float = 1e-4) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0 or len(edges) < 3:
        return float("nan")
    inner = np.asarray(edges[1:-1])
    expected = np.full(len(inner) + 1, 1 / (len(inner) + 1))   # chaque décile contient 10 % à l'entraînement
    actual = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1) / len(values)
    expected, actual = np.clip(expected, eps, None), np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
