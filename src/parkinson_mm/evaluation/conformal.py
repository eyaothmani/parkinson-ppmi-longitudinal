"""Sprint 6, analyse 2 : l'intervalle à 80 % contient-il vraiment 80 % des scores réels ?

Prédiction conforme par validation croisée : pour les exemples du pli k, la demi-largeur de l'intervalle est le
quantile 80 % des erreurs absolues des AUTRES plis. L'erreur d'un exemple ne sert jamais à son propre intervalle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_conformal(oof: pd.DataFrame, level: float = 0.8) -> pd.DataFrame:
    error = (oof["y"] - oof["pred"]).abs()
    width = pd.Series(np.nan, index=oof.index)
    for fold in oof["fold"].unique():
        own = oof["fold"] == fold
        others = error[~own].to_numpy()
        adjusted = min(1.0, np.ceil((len(others) + 1) * level) / len(others))   # correction « n + 1 » classique
        width[own] = np.quantile(others, adjusted, method="higher")
    return oof.assign(demi_largeur=width, couvert=error <= width)


def coverage(data: pd.DataFrame, criteria) -> pd.DataFrame:
    """Couverture (part des scores réels dans l'intervalle) et demi-largeur moyenne : globale, puis par groupe."""
    rows = [{"critère": "Tous", "groupe": "tous", "exemples": len(data), "couverture": data["couvert"].mean(),
             "demi_largeur": data["demi_largeur"].mean()}]
    for criterion in criteria:
        for group, part in data.dropna(subset=[criterion]).groupby(criterion, observed=True):
            rows.append({"critère": criterion, "groupe": group, "exemples": len(part),
                         "couverture": part["couvert"].mean(), "demi_largeur": part["demi_largeur"].mean()})
    return pd.DataFrame(rows)
