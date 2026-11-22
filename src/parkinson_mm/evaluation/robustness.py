"""Sprint 6, analyse 1 : où le modèle se trompe-t-il ? MAE par sous-groupe, sur les prédictions HORS-PLI de la
validation croisée (le test, déjà utilisé une fois, n'est pas relu), avec IC bootstrap sur les patients."""
from __future__ import annotations

import pandas as pd

from .bootstrap import mae_ci


def with_groups(oof: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Ajoute à chaque prédiction hors-pli les groupes de son échantillon (sample_id = ligne des échantillons)."""
    return oof.join(groups, on="sample_id")


def subgroup_errors(data: pd.DataFrame, criteria, models, n_boot: int, seed: int) -> pd.DataFrame:
    """Une ligne par (critère, groupe, modèle) : effectifs, MAE et son IC 95 %."""
    rows = []
    for criterion in criteria:
        for group, part in data.dropna(subset=[criterion]).groupby(criterion, observed=True):
            for model in models:
                d = part[part["model"] == model]
                low, high = mae_ci(d, model, n_boot, seed)
                rows.append({"critère": criterion, "groupe": group, "modèle": model, "exemples": len(d),
                             "patients": d["PATNO"].nunique(), "MAE": (d["pred"] - d["y"]).abs().mean(),
                             "IC95_bas": low, "IC95_haut": high})
    return pd.DataFrame(rows)
