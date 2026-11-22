"""Contrat des échantillons multi-horizon (ADR-0018) : cibles futures, dans leur fenêtre, sans fuite entre patients."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .checks import Check


def horizon_checks(long: pd.DataFrame, months, tolerance: int) -> list[Check]:
    def in_window(s: pd.DataFrame) -> bool:
        return all((((s[f"gap_{h}"] * 12 - h).abs() <= tolerance + 1e-6) | s[f"y_{h}"].isna()).all() for h in months)

    def history_is_past(s: pd.DataFrame) -> bool:
        previous = long.sort_values(["PATNO", "visit_date"]).groupby("PATNO")["np3"].shift().to_numpy()
        expected = previous[s["row_in_long"].to_numpy()]
        return np.allclose(np.nan_to_num(s["np3_prev"], nan=-1), np.nan_to_num(expected, nan=-1))

    return [
        Check("aucun patient à la fois en dev et en test", lambda s: s.groupby("PATNO")["split"].nunique().max() == 1),
        Check("un patient = un seul pli", lambda s: s.groupby("PATNO")["fold"].nunique().max() == 1),
        Check("au moins une cible par origine", lambda s: s[[f"y_{h}" for h in months]].notna().any(axis=1).all()),
        Check("cible dans la fenêtre de son horizon (donc future)", in_window),
        Check("historique calculé sur le passé uniquement", history_is_past),
    ]
