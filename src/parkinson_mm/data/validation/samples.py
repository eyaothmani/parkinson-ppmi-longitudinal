"""Contrat sur les échantillons de modélisation : absence de fuite entre patients et entre passé et futur."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .checks import Check


def samples_checks(long: pd.DataFrame) -> list[Check]:
    def history_is_past(s: pd.DataFrame) -> bool:
        # np3_prev doit être exactement le score de la visite précédente du même patient
        previous = long.sort_values(["PATNO", "visit_date"]).groupby("PATNO")["np3"].shift().to_numpy()
        expected = previous[s["row_in_long"].to_numpy()]
        return np.allclose(np.nan_to_num(s["np3_prev"], nan=-1), np.nan_to_num(expected, nan=-1))

    return [
        Check("aucun patient à la fois en dev et en test", lambda s: s.groupby("PATNO")["split"].nunique().max() == 1),
        Check("un patient = un seul pli", lambda s: s.groupby("PATNO")["fold"].nunique().max() == 1),
        Check("cible définie", lambda s: s["y_next"].notna().all()),
        Check("écart entre visites positif", lambda s: (s["gap_months"] > 0).all()),
        Check("historique calculé sur le passé uniquement", history_is_past),
    ]
