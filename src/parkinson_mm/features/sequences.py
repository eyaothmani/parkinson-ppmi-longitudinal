"""Séquences de visites pour les modèles récurrents : [échantillons, visites, variables], alignées à droite."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .history import add_step_encodings


def build_sequences(long: pd.DataFrame, samples: pd.DataFrame, columns: list[str],
                    max_len: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Pour chaque échantillon, les `max_len` dernières visites <= i. La dernière case = visite courante.
    Les NaN sont conservés : ils seront imputés avec les statistiques du train. `dt_prev` encode l'irrégularité."""
    v = add_step_encodings(long).sort_values(["PATNO", "visit_date"]).reset_index(drop=True)
    v["dt_prev"] = v.groupby("PATNO")["time_years"].diff().fillna(0.0)
    values = v.reindex(columns=[*columns, "dt_prev"]).to_numpy(np.float32)   # colonne absente -> NaN
    position = v.groupby("PATNO").cumcount().to_numpy()       # rang de la visite chez son patient
    X = np.full((len(samples), max_len, values.shape[1]), np.nan, dtype=np.float32)
    mask = np.zeros((len(samples), max_len), dtype=np.float32)
    for k, row in enumerate(samples["row_in_long"].to_numpy()):
        start = max(row - position[row], row - max_len + 1)   # ne jamais déborder sur le patient précédent
        seq = values[start: row + 1]
        X[k, max_len - len(seq):] = seq
        mask[k, max_len - len(seq):] = 1.0
    return X, mask
