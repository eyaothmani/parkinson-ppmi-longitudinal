"""Baseline et modèle tabulaire. Les modèles appris prédisent la VARIATION Δ = NP3(i+1) − NP3(i)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def persistence(samples: pd.DataFrame) -> np.ndarray:
    """« Rien ne change » : le plancher que tout modèle doit battre."""
    return samples["np3"].to_numpy(dtype=float)


def make_tabular(algo: str, seed: int = 42, **tuned):
    """`tuned` : réglages trouvés par la recherche aléatoire du sprint 5 (ADR-0018) ; aucun pour l'objectif 2."""
    if algo == "lgbm":
        from lightgbm import LGBMRegressor

        # Perte de Huber : robuste aux quelques sauts extrêmes conservés dans les données
        return LGBMRegressor(**{"n_estimators": 600, "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 40,
                                "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "objective": "huber",
                                "alpha": 5.0, "random_state": seed, "verbose": -1, **tuned})
    raise ValueError(f"Algorithme tabulaire inconnu : {algo} (ADR-0015 : lgbm)")
