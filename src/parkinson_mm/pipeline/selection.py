"""Sélection des modèles évalués sur le test : décidée sur la seule validation croisée."""
from __future__ import annotations

import pandas as pd


def choose_models(cfg, cv: pd.DataFrame) -> tuple[list[str], str]:
    """Sélection sur la seule validation croisée : modèles imposés + meilleur ML + meilleur DL."""
    best = {family: cv.loc[cv["family"] == family, "model"].iloc[0] for family in ("ml", "dl")
            if (cv["family"] == family).any()}
    learned = cv[cv["family"] != "baseline"].iloc[0]["model"]
    chosen = list(dict.fromkeys([*cfg.evaluation.always_include, *best.values(), learned]))
    return [c for c in chosen if c in set(cv["model"])], learned
