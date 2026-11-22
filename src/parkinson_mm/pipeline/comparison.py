"""Objectif 2 : comparer les quatre modèles ENTRE EUX, au-delà de « bat la persistance ».

1. Différences appariées de MAE entre modèles appris, en validation croisée et sur le test (IC corrigé de Bonferroni).
2. Importance par permutation de groupes pour CHAQUE modèle : utilisent-ils la même information ?
"""
from __future__ import annotations

import pandas as pd

from ..evaluation.bootstrap import pairwise
from ..explainability.permutation import permutation_importance
from .common import tables_dir


def compare(cfg, fitted: dict, predictions: pd.DataFrame, samples, X, mask, test) -> dict:
    folder, n_boot = tables_dir(cfg, "models"), cfg.evaluation.bootstrap_samples
    learned = [name for name, model in fitted.items() if model.spec.family != "baseline"]
    oof = pd.read_csv(folder / "cv_predictions.csv")
    tables = {"cv": pairwise(oof[oof["model"].isin(learned)], learned, n_boot),
              "test": pairwise(predictions, learned, n_boot)}
    for split, table in tables.items():
        table.to_csv(folder / f"{split}_comparaisons_appariees.csv", index=False)
    held = samples.iloc[test].reset_index(drop=True)
    importance = pd.concat([permutation_importance(model, held, X[test], mask[test], cfg.explain.n_repeats,
                                                   cfg.project.seed).assign(modele=name)
                            for name, model in fitted.items()], ignore_index=True)
    importance.to_csv(folder / "importance_par_modele.csv", index=False)
    return {split: table.round(4).to_dict("records") for split, table in tables.items()}
