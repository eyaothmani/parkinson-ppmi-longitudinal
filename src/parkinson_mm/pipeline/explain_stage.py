"""Étape « explain » : importance des groupes de variables du modèle retenu, mesurée sur le test."""
from __future__ import annotations

import numpy as np

from ..explainability.permutation import permutation_importance
from ..serving.predictor import NextVisitPredictor
from ..utils.io import write_json
from .common import load_modeling_data, logger, tables_dir


def explain(cfg) -> dict:
    samples, X, mask = load_modeling_data(cfg)
    test = np.where(samples["split"] == "test")[0]
    predictor = NextVisitPredictor.load(cfg.path("models") / "predictor.pkl")
    importance = permutation_importance(predictor.model, samples.iloc[test].reset_index(drop=True), X[test],
                                        mask[test], cfg.explain.n_repeats, cfg.project.seed)
    importance.to_csv(tables_dir(cfg, "models") / "importance_permutation.csv", index=False)
    result = {"model": predictor.metadata.get("model"), "importance": importance.round(4).to_dict("records")}
    write_json(result, cfg.path("metrics") / "explain.json")
    logger(cfg, "explain").info("groupe le plus utile : %s", importance.iloc[0]["groupe"])
    return result
