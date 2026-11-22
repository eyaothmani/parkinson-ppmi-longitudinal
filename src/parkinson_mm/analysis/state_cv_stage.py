"""Étape « etat_cv » : l'objectif 1 face aux persistances brute et ajustée, en validation croisée seulement."""
from __future__ import annotations

import mlflow
import pandas as pd

from ..evaluation.bootstrap import pairwise
from ..evaluation.metrics import summarize_cv
from ..pipeline.common import data_version, logger, tables_dir
from ..pipeline.run_datasets import cross_validation
from ..training.tracking import log_artifact, model_run, numeric
from ..utils.io import write_json
from .reference import NAME as ADJUSTED
from .reference import adjusted_persistence
from .state_cv import NAME, state_out_of_fold

BEST = "GRU – trajectoire (M1)"


def state_cv(cfg) -> dict:
    samples = pd.read_csv(cfg.path("processed") / "pd_next_visit_samples.csv")
    base = pd.read_csv(tables_dir(cfg, "models") / "cv_predictions.csv")
    state = state_out_of_fold(cfg, samples, sorted(base["fold"].unique()))
    raw = base[base["model"] == "Persistance"]
    oof = pd.concat([raw, base[base["model"] == BEST], state.drop(columns=["pred_sd", "inside_80"]),
                     adjusted_persistence(raw, samples, cfg.reference_etat.statistique)], ignore_index=True)
    summary = summarize_cv(oof)
    pairs = pairwise(oof, [NAME, "Persistance", ADJUSTED, BEST], cfg.evaluation.bootstrap_samples, cfg.project.seed)
    folder = tables_dir(cfg, "etat_cv")
    summary.to_csv(folder / "cv_summary.csv", index=False)
    pairs.to_csv(folder / "cv_comparaisons_appariees.csv", index=False)
    mine = summary.set_index("model").loc[NAME]
    result = {"modele": NAME, "cv_MAE": float(mine["MAE"]), "cv_MAE_sd": float(mine["MAE_sd"]),
              "cv_couverture_80": float(state["inside_80"].mean()), "comparaisons": pairs.round(4).to_dict("records")}
    write_json(result, cfg.path("metrics") / "etat_cv.json")
    with model_run(cfg, cfg.reference_etat.question, NAME, {"ajuste_sur": "plis d'entraînement du dev"},
                   {"stage": "etat_cv", "family": "statistique"}, data_version(cfg), cross_validation(samples)):
        mlflow.log_metrics({**numeric(mine.drop("family").to_dict(), "cv_"),
                            "cv_couverture_80": result["cv_couverture_80"]})
        log_artifact(pairs, "cv_comparaisons_appariees.csv")
    logger(cfg, "etat_cv").info("score d'état en CV : MAE %.3f", result["cv_MAE"])
    return result
