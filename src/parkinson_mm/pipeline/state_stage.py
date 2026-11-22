"""Étape « state » — objectif 1 : score d'état actuel (niveau + vitesse) de chaque patient.

Le modèle mixte est ajusté sur les patients du DEV ; sa validité est vérifiée sur les patients de TEST
en prédisant leur visite suivante (vs persistance, IC bootstrap patient). Le modèle est sauvegardé pour l'API.
"""
from __future__ import annotations

import pickle

import mlflow
import pandas as pd

from ..evaluation.bootstrap import mae_ci, mae_difference
from ..evaluation.metrics import regression_metrics
from ..stats.factors import build_patient_factors
from ..stats.lme import BASE_ADJUSTMENT, prepare
from ..stats.state_apply import current_states, fit_state_model, forecast_samples
from ..training.tracking import model_run, numeric
from ..utils.io import write_json
from .common import data_version, logger, read_long, tables_dir
from .run_datasets import state_model

NAME = "Score d'état (modèle mixte)"
Z80 = 1.2816                                   # quantile normal pour un intervalle à 80 %


def validate_state(pred: pd.DataFrame, n_boot: int) -> dict:
    both = pd.concat([pred.assign(model=NAME), pred.assign(model="Persistance", pred=pred["y_last"])])
    inside = (pred["y"] - pred["pred"]).abs() <= Z80 * pred["pred_sd"]
    return {**regression_metrics(pred["y"], pred["y_last"], pred["pred"]), "couverture_80": float(inside.mean()),
            "MAE_IC95": mae_ci(both, NAME, n_boot), "vs_persistance": mae_difference(both, NAME, "Persistance", n_boot)}


def state(cfg) -> dict:
    long, patients = read_long(cfg)
    d = prepare(long, build_patient_factors(long, patients), cfg.stats.min_visits)
    splits = pd.read_csv(cfg.path("processed") / "splits.csv")
    model = fit_state_model(d[d["PATNO"].isin(splits.loc[splits["split"] == "dev", "PATNO"])])
    current = current_states(model, d)
    current.to_csv(cfg.path("processed") / "pd_score_etat.csv", index=False)

    samples = pd.read_csv(cfg.path("processed") / "pd_next_visit_samples.csv")
    pred = forecast_samples(model, d, samples[samples["split"] == "test"])
    pred.to_csv(tables_dir(cfg, "state") / "state_test_predictions.csv", index=False)
    result = {"patients": len(current), "niveau_median": float(current["niveau_actuel"].median()),
              "vitesse_mediane": float(current["vitesse"].median()),
              "test": validate_state(pred, cfg.evaluation.bootstrap_samples)}

    model_path = cfg.path("models") / "state_model.pkl"
    model_path.write_bytes(pickle.dumps(model))
    with model_run(cfg, "objectif1_score_actuel", NAME, {"formule": "np3 ~ time_years + " + BASE_ADJUSTMENT,
                                                         "ajuste_sur": "dev"}, {"stage": "state"}, data_version(cfg),
                   state_model(long, samples, splits)):
        mlflow.log_metrics(numeric(result["test"], "test_"))
        mlflow.log_artifact(str(model_path))
    write_json(result, cfg.path("metrics") / "state.json")
    logger(cfg, "state").info("score d'état : %d patients ; MAE visite suivante (test) %.3f",
                              result["patients"], result["test"]["MAE"])
    return result
