"""Étape « train » : validation croisée groupée par patient, un run MLflow par modèle, un run imbriqué par pli."""
from __future__ import annotations

import time

import mlflow
import pandas as pd

from ..evaluation.metrics import by_horizon, summarize_cv
from ..training.cv import cross_validate
from ..training.settings import from_params
from ..training.tracking import log_artifact, log_fold, model_run, numeric
from ..utils.env import set_seed
from ..utils.io import write_json
from .common import data_version, device, load_modeling_data, logger, specs, tables_dir
from .run_datasets import cross_validation


def train(cfg, fast: bool = False, only=None) -> dict:
    log, version = logger(cfg, "train"), data_version(cfg)
    set_seed(cfg.project.seed)
    samples, X, mask = load_modeling_data(cfg)
    settings = from_params(cfg, device(), fast)
    folds = list(cfg.fast_mode.folds) if fast else list(range(cfg.split.n_folds))
    all_predictions = []
    for spec in specs(cfg, only):
        params = {"algo": spec.algo, "variables": "M1 (clinique)", "folds": len(folds),
                  **{k: v for k, v in vars(settings).items() if k != "device"}}
        tags = {"horizon": "visite_suivante", "stage": "cv", "fast_mode": fast, "device": settings.device}
        with model_run(cfg, spec.question, spec.name, params, tags, version, cross_validation(samples)):
            started = time.time()
            oof = cross_validate([spec], samples, X, mask, settings, folds, cfg.project.seed, log.info,
                                 on_fold=lambda s, f, m, i: log_fold(s.name, f, m, i, cfg.project.seed + f,
                                                                     cross_validation(samples, fold=f), cfg.root))
            summary = summarize_cv(oof).iloc[0].drop(["model", "family"]).to_dict()
            mlflow.log_metrics(numeric(summary, "cv_"))
            for _, row in by_horizon(oof).iterrows():
                mlflow.log_metric(f"cv_MAE_horizon_{row['horizon']}", float(row["MAE"]))
            mlflow.log_metric("train_seconds_total", time.time() - started)
            log_artifact(oof, "oof_predictions.csv")
        all_predictions.append(oof)
    oof = pd.concat(all_predictions, ignore_index=True)
    oof.to_csv(tables_dir(cfg, "models") / "cv_predictions.csv", index=False)
    summary = summarize_cv(oof)
    summary.to_csv(tables_dir(cfg, "models") / "cv_summary.csv", index=False)
    result = {"fast_mode": fast, "cv": summary.set_index("model")[["MAE", "MAE_sd", "R2_delta"]].to_dict("index")}
    write_json(result, cfg.path("metrics") / "cv.json")
    return result
