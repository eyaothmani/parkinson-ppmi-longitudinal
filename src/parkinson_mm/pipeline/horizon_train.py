"""Étape « train_horizons » (ADR-0018) : réglage à budget égal (plis 0 et 1), puis validation croisée à 5 plis des
4 modèles et des 3 ablations du GRU-D. Experiment MLflow `objectif2_multi_horizon` : un run par modèle (tags `role`,
`protocol`, essais du réglage en artefact), un run imbriqué par pli, datasets déclarés (ADR-0019). Test jamais lu.
"""
from __future__ import annotations

import mlflow
import pandas as pd

from ..evaluation.horizon_export import as_json, as_mlflow_metrics
from ..evaluation.horizon_metrics import summarize, summarize_cv
from ..training.horizon_cv import cross_validate_horizon, horizon_specs
from ..training.settings import from_params
from ..training.tracking import log_artifact, log_fold, model_run, numeric
from ..utils.env import set_seed
from ..utils.io import write_json
from .common import data_version, device, logger, tables_dir
from .horizon_data import load_horizon_data
from .horizon_tune import tune_all, tuned
from .run_datasets import HORIZONS, cross_validation


def _fold_logger(cfg, name: str, samples):
    def on_fold(fold: int, part: pd.DataFrame, info: dict) -> None:
        log_fold(name, fold, as_mlflow_metrics(summarize(part, cfg.horizons.mcid_points)), info,
                 cfg.project.seed + fold, cross_validation(samples, HORIZONS, fold), cfg.root)
    return on_fold


def train_horizons(cfg, fast: bool = False, only=None) -> dict:
    set_seed(cfg.project.seed)
    h, version = cfg.horizons, data_version(cfg)
    samples, X, visit = load_horizon_data(cfg)
    settings = from_params(cfg, device(), fast)
    folds = list(cfg.fast_mode.folds) if fast else list(range(cfg.split.n_folds))
    ablations, results, specs = {e["name"] for e in h.ablations}, [], horizon_specs(cfg, only)
    reglages, trials = tune_all(cfg, specs, samples, X, visit, settings, fast)
    for spec in tuned(specs, reglages):
        params = {"algo": spec.algo, "options": dict(spec.options), "horizons": list(h.months), "folds": len(folds),
                  **{k: v for k, v in vars(settings).items() if k != "device"}}
        tags = {"stage": "cv", "role": "ablation" if spec.name in ablations else "modele_compare",
                "protocol": "ADR-0018", "fast_mode": fast, "device": settings.device}
        with model_run(cfg, h.question, spec.name, params, tags, version, cross_validation(samples, HORIZONS)):
            oof = cross_validate_horizon(spec, samples, X, visit, settings, h, folds, cfg.project.seed,
                                         _fold_logger(cfg, spec.name, samples))
            mlflow.log_metrics(numeric(as_mlflow_metrics(summarize_cv(oof, h.mcid_points), "cv_")))
            log_artifact(oof, "oof_predictions.csv")
            if spec.algo in trials and spec.name not in ablations:
                log_artifact(trials[spec.algo], "reglage_essais.csv")
        results.append(oof)
    oof = pd.concat(results, ignore_index=True)
    oof.to_csv(tables_dir(cfg, "horizons") / "cv_predictions.csv", index=False)
    table = summarize_cv(oof, h.mcid_points)
    table.to_csv(tables_dir(cfg, "horizons") / "cv_summary.csv", index=False)
    result = {"fast_mode": fast, "reglages": reglages, "cv": as_json(table)}
    write_json(result, cfg.path("metrics") / "horizons_cv.json")
    logger(cfg, "train_horizons").info("validation croisée : %s", table[["model", "horizon", "MAE"]].to_dict("records"))
    return result
