"""Étape « evaluate_horizons » (ADR-0018) : évaluation UNIQUE sur le test, après le tag `gel-multi-horizon`.

Chaque modèle est réentraîné sur tout le dev avec 5 graines : prédiction = moyenne des graines ; l'écart-type de la MAE
entre graines mesure la stabilité. Les choix (modèle tabulaire de H1, modèle retenu) viennent de la validation croisée.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.bootstrap import mae_ci, mae_difference
from ..evaluation.horizon_export import as_json
from ..evaluation.horizon_metrics import long_format, summarize
from ..evaluation.hypotheses import evaluate_hypotheses
from ..serving.horizon_predictor import HorizonPredictor
from ..training.horizon_cv import horizon_specs
from ..training.horizon_fit import fit_horizon
from ..training.settings import from_params
from ..utils.env import set_seed
from ..utils.io import write_json
from .common import device, logger, tables_dir
from .horizon_choices import cv_choices, speeds
from .horizon_data import load_horizon_data
from .horizon_release import release_horizons
from .horizon_tune import frozen_reglages, tuned


def evaluate_horizons(cfg, fast: bool = False) -> dict:
    set_seed(cfg.project.seed)
    h, choices, folder = cfg.horizons, cv_choices(cfg), tables_dir(cfg, "horizons")
    samples, X, visit = load_horizon_data(cfg)
    dev, test = (np.where(samples["split"] == s)[0] for s in ("dev", "test"))
    settings, n_seeds = from_params(cfg, device(), fast), 1 if fast else h.final_seeds
    fitted, quantiles, parts, per_seed = {}, {}, [], []
    for spec in tuned(horizon_specs(cfg, ablations=False), frozen_reglages(cfg)):   # réglages figés avant le test
        fitted[spec.name] = fit_horizon(spec, samples, X, visit, dev, settings, h, cfg.project.seed, n_seeds)
        each = fitted[spec.name].predict_each(samples.iloc[test], X[test], visit[test])
        quantiles[spec.name] = np.mean(each, axis=0)
        parts.append(long_format(samples, test, quantiles[spec.name], h.months, spec.name, family=spec.family))
        per_seed += [summarize(long_format(samples, test, q, h.months, spec.name), h.mcid_points) for q in each]
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(folder / "test_predictions.csv", index=False)
    spread = pd.concat(per_seed).groupby(["model", "horizon"])["MAE"].std().fillna(0.0).rename("MAE_sd_graines")
    table = summarize(predictions, h.mcid_points).merge(spread.reset_index(), on=["model", "horizon"])
    table["MAE_IC95"] = [mae_ci(predictions[predictions["horizon"] == r["horizon"]], r["model"])
                         for _, r in table.iterrows()]
    table.to_csv(folder / "test_summary.csv", index=False)
    hypotheses = evaluate_hypotheses(predictions, choices["dl"], choices["meilleur_tabulaire"],
                                     speeds(cfg, samples, test, quantiles[choices["dl"]]), h)
    primary = predictions[predictions["horizon"] == h.primary]
    ci = mae_difference(primary, choices["retenu"], choices["persistance"], cfg.evaluation.bootstrap_samples)
    registry = release_horizons(cfg, fitted, table, choices["retenu"], (ci["IC95_bas"], ci["IC95_haut"]),
                                choices["retenu_cv_MAE_sd"], samples, choices.get("retenu_cv_MAE"))
    HorizonPredictor(fitted[choices["dl"]], cfg.samples.max_seq_len, {"model": choices["dl"]}).save(
        cfg.path("models") / "horizon_predictor.pkl")
    result = {"fast_mode": fast, "choix_cv": choices, "hypotheses": hypotheses, "registry": registry,
              "test": as_json(table)}
    write_json(result, cfg.path("metrics") / "horizons_test.json")
    logger(cfg, "evaluate_horizons").info("H1 : %s", hypotheses["H1"])
    return result
