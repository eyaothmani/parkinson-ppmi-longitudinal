"""Étape « sensibilite » : modèles sans l'état de l'examen cible, et modèles « une seule visite » (RQ1).
Validation croisée seulement : le test n'est pas relu."""
from __future__ import annotations

import mlflow
import pandas as pd

from ..evaluation.bootstrap import mae_difference
from ..evaluation.metrics import summarize_cv
from ..pipeline.common import data_version, device, load_modeling_data, logger, tables_dir
from ..pipeline.run_datasets import cross_validation
from ..training.cv import cross_validate
from ..training.settings import from_params
from ..training.tracking import model_run, numeric
from ..utils.env import set_seed
from ..utils.io import write_json
from .sensitivity import neutralize, planned_pairs, variant_specs


def run_variants(cfg, fast: bool, log) -> pd.DataFrame:
    samples, X, mask = load_modeling_data(cfg)
    settings = from_params(cfg, device(), fast)
    folds = list(cfg.fast_mode.folds) if fast else list(range(cfg.split.n_folds))
    parts = []
    for label, entry in cfg.sensibilite.variantes.items():
        s, Xv = neutralize(samples, X, entry["retirer"])
        for spec in variant_specs(cfg, label, entry["modeles"]):
            params = {"algo": spec.algo, "neutralise": ", ".join(entry["retirer"]), "folds": len(folds)}
            with model_run(cfg, spec.question, spec.name, params, {"stage": "sensibilite", "variante": label,
                           "fast_mode": fast}, data_version(cfg), cross_validation(samples)):
                oof = cross_validate([spec], s, Xv, mask, settings, folds, cfg.project.seed, log.info)
                mlflow.log_metrics(numeric(summarize_cv(oof).iloc[0].drop(["model", "family"]).to_dict(), "cv_"))
            parts.append(oof)
    return pd.concat(parts, ignore_index=True)


def sensitivity(cfg, fast: bool = False) -> dict:
    log = logger(cfg, "sensibilite")
    set_seed(cfg.project.seed)
    variants = run_variants(cfg, fast, log)
    oof = pd.concat([pd.read_csv(tables_dir(cfg, "models") / "cv_predictions.csv"), variants], ignore_index=True)
    oof = oof[oof["fold"].isin(variants["fold"].unique())]                 # mêmes plis pour tous (mode rapide)
    pairs, n_boot = planned_pairs(cfg), cfg.evaluation.bootstrap_samples
    table = pd.DataFrame([mae_difference(oof, a, b, n_boot, cfg.project.seed, 0.05 / len(pairs)) for a, b in pairs])
    summary = summarize_cv(oof)
    folder = tables_dir(cfg, "sensibilite")
    variants.to_csv(folder / "cv_predictions.csv", index=False)
    summary.to_csv(folder / "cv_summary.csv", index=False)
    table.assign(niveau_IC=f"{100 * (1 - 0.05 / len(pairs)):.1f} % (Bonferroni, {len(pairs)} comparaisons)").to_csv(
        folder / "cv_comparaisons.csv", index=False)
    result = {"cv_MAE": summary.set_index("model")["MAE"].round(4).to_dict(), "comparaisons": table.round(4).to_dict(
        "records")}
    write_json(result, cfg.path("metrics") / "sensibilite.json")
    log.info("variantes : %s", {k: v for k, v in result["cv_MAE"].items() if " – " in k})
    return result
