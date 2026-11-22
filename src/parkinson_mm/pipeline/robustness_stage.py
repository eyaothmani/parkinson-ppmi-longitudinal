"""Étape « robustesse » (sprint 6) : où le modèle retenu se trompe-t-il, et son intervalle à 80 % est-il fiable ?
Uniquement sur les prédictions HORS-PLI de la validation croisée : le test, déjà utilisé une fois, n'est pas relu."""
from __future__ import annotations

import mlflow
import pandas as pd

from ..evaluation.conformal import coverage, cross_conformal
from ..evaluation.robustness import subgroup_errors, with_groups
from ..evaluation.subgroups import assign_groups
from ..training.tracking import model_run, numeric
from ..utils.io import write_json
from ..visualization import robustness as figures
from ..visualization import set_style
from .common import data_version, logger, tables_dir
from .run_datasets import out_of_fold


def summary(errors: pd.DataFrame, cover: pd.DataFrame, model: str) -> dict:
    mine, groups = errors[errors["modèle"] == model], cover[cover["critère"] != "Tous"]
    worst, best, low = mine.loc[mine["MAE"].idxmax()], mine.loc[mine["MAE"].idxmin()], groups.loc[
        groups["couverture"].idxmin()]
    return {"modele": model, "couverture_globale": float(cover.iloc[0]["couverture"]),
            "demi_largeur_moyenne": float(cover.iloc[0]["demi_largeur"]),
            "groupe_MAE_max": f"{worst['critère']} · {worst['groupe']}", "MAE_max": float(worst["MAE"]),
            "groupe_MAE_min": f"{best['critère']} · {best['groupe']}", "MAE_min": float(best["MAE"]),
            "groupe_couverture_min": f"{low['critère']} · {low['groupe']}", "couverture_min": float(low["couverture"])}


def robustness(cfg) -> dict:
    r = cfg.robustesse
    oof = pd.read_csv(tables_dir(cfg, "models") / "cv_predictions.csv")
    samples = pd.read_csv(cfg.path("processed") / "pd_next_visit_samples.csv")
    groups = assign_groups(samples)
    data = with_groups(oof, groups)
    errors = subgroup_errors(data, groups.columns, [r.modele, r.reference], cfg.evaluation.bootstrap_samples,
                             cfg.project.seed)
    covered = cross_conformal(data[data["model"] == r.modele].reset_index(drop=True), r.niveau)
    cover = coverage(covered, groups.columns)
    folder, figs = tables_dir(cfg, "robustesse"), cfg.path("reports") / "figures" / "robustesse"
    errors.to_csv(folder / "mae_par_groupe.csv", index=False)
    cover.to_csv(folder / "couverture_par_groupe.csv", index=False)
    set_style()
    paths = [figures.mae_by_group(errors, r.modele, r.reference, figs),
             figures.coverage_by_group(cover, r.niveau, figs), figures.residuals(covered, figs),
             folder / "mae_par_groupe.csv", folder / "couverture_par_groupe.csv"]
    result = summary(errors, cover, r.modele)
    write_json(result, cfg.path("metrics") / "robustesse.json")
    params = {"modele": r.modele, "reference": r.reference, "niveau": r.niveau, "donnees": "CV hors-pli (dev)"}
    with model_run(cfg, r.question, f"Robustesse – {r.modele}", params, {"stage": "robustesse"}, data_version(cfg),
                   out_of_fold(samples)):
        mlflow.log_metrics(numeric(result))
        for path in paths:
            mlflow.log_artifact(str(path))
    logger(cfg, "robustesse").info("couverture globale %.3f ; MAE max : %s", result["couverture_globale"],
                                   result["groupe_MAE_max"])
    return result
