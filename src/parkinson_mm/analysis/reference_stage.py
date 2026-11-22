"""Étape « reference_etat » : la persistance ajustée à l'état face aux 4 modèles de l'objectif 2.
Validation croisée seulement (prédictions hors-pli du dev) : le test n'est pas relu."""
from __future__ import annotations

import mlflow
import pandas as pd

from ..evaluation.bootstrap import pairwise
from ..evaluation.metrics import summarize_cv
from ..pipeline.common import data_version, logger, tables_dir
from ..pipeline.run_datasets import out_of_fold
from ..training.tracking import log_artifact, model_run, numeric
from ..utils.io import write_json
from .reference import NAME, adjusted_persistence, changed_share, errors_by_transition, transitions


def tables(cfg, oof: pd.DataFrame, samples: pd.DataFrame) -> dict:
    r = cfg.reference_etat
    oof = pd.concat([oof, adjusted_persistence(oof[oof["model"] == r.persistance], samples, r.statistique)],
                    ignore_index=True)
    oof["transition"] = transitions(samples).reindex(oof["sample_id"]).to_numpy()
    summary = summarize_cv(oof)
    pairs = pairwise(oof, summary["model"].tolist(), cfg.evaluation.bootstrap_samples, cfg.project.seed)
    return {"cv_summary": summary, "cv_comparaisons_appariees": pairs, "mae_par_transition": errors_by_transition(oof)}


def reference_state(cfg) -> dict:
    r = cfg.reference_etat
    oof = pd.read_csv(tables_dir(cfg, "models") / "cv_predictions.csv")
    samples = pd.read_csv(cfg.path("processed") / "pd_next_visit_samples.csv")
    out = tables(cfg, oof, samples)
    folder = tables_dir(cfg, "reference_etat")
    for name, table in out.items():
        table.to_csv(folder / f"{name}.csv", index=False)
    summary = out["cv_summary"].set_index("model")
    learned = summary[summary["family"] != "baseline"]
    result = {"reference": NAME, "cv_MAE": float(summary.loc[NAME, "MAE"]),
              "cv_MAE_sd": float(summary.loc[NAME, "MAE_sd"]),
              "cv_MAE_persistance": float(summary.loc[r.persistance, "MAE"]),
              "meilleur_modele": learned["MAE"].idxmin(), "cv_MAE_meilleur": float(learned["MAE"].min()),
              "part_changement_etat": changed_share(samples, oof.loc[oof["model"] == r.persistance, "sample_id"])}
    write_json(result, cfg.path("metrics") / "reference_etat.json")
    params = {"statistique": r.statistique, "depart": r.persistance, "donnees": "CV hors-pli (dev)"}
    with model_run(cfg, r.question, NAME, params, {"stage": "reference_etat", "family": "baseline"},
                   data_version(cfg), out_of_fold(samples)):
        mlflow.log_metrics(numeric(summary.loc[NAME].drop("family").to_dict(), "cv_"))
        for name, table in out.items():
            log_artifact(table, f"{name}.csv")
    logger(cfg, "reference_etat").info("CV : %s %.3f ; persistance %.3f ; meilleur modèle %.3f", NAME,
                                       result["cv_MAE"], result["cv_MAE_persistance"], result["cv_MAE_meilleur"])
    return result
