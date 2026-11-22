"""Étape « reference_horizons » : persistance ajustée à l'état face aux modèles multi-horizon (CV seulement)."""
from __future__ import annotations

import mlflow
import pandas as pd

from ..evaluation.bootstrap import mae_difference
from ..evaluation.horizon_metrics import summarize_cv
from ..pipeline.common import data_version, logger, tables_dir
from ..pipeline.run_datasets import HORIZONS, cross_validation
from ..training.tracking import log_artifact, model_run
from ..utils.io import write_json
from .horizon_reference import adjusted_horizons
from .reference import NAME


def comparisons(cfg, oof: pd.DataFrame, months) -> pd.DataFrame:
    learned = [m for m in oof.loc[oof["family"] != "baseline", "model"].unique()]
    alpha = 0.05 / (len(learned) * len(months))                  # Bonferroni : modèles × horizons
    rows = [{"horizon": h, **mae_difference(oof[oof["horizon"] == h], model, NAME, cfg.evaluation.bootstrap_samples,
                                            cfg.project.seed, alpha)} for h in months for model in learned]
    return pd.DataFrame(rows).assign(niveau_IC=f"{100 * (1 - alpha):.1f} % (Bonferroni)")


def reference_horizons(cfg) -> dict:
    h, months = cfg.horizons, list(cfg.horizons.months)
    oof = pd.read_csv(tables_dir(cfg, "horizons") / "cv_predictions.csv")
    oof = oof[~oof["model"].str.contains(" sans ")]                       # ablations du GRU-D : hors comparaison
    samples = pd.read_csv(cfg.path("processed") / "pd_multi_horizon_samples.csv")
    oof = pd.concat([oof, adjusted_horizons(oof, samples, months, cfg.reference_etat.statistique)], ignore_index=True)
    summary, pairs = summarize_cv(oof, h.mcid_points), comparisons(cfg, oof, months)
    folder = tables_dir(cfg, "reference_horizons")
    summary.to_csv(folder / "cv_summary.csv", index=False)
    pairs.to_csv(folder / "cv_comparaisons.csv", index=False)
    mine = summary[summary["model"] == NAME].set_index("horizon")
    result = {"reference": NAME, **{f"cv_MAE_{m}m": float(mine.loc[m, "MAE"]) for m in months},
              "comparaisons": pairs.round(4).to_dict("records")}
    write_json(result, cfg.path("metrics") / "reference_horizons.json")
    with model_run(cfg, h.question, NAME, {"statistique": cfg.reference_etat.statistique, "donnees": "CV hors-pli"},
                   {"stage": "reference_horizons", "family": "baseline"}, data_version(cfg),
                   cross_validation(samples, HORIZONS)):
        mlflow.log_metrics({k: v for k, v in result.items() if k.startswith("cv_")})
        log_artifact(summary, "cv_summary.csv")
        log_artifact(pairs, "cv_comparaisons.csv")
    logger(cfg, "reference_horizons").info("persistance ajustée : %s", mine["MAE"].round(3).to_dict())
    return result
