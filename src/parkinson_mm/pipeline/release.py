"""Modèle retenu : empaquetage pour l'API, enregistrement dans le registry MLflow et décision de promotion."""
from __future__ import annotations

from pathlib import Path

import mlflow

from ..models import versions
from ..models.promotion import promote
from ..models.registry import register
from ..training.tracking import log_artifact, model_run, numeric, question_test, tracking_uri
from .common import data_version
from .packaging import package
from .run_datasets import test_evaluation

SOURCE_DIR = Path(__file__).resolve().parents[1]   # le package est embarqué avec le modèle (code_paths)


def release(cfg, model, name: str, test_metrics: dict, comparisons: list, cv, predictions, samples, dev,
            reference: dict) -> dict:
    predictor, cover, by_horizon, metadata = package(cfg, model, name, test_metrics["MAE"], predictions, samples, dev,
                                                     reference)
    cover.to_csv(cfg.path("reports") / "tables" / "models" / "test_couverture_horizon.csv", index=False)
    predictor.save(cfg.path("models") / "predictor.pkl")
    mlflow.set_tracking_uri(tracking_uri(cfg))      # l'étape peut tourner seule (dvc repro) : base explicite
    question = question_test(model.spec.question)          # runs TEST à part des runs CV (ADR-0022)
    registered, experiment = cfg.tracking.registered_model, f"{cfg.tracking.experiment_prefix}/{question}"
    done = versions.find_evaluated(experiment, registered, f"TEST – {name}", data_version(cfg), cfg.fingerprint)
    if done is not None:                  # mêmes données, même configuration : aucune nouvelle version (ADR-0017)
        versions.label(registered, done.version, model.spec.algo)
        return versions.decision(registered, done.version)
    ci = next(c for c in comparisons if c["comparaison"].startswith(name))
    with model_run(cfg, question, f"TEST – {name}", {"selection": "cv_MAE"}, {"stage": "test"}, data_version(cfg),
                   test_evaluation(samples)):
        mlflow.log_metrics(numeric({k: v for k, v in test_metrics.items() if k != "MAE_IC95"}, "test_"))
        mlflow.log_metrics({**{f"test_interval_coverage_{h}": c for h, c in by_horizon.items() if h != "global"},
                            "test_interval_coverage": by_horizon["global"], "ci_vs_persistence_high": ci["IC95_haut"]})
        log_artifact(metadata, "model_metadata.json")
        version = register(predictor, registered, SOURCE_DIR)
        cv_row = cv.set_index("model").loc[name]
        decision = promote(registered, version, test_metrics["MAE"], (ci["IC95_bas"], ci["IC95_haut"]),
                           float(cv_row["MAE_sd"]), cfg.promotion.max_cv_mae_sd, float(cv_row["MAE"]))
    versions.label(registered, version, model.spec.algo)
    return decision
