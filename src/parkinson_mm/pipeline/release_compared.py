"""Modèles comparés mais non servis : un run TEST et une version du registry chacun, sans promotion (ADR-0017)."""
from __future__ import annotations

import mlflow
from mlflow import MlflowClient

from ..models import versions
from ..models.registry import register
from ..training.tracking import log_artifact, model_run, numeric, question_test, tracking_uri
from .common import data_version
from .packaging import package
from .release import SOURCE_DIR
from .run_datasets import test_evaluation


def _reason(name: str, family: str, selected: str, cv) -> str:
    if family == "baseline":
        return "baseline : plancher de comparaison, jamais servie"
    return (f"MAE de validation croisée {cv.loc[name, 'MAE']:.3f} contre {cv.loc[selected, 'MAE']:.3f} "
            f"pour {selected} : le modèle servi est choisi sur la validation croisée seule (ADR-0004)")


def _register(cfg, name, model, selected, results, comparisons, cv, predictions, samples, dev, reference) -> str:
    predictor, _, by_horizon, metadata = package(cfg, model, name, results[name]["MAE"], predictions, samples, dev,
                                                 reference)
    ci = next((c for c in comparisons if c["comparaison"].startswith(name)), None)   # absent pour la persistance
    with model_run(cfg, question_test(model.spec.question), f"TEST – {name}", {"selection": "comparaison (non servi)"},
                   {"stage": "test"}, data_version(cfg), test_evaluation(samples)):
        mlflow.log_metrics(numeric({k: v for k, v in results[name].items() if k != "MAE_IC95"}, "test_"))
        mlflow.log_metrics({**{f"test_interval_coverage_{h}": c for h, c in by_horizon.items() if h != "global"},
                            "test_interval_coverage": by_horizon["global"],
                            **({"ci_vs_persistence_high": ci["IC95_haut"]} if ci else {})})
        log_artifact(metadata, "model_metadata.json")
        version = register(predictor, cfg.tracking.registered_model, SOURCE_DIR)
    tags = {"test_MAE": f"{results[name]['MAE']:.4f}", "cv_MAE_sd": f"{cv.loc[name, 'MAE_sd']:.4f}",
            "promotion_decision": "comparé (non servi)",
            "promotion_reasons": _reason(name, model.spec.family, selected, cv)}
    versions.label(cfg.tracking.registered_model, version, model.spec.algo, tags)
    return version


def release_compared(cfg, fitted: dict, selected: str, results: dict, comparisons: list, cv, predictions, samples, dev,
                     reference: dict) -> dict:
    """Enregistre les modèles autres que le modèle retenu ; renvoie {modèle: numéro de version}."""
    mlflow.set_tracking_uri(tracking_uri(cfg))
    registered, cv, out = cfg.tracking.registered_model, cv.set_index("model"), {}
    for name, model in fitted.items():
        experiment = f"{cfg.tracking.experiment_prefix}/{question_test(model.spec.question)}"
        if name == selected:
            continue
        done = versions.find_evaluated(experiment, registered, f"TEST – {name}", data_version(cfg), cfg.fingerprint)
        out[name] = done.version if done is not None else _register(cfg, name, model, selected, results, comparisons,
                                                                    cv, predictions, samples, dev, reference)
        if done is not None:
            versions.label(registered, done.version, model.spec.algo)
    MlflowClient().set_registered_model_tag(registered, "modeles_compares", ", ".join(out) + f", {selected}")
    return out
