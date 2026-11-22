"""Registry du multi-horizon (ADR-0018, même logique que l'ADR-0017) : un run TEST et une version par modèle comparé,
un alias par modèle. Seul le modèle retenu sur la validation croisée à l'horizon principal peut devenir `champion`
(critères objectifs de l'ADR-0004). Réévaluer avec les mêmes données et la même configuration ne crée aucune version."""
from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
from mlflow.models import infer_signature

from ..evaluation.horizon_export import as_mlflow_metrics
from ..models import versions
from ..models.promotion import promote
from ..serving.horizon_predictor import HorizonPredictor, HorizonPyfunc
from ..serving.pyfunc import example_input
from ..training.tracking import model_run, numeric, question_test, tracking_uri
from .common import data_version
from .release import SOURCE_DIR
from .run_datasets import HORIZONS, test_evaluation


def register_horizon(predictor: HorizonPredictor, registered: str) -> str:
    example = example_input()
    signature = infer_signature(example, predictor.predict_frame(example))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "predictor.pkl"
        predictor.save(path)
        info = mlflow.pyfunc.log_model(name="model", python_model=HorizonPyfunc(), artifacts={"predictor": str(path)},
                                       code_paths=[str(SOURCE_DIR)], signature=signature, input_example=example,
                                       registered_model_name=registered)
    return str(info.registered_model_version)


def release_horizons(cfg, fitted: dict, table, selected: str, ci: tuple, cv_sd: float, samples,
                     cv_mae: float | None = None) -> dict:
    """Enregistre les modèles comparés ; renvoie {modèle: version} et la décision de promotion du modèle retenu."""
    h, version_tag, question = cfg.horizons, data_version(cfg), question_test(cfg.horizons.question)   # ADR-0022
    mlflow.set_tracking_uri(tracking_uri(cfg))
    registered, experiment, out = h.registered_model, f"{cfg.tracking.experiment_prefix}/{question}", {}
    for name, model in fitted.items():
        done = versions.find_evaluated(experiment, registered, f"TEST – {name}", version_tag, cfg.fingerprint)
        if done is not None:                     # mêmes données, même configuration : aucune nouvelle version
            out[name] = done.version
            versions.label(registered, done.version, model.spec.algo)
            continue
        mine = table[table["model"] == name]
        with model_run(cfg, question, f"TEST – {name}", {"graines": model.info.get("seeds", 1)},
                       {"stage": "test", "protocol": "ADR-0018"}, version_tag, test_evaluation(samples, HORIZONS)):
            mlflow.log_metrics(numeric(as_mlflow_metrics(mine, "test_")))
            mlflow.log_metrics(numeric({f"test_MAE_sd_graines_{int(r['horizon'])}m": r["MAE_sd_graines"]
                                        for _, r in mine.iterrows()}))
            out[name] = register_horizon(HorizonPredictor(model, cfg.samples.max_seq_len, {"model": name}), registered)
        primary = float(mine.loc[mine["horizon"] == h.primary, "MAE"].iloc[0])
        decision = "retenu sur la validation croisée" if name == selected else "comparé (non servi)"
        versions.label(registered, out[name], model.spec.algo, {"promotion_decision": decision,
                                                                "test_MAE": f"{primary:.4f}"})
        if name == selected:
            promote(registered, out[name], primary, ci, cv_sd, cfg.promotion.max_cv_mae_sd, cv_mae)
    return {"versions": out, "decision": versions.decision(registered, out[selected])}
