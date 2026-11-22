"""Suivi MLflow : un experiment par question, un run par modèle, un run imbriqué par pli (ADR-0003).
Tags git_commit, dvc_rev, data_version, config_fingerprint, datasets déclarés (ADR-0019) : d'où vient chaque chiffre."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import mlflow

from ..utils.lineage import lineage
from .datasets import log_datasets
from .experiments import experiment_tags, question_test  # noqa: F401 (question_test : experiments TEST)
from .run_log import log_artifact, log_fold, numeric  # noqa: F401 (écrire dans un run : run_log.py)


def tracking_uri(cfg) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", cfg.tracking.uri)
    if uri.startswith(("http://", "https://")):   # le pipeline logue des prédictions par patient et des modèles
        raise SystemExit(f"MLFLOW_TRACKING_URI={uri} refusé : le suivi reste local (DUA PPMI, ADR-0003).")
    if uri.startswith("sqlite:///") and not Path(uri[10:]).is_absolute():
        uri = f"sqlite:///{cfg.root / uri[10:]}"        # base relative -> ancrée à la racine du projet
    return uri


@contextmanager
def model_run(cfg, question: str, name: str, params: dict, tags: dict, data_version: str | None, datasets=()):
    mlflow.set_tracking_uri(tracking_uri(cfg))
    experiment = f"{cfg.tracking.experiment_prefix}/{question}"
    if mlflow.get_experiment_by_name(experiment) is None:   # artefacts sous <projet>/mlruns, jamais sous le cwd
        mlflow.create_experiment(experiment, artifact_location=(cfg.path("mlruns") / question).as_uri(),
                                 tags=experiment_tags(cfg.root, question))   # description du catalogue (ADR-0022)
    mlflow.set_experiment(experiment)
    trace = lineage(cfg, data_version)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({**{k: str(v) for k, v in trace.items() if k != "versions"},
                         **{f"lib.{k}": v for k, v in trace["versions"].items()},
                         **{k: str(v) for k, v in tags.items()}})
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        log_datasets(cfg.root, datasets)
        yield run
