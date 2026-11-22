"""Colonne « Dataset » de MLflow : chaque run déclare les données lues et leur rôle (ADR-0019).

MLflow n'enregistre que le nom, l'empreinte (digest), le schéma et le nombre de lignes : les données PPMI ne quittent
jamais le disque (DUA). `source` = chemin du fichier suivi par DVC ; son md5 (le hash de `dvc.lock`) est mis en tag.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import NamedTuple

import mlflow
import mlflow.data
import pandas as pd
from mlflow.entities import DatasetInput, InputTag

_CACHE: dict = {}   # md5 par (fichier, date de modification) ; description MLflow par sous-ensemble


class Data(NamedTuple):
    frame: pd.DataFrame
    name: str                   # ex. « visite_suivante:pli 3 »
    context: str                # training | validation | test
    file: str                   # chemin relatif à la racine du projet, ex. data/processed/pd_next_visit_samples.csv
    targets: str | None = None


def file_md5(root: Path, file: str) -> str:
    path = Path(root) / file
    key = ("md5", str(path), path.stat().st_mtime if path.exists() else None)
    if key not in _CACHE:
        _CACHE[key] = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest() if path.exists() else "absent"
    return _CACHE[key]


def _entity(d: Data, md5: str):
    key = ("dataset", d.file, md5, d.name, len(d.frame))
    if key not in _CACHE:
        with warnings.catch_warnings():   # PATNO entier : l'avertissement de schéma MLflow ne s'applique pas ici
            warnings.simplefilter("ignore", UserWarning)
            dataset = mlflow.data.from_pandas(d.frame, source=d.file, name=d.name, targets=d.targets)
            _CACHE[key] = dataset._to_mlflow_entity()   # même conversion que mlflow.log_input (schéma calculé ici)
    return _CACHE[key]


def dataset_inputs(root: Path, entries) -> list:
    """Entités MLflow prêtes à être attachées à un run, en cours ou déjà terminé (rattrapage)."""
    inputs = []
    for d in entries:
        tags = {"mlflow.data.context": d.context, "md5": file_md5(root, d.file)}
        inputs.append(DatasetInput(_entity(d, tags["md5"]), [InputTag(k, v) for k, v in tags.items()]))
    return inputs


def log_datasets(root: Path, entries) -> None:
    """Attache les datasets au run actif (un seul appel au backend)."""
    if entries:
        mlflow.MlflowClient().log_inputs(mlflow.active_run().info.run_id, dataset_inputs(root, entries))
