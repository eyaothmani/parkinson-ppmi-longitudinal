"""Écrire dans un run MLflow ouvert : plis imbriqués (avec leurs datasets), artefacts et métriques numériques."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow

from .datasets import log_datasets


def log_fold(name: str, fold: int, metrics: dict, info: dict, seed: int, datasets=(), root=None) -> None:
    with mlflow.start_run(run_name=f"{name} | pli {fold}", nested=True):
        mlflow.set_tags({"fold": str(fold), "seed": str(seed)})
        log_datasets(root, datasets)
        mlflow.log_metrics(numeric({**metrics, **info}))


def log_artifact(obj, filename: str) -> None:
    """Enregistre un DataFrame (csv) ou un dict (json) comme artefact du run courant."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / filename
        if hasattr(obj, "to_csv"):
            obj.to_csv(path, index=False)
        else:
            path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        mlflow.log_artifact(str(path))


def numeric(values: dict, prefix: str = "") -> dict:
    """Garde les valeurs numériques finies ; renomme les clés que MLflow refuse (%)."""
    return {f"{prefix}{key}".replace("%", "pct"): float(value) for key, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value}
