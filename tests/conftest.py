"""Fixtures partagées. Aucun test n'utilise les vraies données PPMI : tout part du jeu synthétique."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from parkinson_mm.data.synthetic import generate
from parkinson_mm.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """Projet temporaire : params.yaml du dépôt + données synthétiques + MLflow local isolé."""
    root = tmp_path_factory.mktemp("projet")
    shutil.copy(ROOT / "params.yaml", root / "params.yaml")
    (root / "docs").mkdir()
    shutil.copy(ROOT / "docs" / "mlflow_experiments.yaml", root / "docs")   # descriptions des experiments (ADR-0022)
    generate(root / "data" / "synthetic", n_pd=120, seed=0)
    os.environ["PPMI_RAW_DIR"] = str(root / "data" / "synthetic")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{root / 'mlflow.db'}"
    return load_config(root / "params.yaml").make_dirs()


@pytest.fixture(scope="session")
def prepared(project):
    """Données préparées jusqu'aux échantillons de modélisation (ingest -> features)."""
    from parkinson_mm.pipeline import features, ingest, split, validate

    for stage in (ingest, validate, split, features):
        stage(project)
    return project
