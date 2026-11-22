"""Chargement du modèle servi : depuis le registry MLflow (alias champion) ou un fichier local en test."""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import pandas as pd

from .predictor import NextVisitPredictor


def load_served_model() -> tuple[NextVisitPredictor, dict]:
    name = os.environ.get("REGISTERED_MODEL", "parkinson-next-visit-updrs3")
    alias = os.environ.get("MODEL_ALIAS", "champion")
    uri = os.environ.get("MODEL_URI", f"models:/{name}@{alias}")
    if uri.endswith(".pkl"):                       # tests et démo hors registry
        return NextVisitPredictor.load(uri), {"version": "local", "source": uri}
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    predictor = mlflow.pyfunc.load_model(uri).unwrap_python_model().predictor
    version = mlflow.MlflowClient().get_model_version_by_alias(name, alias).version
    return predictor, {"version": str(version), "source": uri}


def load_state_model():
    """Modèle mixte du score d'état (fichier de notre pipeline, comme predictor.pkl)."""
    with open(os.environ.get("STATE_MODEL_PATH", "models/state_model.pkl"), "rb") as f:
        return pickle.load(f)


def load_factors() -> dict:
    """Résultats AGRÉGÉS du criblage statistique (jamais de données patient)."""
    root = Path(os.environ.get("PROJECT_ROOT", "."))
    table = pd.read_csv(root / "reports" / "tables" / "stats" / "criblage_facteurs_FDR.csv").round(4)
    summary = json.loads((root / "reports" / "metrics" / "stats.json").read_text(encoding="utf-8"))
    return {"mean_slope_pts_per_year": summary["mean_slope_pts_per_year"],
            "sd_individual_slopes": summary["sd_individual_slopes"],
            "factors": table.astype(object).where(table.notna(), None).to_dict("records")}
