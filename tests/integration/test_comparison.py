"""Limites corrigées (v1.1) : comparaisons appariées, importance par modèle, intervalle par horizon."""
import mlflow
import pandas as pd

from parkinson_mm.serving.predictor import NextVisitPredictor


def test_models_are_compared_pairwise_and_intervals_depend_on_horizon(trained):
    cfg, result = trained
    folder = cfg.path("reports") / "tables" / "models"
    for name in ("cv_comparaisons_appariees.csv", "test_comparaisons_appariees.csv", "importance_par_modele.csv",
                 "test_couverture_horizon.csv"):
        assert not pd.read_csv(folder / name).empty, name
    assert set(result["comparaisons_appariees"]) == {"cv", "test"}
    predictor = NextVisitPredictor.load(cfg.path("models") / "predictor.pkl")
    assert "global" in predictor.interval and predictor.reference["tabular"]


def test_artifacts_stay_inside_the_project(trained):
    """Les artefacts MLflow d'un projet restent sous SA racine (avant : ./mlruns du dossier courant)."""
    cfg, _ = trained
    uris = mlflow.search_runs(search_all_experiments=True)["artifact_uri"]
    assert uris.str.startswith(cfg.root.as_uri()).all()
