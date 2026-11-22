"""De bout en bout : validation croisée tracée, évaluation, registry MLflow, explicabilité."""
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from parkinson_mm.serving.predictor import NextVisitPredictor
from parkinson_mm.serving.pyfunc import example_input


def test_every_experiment_has_cv_results(trained):
    cfg, _ = trained
    summary = pd.read_csv(cfg.path("reports") / "tables" / "models" / "cv_summary.csv")
    assert {"Persistance", "GRU – trajectoire (M1)"} <= set(summary["model"])


def test_runs_are_traced_with_lineage_tags(trained):
    cfg, _ = trained
    runs = mlflow.search_runs(search_all_experiments=True)
    assert {"tags.git_commit", "tags.data_version", "tags.config_fingerprint"} <= set(runs.columns)
    assert (runs["tags.fold"].notna()).sum() >= 4          # runs imbriqués : un par pli


def test_model_is_registered_with_a_decision(trained):
    cfg, result = trained
    version = MlflowClient().get_model_version(cfg.tracking.registered_model, result["registry"]["version"])
    assert version.tags["promotion_decision"] in {"champion", "candidate"}
    assert "candidate" in version.aliases


def test_predictions_are_finite_and_in_range(trained):
    cfg, _ = trained
    out = NextVisitPredictor.load(cfg.path("models") / "predictor.pkl").predict_frame(example_input())
    assert np.isfinite(out["prediction"]).all() and out["prediction"].between(0, 132).all()
    assert (out["lower"] <= out["prediction"]).all() and (out["prediction"] <= out["upper"]).all()


def test_registered_pyfunc_matches_local_predictor(trained):
    """Compatibilité prétraitement/modèle : le modèle du registry prédit exactement comme le modèle local."""
    cfg, result = trained
    uri = f"models:/{cfg.tracking.registered_model}/{result['registry']['version']}"
    served = mlflow.pyfunc.load_model(uri).predict(example_input())
    local = NextVisitPredictor.load(cfg.path("models") / "predictor.pkl").predict_frame(example_input())
    np.testing.assert_allclose(served["prediction"], local["prediction"])


def test_permutation_importance_is_written(trained):
    cfg, _ = trained
    assert not pd.read_csv(cfg.path("reports") / "tables" / "models" / "importance_permutation.csv").empty



def test_runs_declare_their_datasets(trained, analysed):
    """ADR-0019 : colonne « Dataset » remplie ; un pli déclare entraînement ET validation, un run TEST son test."""
    runs = mlflow.search_runs(search_all_experiments=True, output_format="list")
    contexts = {r.info.run_name: {t.value for d in r.inputs.dataset_inputs for t in d.tags
                                  if t.key == "mlflow.data.context"} for r in runs}
    assert all(contexts.values())
    assert all(c == {"training", "validation"} for n, c in contexts.items() if " | pli " in n)
    assert all(c == {"training", "test"} for n, c in contexts.items() if n.startswith("TEST – "))
