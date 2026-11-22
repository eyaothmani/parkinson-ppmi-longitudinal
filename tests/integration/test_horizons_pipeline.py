"""Sprint 5 de bout en bout (synthétique) : CV, test unique, hypothèses, registry (ADR-0018), datasets (ADR-0019)."""
import mlflow
import pytest
from mlflow import MlflowClient

from parkinson_mm.serving.pyfunc import example_input

FAST = ["Persistance", "LightGBM (un par horizon)", "MLP – dernière visite, multi-sortie", "GRU-D multi-horizon",
        "GRU-D sans masques"]


@pytest.fixture(scope="module")
def horizon_run(prepared, analysed):
    from parkinson_mm.pipeline import evaluate_horizons, horizons, train_horizons

    data = horizons(prepared)
    train_horizons(prepared, fast=True, only=FAST)
    return prepared, data, evaluate_horizons(prepared, fast=True)


def test_every_horizon_has_pairs(horizon_run):
    _, data, _ = horizon_run
    assert all(row["paires"] > 0 for row in data["par_horizon"].values())


def test_hypotheses_are_reported_with_their_role(horizon_run):
    _, _, result = horizon_run
    assert {"H1", "H2", "H4", "H5"} <= set(result["hypotheses"])
    assert result["hypotheses"]["H1"]["role"] == "confirmatoire"
    assert result["choix_cv"]["dl"] == "GRU-D multi-horizon"
    assert result["choix_cv"]["meilleur_tabulaire"] in set(FAST[1:3])


def test_four_compared_models_are_registered_with_their_alias(horizon_run):
    cfg, _, result = horizon_run
    client, name = MlflowClient(), cfg.horizons.registered_model
    assert len(result["registry"]["versions"]) == 4
    assert client.get_model_version_by_alias(name, "grud").tags["modele"] == "grud"


def test_registered_gru_d_predicts_three_ordered_horizons(horizon_run):
    cfg, _, result = horizon_run
    version = result["registry"]["versions"]["GRU-D multi-horizon"]
    out = mlflow.pyfunc.load_model(f"models:/{cfg.horizons.registered_model}/{version}").predict(example_input())
    assert out["horizon_mois"].tolist() == [12, 24, 36]
    assert (out["borne_basse_10"] <= out["prediction"]).all() and (out["prediction"] <= out["borne_haute_90"]).all()


def test_every_run_declares_its_datasets(horizon_run, trained):
    """ADR-0019 : plus aucune colonne « Dataset » vide, quel que soit l'objectif."""
    runs = mlflow.search_runs(search_all_experiments=True, output_format="list")
    assert runs and all(run.inputs.dataset_inputs for run in runs)
    contexts = {tag.value for run in runs for d in run.inputs.dataset_inputs for tag in d.tags
                if tag.key == "mlflow.data.context"}
    assert contexts == {"training", "validation", "test"}
