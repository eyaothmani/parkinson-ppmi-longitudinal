"""ADR-0017 : les quatre modèles comparés sont dans le registry ; réévaluer à l'identique ne crée aucune version."""
from mlflow import MlflowClient

from parkinson_mm.models.versions import ALIASES
from parkinson_mm.pipeline import evaluate

OBJECTIF2 = {"persistence", "lgbm", "mlp", "gru"}   # le GRU-D a son propre registered model (ADR-0018)


def _versions(cfg):
    return MlflowClient().search_model_versions(f"name='{cfg.tracking.registered_model}'")


def test_four_models_are_registered_with_their_own_alias(trained):
    cfg, result = trained
    client, name = MlflowClient(), cfg.tracking.registered_model
    assert {v.tags.get("modele") for v in _versions(cfg)} == OBJECTIF2
    for algo, alias in {a: ALIASES[a] for a in OBJECTIF2}.items():
        assert client.get_model_version_by_alias(name, alias).tags["modele"] == algo
    assert len(result["registry_comparaison"]) == 3 and result["selected"] not in result["registry_comparaison"]


def test_compared_models_are_never_promoted(trained):
    cfg, result = trained
    client, name = MlflowClient(), cfg.tracking.registered_model
    for version in result["registry_comparaison"].values():
        found = client.get_model_version(name, version)
        assert found.tags["promotion_decision"] == "comparé (non servi)"
        assert not {"champion", "candidate"} & set(found.aliases)


def test_reevaluating_with_the_same_data_and_config_creates_no_version(trained):
    cfg, result = trained
    before = len(_versions(cfg))
    again = evaluate(cfg, fast=True)
    assert len(_versions(cfg)) == before
    assert again["registry"]["version"] == result["registry"]["version"]
