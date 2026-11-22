"""Entraînement rapide (une fois par session) partagé par les tests d'intégration modèle / registry / API."""
import pytest

FAST_EXPERIMENTS = ["Persistance", "LightGBM (M1)", "MLP – dernière visite (M1)", "GRU – trajectoire (M1)"]


@pytest.fixture(scope="session")
def trained(prepared):
    from parkinson_mm.pipeline import evaluate, explain, train

    train(prepared, fast=True, only=FAST_EXPERIMENTS)
    result = evaluate(prepared, fast=True)
    explain(prepared)
    return prepared, result


@pytest.fixture(scope="session")
def analysed(prepared):
    """Objectifs 1 et 3 exécutés une seule fois : score d'état (state) et facteurs associés (stats)."""
    from parkinson_mm.pipeline import state, stats

    return {"stats": stats(prepared), "state": state(prepared)}


@pytest.fixture
def client(trained, analysed, monkeypatch):
    """API complète (3 objectifs) branchée sur les modèles et résultats du projet temporaire."""
    from fastapi.testclient import TestClient

    cfg, _ = trained
    monkeypatch.setenv("MODEL_URI", str(cfg.path("models") / "predictor.pkl"))
    monkeypatch.setenv("STATE_MODEL_PATH", str(cfg.path("models") / "state_model.pkl"))
    monkeypatch.setenv("PROJECT_ROOT", str(cfg.root))
    monkeypatch.setenv("API_KEY", "secret-de-test")
    from parkinson_mm.serving.api import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def key():
    return {"X-API-Key": "secret-de-test"}


@pytest.fixture
def visits():
    return [{"time_years": 0, "np3": 20, "med_state": "untreated"},
            {"time_years": 1, "np3": 24, "med_state": "OFF", "ledd": 300}]
