"""Promotion (ADR-0004) : le critère 3 compare la validation croisée, jamais le test."""
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException

from parkinson_mm.models import promotion


class FakeClient:
    def __init__(self):
        self.aliases, self.tags = {}, {}

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            raise MlflowException("aucun alias")
        version = self.aliases[alias]
        return SimpleNamespace(version=version, tags=self.tags.get(version, {}))

    def set_model_version_tag(self, name, version, key, value):
        self.tags.setdefault(version, {})[key] = value


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(promotion, "MlflowClient", lambda: fake)
    return fake


def test_better_test_but_worse_cv_is_not_promoted(client):
    assert promotion.promote("m", "1", 6.0, (-1.0, -0.5), 0.1, 0.3, cv_mae=6.0)["champion"]
    decision = promotion.promote("m", "2", 5.0, (-1.0, -0.5), 0.1, 0.3, cv_mae=6.1)
    assert not decision["champion"] and client.aliases["champion"] == "1"
    assert "validation croisée" in decision["reasons"][0]


def test_worse_test_but_better_cv_is_promoted(client):
    promotion.promote("m", "1", 6.0, (-1.0, -0.5), 0.1, 0.3, cv_mae=6.0)
    assert promotion.promote("m", "2", 9.0, (-1.0, -0.5), 0.1, 0.3, cv_mae=5.9)["champion"]
    assert client.tags["2"]["cv_MAE"] == "5.9000"
