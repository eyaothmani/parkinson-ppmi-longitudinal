"""params.yaml est la seule source des paramètres : il doit être complet, strict et traçable."""
from pathlib import Path

import pytest
import yaml

from parkinson_mm.utils.config import Config, load_config

ROOT = Path(__file__).resolve().parents[2]


def test_repository_params_are_valid():
    cfg = load_config(ROOT / "params.yaml")
    assert cfg.split.test_size == pytest.approx(0.2)
    # ADR-0015 : exactement quatre modèles comparés, tous dans le même experiment MLflow
    assert [e["name"] for e in cfg.experiments] == ["Persistance", "LightGBM (M1)", "MLP – dernière visite (M1)",
                                                   "GRU – trajectoire (M1)"]
    assert {e["question"] for e in cfg.experiments} == {"objectif2_score_futur"}


def test_unknown_section_is_rejected(tmp_path):
    raw = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    raw["typo_section"] = {}
    with pytest.raises(ValueError, match="inconnues"):
        Config(raw, tmp_path)


def test_raw_dir_can_be_redirected(monkeypatch):
    monkeypatch.setenv("PPMI_RAW_DIR", "/tmp/synthetique")
    assert str(load_config(ROOT / "params.yaml").path("raw")).endswith("synthetique")


def test_fingerprint_is_stable():
    assert load_config(ROOT / "params.yaml").fingerprint == load_config(ROOT / "params.yaml").fingerprint
