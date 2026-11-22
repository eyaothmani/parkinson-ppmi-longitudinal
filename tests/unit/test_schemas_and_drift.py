"""Schémas de l'API (rejet des entrées invalides) et indicateur de dérive."""
import numpy as np
import pytest
from pydantic import ValidationError

from parkinson_mm.monitoring.drift import psi
from parkinson_mm.serving.schemas import PredictRequest

VALID = {"visits": [{"time_years": 0, "np3": 20, "med_state": "untreated"},
                    {"time_years": 1, "np3": 24, "med_state": "OFF"}],
         "horizon_months": 12, "target_state": "OFF"}


def test_valid_request():
    assert PredictRequest(**VALID).horizon_months == 12


@pytest.mark.parametrize("change", [
    {"visits": [{"time_years": 0, "np3": 200, "med_state": "OFF"}]},                     # hors échelle
    {"visits": list(reversed(VALID["visits"]))},                                         # ordre incohérent
    {"horizon_months": 60},                                                              # horizon non appris
    {"patient_name": "Jean"},                                                            # champ inattendu
])
def test_invalid_requests_are_rejected(change):
    with pytest.raises(ValidationError):
        PredictRequest(**{**VALID, **change})


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    edges = np.quantile(rng.normal(0, 1, 5000), np.linspace(0, 1, 11)).tolist()
    assert psi(rng.normal(0, 1, 2000), edges) < 0.05
    assert psi(rng.normal(2, 1, 2000), edges) > 0.25
