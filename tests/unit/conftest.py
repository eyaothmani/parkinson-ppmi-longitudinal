"""Petites données construites à la main pour les tests unitaires (lisibles et rapides)."""
import pandas as pd
import pytest


@pytest.fixture
def visits():
    """Deux patients, visites mensuelles irrégulières, scores connus."""
    return pd.DataFrame({
        "PATNO": [1, 1, 1, 2, 2],
        "visit_date": pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01", "2020-01-01", "2021-01-01"]),
        "months_from_bl": [0, 6, 12, 0, 12],
        "time_years": [0, 0.5, 1.0, 0, 1.0],
        "np3": [20.0, 22.0, 26.0, 30.0, 31.0],
        "med_state": ["untreated", "OFF", "OFF", "ON", "ON"],
    })
