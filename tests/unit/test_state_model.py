"""Score d'état : le filtrage ne doit utiliser que le passé, et reproduire le BLUP de statsmodels."""
import numpy as np
import pandas as pd
import pytest

from parkinson_mm.stats.design import design_matrix
from parkinson_mm.stats.lme import BASE_ADJUSTMENT, fit_lme
from parkinson_mm.stats.state import StateModel


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    rows = []
    for patno in range(60):
        slope, level = rng.normal(2, 1.5), rng.normal(0, 6)
        age, duration = rng.normal(), rng.normal()           # covariables de base : fixes par patient
        for t in np.arange(0, 5, 0.5):
            med = "ON" if t > 2 else "untreated"
            rows.append({"PATNO": patno, "time_years": t, "med": med, "age_c": age, "male": patno % 2,
                         "dur_c": duration, "np3": 24 + level + slope * t - 5 * (med == "ON") + rng.normal(0, 3)})
    d = pd.DataFrame(rows)
    return fit_lme("np3 ~ time_years + " + BASE_ADJUSTMENT, d), d


def test_design_matrix_matches_statsmodels(fitted):
    result, d = fitted
    X = design_matrix(list(result.fe_params.index), result.model.data.frame)
    assert np.allclose(X, result.model.exog)


def test_posterior_equals_statsmodels_blup(fitted):
    result, d = fitted
    model = StateModel(result, {"age_c": 0.0, "male": 0.0, "dur_c": 0.0})
    history = d[d["PATNO"] == 3]
    assert np.allclose(model.posterior(history)[0], np.asarray(result.random_effects[3]), atol=1e-6)


def test_state_ignores_future_visits(fitted):
    result, d = fitted
    model = StateModel(result, {"age_c": 0.0, "male": 0.0, "dur_c": 0.0})
    past = d[(d["PATNO"] == 3) & (d["time_years"] <= 2)]
    assert model.state(past)["time_years"] == 2.0                      # état à la dernière visite connue
    assert model.state(past)["niveau_sd"] > model.state(d[d["PATNO"] == 3])["niveau_sd"]   # + de visites = + précis


def test_forecast_uncertainty_includes_measurement_noise(fitted):
    result, d = fitted
    model = StateModel(result, {"age_c": 0.0, "male": 0.0, "dur_c": 0.0})
    history = d[(d["PATNO"] == 3) & (d["time_years"] <= 2)]
    future = history.tail(1).assign(time_years=3.0)
    assert model.predict(history, future, noise=True)[1][0] > model.predict(history, future, noise=False)[1][0]


def test_missing_covariate_is_filled_with_dev_median(fitted):
    result, d = fitted
    model = StateModel(result, {"age_c": 0.0, "male": 0.0, "dur_c": 0.0})
    history = d[d["PATNO"] == 3].assign(age_c=np.nan)
    assert np.isfinite(model.state(history)["niveau_actuel"])
