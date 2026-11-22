"""API (objectif 2) de bout en bout avec le modèle entraîné sur données synthétiques."""


def request_body(visits):
    return {"visits": visits, "patient": {"male": 1, "age_bl": 63}, "horizon_months": 12, "target_state": "OFF"}


def test_health_reports_the_three_objectives(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "state_model_loaded": True,
                                            "factors_loaded": True}


def test_predict_returns_value_interval_and_disclaimer(client, key, visits):
    body = client.post("/predict", json=request_body(visits), headers=key).json()
    assert 0 <= body["prediction"] <= 132
    assert body["interval_80"][0] <= body["prediction"] <= body["interval_80"][1]
    assert "non diagnostique" in body["disclaimer"]


def test_missing_api_key_is_refused(client, visits):
    assert client.post("/predict", json=request_body(visits)).status_code == 401


def test_invalid_input_is_rejected(client, key, visits):
    bad = {**request_body(visits), "horizon_months": 99}
    assert client.post("/predict", json=bad, headers=key).status_code == 422
