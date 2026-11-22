"""POST /explain : même entrée que /predict, une explication par groupe de variables, cohérente avec /predict."""


def request_body(visits):
    return {"visits": visits, "patient": {"male": 1, "age_bl": 63}, "horizon_months": 12, "target_state": "OFF"}


def test_explain_is_consistent_with_predict(client, key, visits):
    response = client.post("/explain", json=request_body(visits), headers=key)
    predicted = client.post("/predict", json=request_body(visits), headers=key).json()
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == predicted["prediction"]
    assert body["current_score"] == visits[-1]["np3"]
    assert body["predicted_change"] == round(body["prediction"] - body["current_score"], 2)
    assert {"UPDRS III (historique)", "Temps"} <= {e["group"] for e in body["effects"]}
    assert "non diagnostique" in body["disclaimer"] and "pas une cause" in body["method"]


def test_explain_requires_the_api_key(client, visits):
    assert client.post("/explain", json=request_body(visits)).status_code == 401
