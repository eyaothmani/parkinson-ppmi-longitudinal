"""API : objectif 1 (/state) et objectif 3 (/factors)."""
import pytest


def test_state_gives_level_speed_and_projection(client, key, visits):
    request = {"visits": visits, "patient": {"male": 1, "age_bl": 63, "disease_dur_bl": 1.0}}
    body = client.post("/state", json=request, headers=key).json()
    low, high = body["current_level_ci95"]
    assert low <= body["current_level"] <= high
    assert body["speed_ci95"][0] <= body["speed_pts_per_year"] <= body["speed_ci95"][1]
    assert [p["months"] for p in body["projection"]] == [0, 6, 12, 18, 24]
    assert all(p["lower"] <= p["level"] <= p["upper"] for p in body["projection"])
    assert "non diagnostique" in body["disclaimer"]


def test_state_works_with_a_single_visit_and_no_patient_info(client, key):
    one = {"visits": [{"time_years": 0, "np3": 22, "med_state": "untreated"}]}
    body = client.post("/state", json=one, headers=key).json()
    assert body["n_visits"] == 1 and body["current_level_ci95"][1] > body["current_level_ci95"][0]


def test_factors_are_aggregated_results_only(client, key):
    body = client.get("/factors", headers=key).json()
    assert body["factors"] and "PATNO" not in body["factors"][0]
    assert {"facteur", "effet_pente_pts_an", "p_FDR", "significatif_FDR"} <= set(body["factors"][0])
    assert body["mean_slope_pts_per_year"] > 0


@pytest.mark.parametrize("route, method", [("/state", "post"), ("/factors", "get")])
def test_new_routes_need_the_api_key(client, route, method):
    assert getattr(client, method)(route).status_code == 401


def test_state_rejects_unordered_visits(client, key, visits):
    assert client.post("/state", json={"visits": visits[::-1]}, headers=key).status_code == 422
