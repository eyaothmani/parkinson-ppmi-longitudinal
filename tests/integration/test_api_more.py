"""API : /summary (résultats agrégés), clés d'API des nouvelles routes et tableau de bord servi sur /app/."""
import pytest


def test_summary_gives_aggregated_results_only(client, key):
    body = client.get("/summary", headers=key).json()
    assert {"ingest", "split", "state", "cv", "stats"} <= set(body) and "non diagnostique" in body["disclaimer"]
    assert "PATNO" not in str(body)


@pytest.mark.parametrize("route, method", [("/summary", "get"), ("/rag/ask", "post")])
def test_new_routes_need_the_api_key(client, route, method):
    assert getattr(client, method)(route).status_code == 401


def test_dashboard_is_served_without_key(client):
    assert client.get("/", follow_redirects=False).headers["location"] == "/app/"
    page = client.get("/app/")
    assert page.status_code == 200 and "Trajectoires PPMI" in page.text
    assert client.get("/app/js/main.js").headers["content-type"].startswith("text/javascript")
