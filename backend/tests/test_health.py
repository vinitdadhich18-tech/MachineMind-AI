"""
test_health.py - Tests for GET /api/health endpoint.
"""


def test_health_endpoint_contract(client):
    """Verifies response status, JSON envelope, and field structure of /api/health."""
    response = client.get("/api/health")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    assert payload["error"] is None

    data = payload["data"]
    assert data["service"] == "machinemind-backend"
    assert data["status"] in ["ok", "degraded"]
    assert data["database"] in ["connected", "unavailable"]
    assert isinstance(data["model"], dict)
    assert "loaded" in data["model"]
    assert "version" in data["model"]
    assert "time" in data


def test_404_handler_returns_json_envelope(client):
    """Verifies that unknown routes return 404 with standard JSON envelope without HTML."""
    response = client.get("/api/unknown_route")
    assert response.status_code == 404

    payload = response.get_json()
    assert payload["success"] is False
    assert payload["data"] is None
    assert payload["error"]["code"] == "NOT_FOUND"
