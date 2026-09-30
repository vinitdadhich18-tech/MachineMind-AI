"""
test_api_client.py - Unit tests for Streamlit HTTP API client.
"""

import pytest
from unittest.mock import patch, MagicMock
import requests

from frontend.services.api_client import (
    get_health,
    list_machines,
    create_machine,
    run_inference,
    get_predictions,
    list_alerts,
    update_alert_status,
    ApiError
)


@patch("requests.request")
def test_api_client_success_envelope(mock_request):
    """Verifies parsing of standard backend success envelopes."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": {"service": "machinemind-backend", "status": "ok", "database": "connected"},
        "error": None
    }
    mock_request.return_value = mock_resp

    res = get_health()
    assert res["service"] == "machinemind-backend"
    assert res["status"] == "ok"
    assert res["database"] == "connected"


@patch("requests.request")
def test_api_client_error_envelope(mock_request):
    """Verifies that backend error envelopes raise ApiError with code and message."""
    mock_resp = MagicMock()
    mock_resp.status_code = 409
    mock_resp.json.return_value = {
        "success": False,
        "data": None,
        "error": {
            "code": "MACHINE_EXISTS",
            "message": "Machine already exists.",
            "details": {"machine_id": "dup_rig"}
        }
    }
    mock_request.return_value = mock_resp

    with pytest.raises(ApiError) as exc:
        create_machine("dup_rig", "Duplicate Rig")

    assert exc.value.code == "MACHINE_EXISTS"
    assert "Machine already exists" in exc.value.message
    assert exc.value.details["machine_id"] == "dup_rig"


@patch("requests.request")
def test_api_client_connection_refused(mock_request):
    """Verifies friendly ApiError when backend connection is refused."""
    mock_request.side_effect = requests.exceptions.ConnectionError("Connection refused")

    with pytest.raises(ApiError) as exc:
        run_inference("test_mach", b"dummy_content", "test.csv")

    assert exc.value.code == "CONNECTION_ERROR"
    assert "Backend server is unreachable" in exc.value.message


@patch("requests.request")
def test_api_client_non_json_response(mock_request):
    """Verifies graceful handling when backend returns non-JSON text (e.g. 502 Bad Gateway HTML)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.json.side_effect = ValueError("Invalid JSON")
    mock_resp.text = "<html>502 Bad Gateway</html>"
    mock_request.return_value = mock_resp

    with pytest.raises(ApiError) as exc:
        list_machines()

    assert exc.value.code == "INVALID_RESPONSE"
    assert "Backend returned non-JSON response" in exc.value.message
