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


@patch("requests.request")
def test_get_latest_telemetry_success(mock_request):
    """Verifies get_latest_telemetry returns data on HTTP 200."""
    from frontend.services.api_client import get_latest_telemetry
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": {"machine_id": "ims_set2_rig", "snapshot_sequence": 10},
        "error": None
    }
    mock_request.return_value = mock_resp

    res = get_latest_telemetry("ims_set2_rig")
    assert res is not None
    assert res["machine_id"] == "ims_set2_rig"
    assert res["snapshot_sequence"] == 10


@patch("requests.request")
def test_get_latest_telemetry_not_found_returns_none(mock_request):
    """Verifies get_latest_telemetry returns None when backend responds with 404 TELEMETRY_NOT_FOUND."""
    from frontend.services.api_client import get_latest_telemetry
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {
        "success": False,
        "data": None,
        "error": {
            "code": "TELEMETRY_NOT_FOUND",
            "message": "No telemetry recorded yet for machine 'ims_set2_rig'."
        }
    }
    mock_request.return_value = mock_resp

    res = get_latest_telemetry("ims_set2_rig")
    assert res is None


@patch("requests.request")
def test_api_client_dynamic_backend_url(mock_request, monkeypatch):
    """Verifies that _request dynamically evaluates get_backend_url() at runtime."""
    from frontend.services.api_client import get_latest_telemetry

    monkeypatch.setenv("BACKEND_URL", "https://dynamic-render-backend.example.com")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": {"machine_id": "ims_set2_rig", "snapshot_sequence": 99},
        "error": None
    }
    mock_request.return_value = mock_resp

    res = get_latest_telemetry("ims_set2_rig")
    assert res["snapshot_sequence"] == 99

    called_url = mock_request.call_args[1]["url"]
    assert called_url.startswith("https://dynamic-render-backend.example.com/api/telemetry/latest")


def test_get_config_val_secrets_and_casing(monkeypatch):
    """Verifies config resolution across casing and st.secrets nested sections."""
    import streamlit as st
    from frontend.utils.config import get_config_val

    # Clear env
    monkeypatch.delenv("CUSTOM_KEY", raising=False)
    monkeypatch.delenv("custom_key", raising=False)

    # 1. Lowercase env
    monkeypatch.setenv("custom_key", "from_lower_env")
    assert get_config_val("CUSTOM_KEY", "default") == "from_lower_env"
    monkeypatch.delenv("custom_key")

    # 2. Nested secrets section
    mock_secrets = {"general": {"custom_key": "https://secret-url.example.com"}}
    monkeypatch.setattr(st, "secrets", mock_secrets)
    assert get_config_val("CUSTOM_KEY", "default") == "https://secret-url.example.com"
