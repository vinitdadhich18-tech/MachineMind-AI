"""
test_bridge_forwarder.py - Unit tests for MQTT Dashboard Bridge HTTP forwarding, auth, and fault tolerance.
"""

import sys
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import requests

# Ensure ml-service root is in sys.path
ML_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

from src.ingestion_consumer import TelemetryRecord
from src.mqtt_dashboard_bridge import (
    forward_state_to_backend,
    write_state,
    STATE_FILE,
    reset_latest_written_sequence
)


@pytest.fixture(autouse=True)
def clean_bridge_state(tmp_path, monkeypatch):
    """Isolates STATE_FILE to a temporary directory for each test."""
    reset_latest_written_sequence()
    temp_state = tmp_path / "live_telemetry.json"
    monkeypatch.setattr("src.mqtt_dashboard_bridge.STATE_FILE", temp_state)
    monkeypatch.setattr("src.mqtt_dashboard_bridge.FRONTEND_DATA_DIR", tmp_path)
    yield
    reset_latest_written_sequence()


def make_dummy_state(sequence=5, machine_id="ims_set2_rig", is_replay_reset=False):
    """Creates a sample processed dashboard state dict."""
    return {
        "machine_id": machine_id,
        "snapshot_sequence": sequence,
        "original_timestamp": "2004-02-12T10:32:39Z",
        "ingest_timestamp": "2026-10-25T12:00:00Z",
        "sampling_rate_hz": 20480.0,
        "sample_count": 20480,
        "source": "replayed_nasa_ims",
        "feature_count": 36,
        "channels": [{"channel": i, "rms": 0.1 * i} for i in range(1, 5)],
        "ingestion": {
            "is_duplicate": False,
            "is_out_of_order": False,
            "is_sequence_gap": False,
            "processing_latency_ms": 12.0,
            "is_replay_reset": is_replay_reset
        },
        "dashboard_updated_at": 1700000000.0
    }


def make_dummy_record(sequence=5, machine_id="ims_set2_rig", is_replay_reset=False):
    """Creates a TelemetryRecord with 20480 samples x 4 channels."""
    channels = {f"ch{i}": np.random.normal(0, 0.1, 20480).astype(np.float64) for i in range(1, 5)}
    return TelemetryRecord(
        machine_id=machine_id,
        snapshot_sequence=sequence,
        original_timestamp="2004-02-12T10:32:39Z",
        ingest_timestamp="2026-10-25T12:00:00Z",
        sampling_rate_hz=20480.0,
        sample_count=20480,
        source="replayed_nasa_ims",
        channels=channels,
        ingestion_metadata={
            "is_duplicate": False,
            "is_out_of_order": False,
            "is_sequence_gap": False,
            "processing_latency_ms": 15.0,
            "is_replay_reset": is_replay_reset
        }
    )


# ============================================================
# 1. FORWARD_STATE_TO_BACKEND TESTS
# ============================================================

def test_forward_state_no_backend_url(monkeypatch):
    """When BACKEND_URL is not set, forward_state_to_backend returns False without network requests."""
    monkeypatch.setenv("BACKEND_URL", "")
    with patch("requests.post") as mock_post:
        result = forward_state_to_backend(make_dummy_state(), backend_url="")
        assert result is False
        mock_post.assert_not_called()


def test_forward_state_successful_with_auth_header():
    """When BACKEND_URL and API_KEY are configured, sends POST with X-API-KEY and returns True."""
    state = make_dummy_state(sequence=12)
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = forward_state_to_backend(
            state,
            backend_url="https://api.example.com",
            api_key="secret-key-123",
            timeout=2.5
        )

        assert result is True
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.example.com/api/telemetry/live"
        assert kwargs["json"] == state
        assert kwargs["timeout"] == 2.5
        assert kwargs["headers"]["X-API-KEY"] == "secret-key-123"
        assert kwargs["headers"]["Content-Type"] == "application/json"


def test_forward_state_omits_api_key_when_empty():
    """When API_KEY is empty, X-API-KEY header is not included."""
    state = make_dummy_state(sequence=3)
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = forward_state_to_backend(
            state,
            backend_url="https://api.example.com",
            api_key="",
            timeout=3.0
        )

        assert result is True
        _, kwargs = mock_post.call_args
        assert "X-API-KEY" not in kwargs["headers"]


def test_forward_state_auth_error_401():
    """When backend returns 401 Unauthorized, returns False without raising exception."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401

    with patch("requests.post", return_value=mock_resp):
        result = forward_state_to_backend(
            make_dummy_state(),
            backend_url="https://api.example.com",
            api_key="invalid-key"
        )
        assert result is False


def test_forward_state_conflict_409():
    """When backend returns 409 Conflict (sequence duplicate/out of order), returns False without raising."""
    mock_resp = MagicMock()
    mock_resp.status_code = 409

    with patch("requests.post", return_value=mock_resp):
        result = forward_state_to_backend(
            make_dummy_state(sequence=5),
            backend_url="https://api.example.com"
        )
        assert result is False


def test_forward_state_server_error_500():
    """When backend returns 500/503, returns False without raising exception."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503

    with patch("requests.post", return_value=mock_resp):
        result = forward_state_to_backend(
            make_dummy_state(),
            backend_url="https://api.example.com"
        )
        assert result is False


def test_forward_state_timeout_handled_gracefully():
    """When HTTP request times out, logs warning and returns False without crashing."""
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Read timed out")):
        result = forward_state_to_backend(
            make_dummy_state(),
            backend_url="https://api.example.com",
            timeout=1.0
        )
        assert result is False


def test_forward_state_connection_error_handled_gracefully():
    """When backend server is unreachable (ConnectionError), returns False without crashing."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        result = forward_state_to_backend(
            make_dummy_state(),
            backend_url="https://api.example.com"
        )
        assert result is False


# ============================================================
# 2. WRITE_STATE INTEGRATION & LOCAL FALLBACK TESTS
# ============================================================

def test_write_state_preserves_local_file_when_backend_fails(monkeypatch, tmp_path):
    """Even if HTTP backend forwarding fails or times out, local JSON file is still written accurately."""
    monkeypatch.setenv("BACKEND_URL", "https://unreachable-backend.example.com")
    monkeypatch.setenv("API_KEY", "test-key")

    target_file = tmp_path / "live_telemetry.json"
    monkeypatch.setattr("src.mqtt_dashboard_bridge.STATE_FILE", target_file)

    record = make_dummy_record(sequence=15, machine_id="ims_set2_rig")

    # Simulate network failure during forwarding
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Server unreachable")):
        # Must not raise exception
        write_state(record)

    # Verify local file was written despite backend failure
    assert target_file.exists()
    with open(target_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data["machine_id"] == "ims_set2_rig"
    assert saved_data["snapshot_sequence"] == 15
    assert saved_data["feature_count"] == 36
    assert len(saved_data["channels"]) == 4


def test_write_state_forwards_is_replay_reset_flag(monkeypatch):
    """Verifies that is_replay_reset metadata flag is forwarded in state to the backend."""
    monkeypatch.setenv("BACKEND_URL", "https://api.example.com")
    record = make_dummy_record(sequence=0, is_replay_reset=True)

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        write_state(record)

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        forwarded_state = kwargs["json"]
        assert forwarded_state["snapshot_sequence"] == 0
        assert forwarded_state["ingestion"]["is_replay_reset"] is True
