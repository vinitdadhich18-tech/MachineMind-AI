"""
test_telemetry.py - Comprehensive tests for Live Telemetry ingestion, authentication, sequence-guarding, and retrieval.
"""

import copy
import pytest
from unittest.mock import patch
from pymongo.errors import PyMongoError


def make_valid_telemetry_payload(machine_id="ims_set2_rig", sequence=0, is_replay_reset=False, is_duplicate=False, is_out_of_order=False):
    """Helper creating a fully conforming canonical 36-feature telemetry payload."""
    channels = []
    for ch in range(1, 5):
        channels.append({
            "channel": ch,
            "mean": 0.05 * ch,
            "std": 0.12 * ch,
            "rms": 0.13 * ch,
            "p2p": 0.85 * ch,
            "skewness": 0.02 * ch,
            "kurtosis": 3.01 * ch,
            "crest_factor": 3.25 * ch,
            "spectral_energy": 12.5 * ch,
            "spectral_centroid": 1540.2 * ch
        })

    return {
        "machine_id": machine_id,
        "snapshot_sequence": sequence,
        "original_timestamp": "2004-02-12T10:32:39Z",
        "ingest_timestamp": "2026-10-25T12:00:00Z",
        "sampling_rate_hz": 20480.0,
        "sample_count": 20480,
        "source": "replayed_nasa_ims",
        "feature_count": 36,
        "channels": channels,
        "ingestion": {
            "is_duplicate": is_duplicate,
            "is_out_of_order": is_out_of_order,
            "is_sequence_gap": False,
            "processing_latency_ms": 14.5,
            "is_replay_reset": is_replay_reset
        },
        "dashboard_updated_at": 1700000000.0
    }


# ============================================================
# 1. AUTHENTICATION & CONFIGURATION TESTS
# ============================================================

def test_post_live_telemetry_missing_api_key_configuration(app, client):
    """When API_KEY is unset or blank in backend environment, endpoint must fail closed with 500."""
    app.config["API_KEY"] = ""
    payload = make_valid_telemetry_payload()

    res = client.post("/api/telemetry/live", json=payload, headers={"X-API-KEY": "some-key"})
    assert res.status_code == 500
    data = res.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "AUTH_CONFIGURATION_ERROR"
    # Ensure no secrets leaked in error details
    assert "some-key" not in str(data)


def test_post_live_telemetry_missing_auth_header(app, client):
    """When API_KEY is configured, requests without X-API-KEY header must return 401."""
    app.config["API_KEY"] = "super-secret-key-123"
    payload = make_valid_telemetry_payload()

    res = client.post("/api/telemetry/live", json=payload)
    assert res.status_code == 401
    data = res.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "UNAUTHORIZED"
    assert "X-API-KEY" in data["error"]["message"]


def test_post_live_telemetry_invalid_auth_header(app, client):
    """When API_KEY is configured, requests with wrong X-API-KEY header must return 401."""
    app.config["API_KEY"] = "super-secret-key-123"
    payload = make_valid_telemetry_payload()

    res = client.post("/api/telemetry/live", json=payload, headers={"X-API-KEY": "wrong-secret"})
    assert res.status_code == 401
    data = res.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "UNAUTHORIZED"
    assert "wrong-secret" not in str(data)


def test_post_live_telemetry_valid_auth_success(app, client):
    """When API_KEY matches, request is authorized and processed."""
    app.config["API_KEY"] = "super-secret-key-123"
    payload = make_valid_telemetry_payload(sequence=0)

    res = client.post("/api/telemetry/live", json=payload, headers={"X-API-KEY": "super-secret-key-123"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["error"] is None
    assert data["data"]["snapshot_sequence"] == 0
    assert data["data"]["machine_id"] == "ims_set2_rig"


# ============================================================
# 2. PAYLOAD VALIDATION TESTS
# ============================================================

def test_post_live_telemetry_non_json_body(app, client):
    """Non-JSON request body returns 400 VALIDATION_ERROR."""
    app.config["API_KEY"] = "test-key"
    res = client.post(
        "/api/telemetry/live",
        data="plain text body",
        content_type="text/plain",
        headers={"X-API-KEY": "test-key"}
    )
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_post_live_telemetry_invalid_machine_id(app, client):
    """Invalid machine_id format returns 400 VALIDATION_ERROR."""
    app.config["API_KEY"] = "test-key"
    payload = make_valid_telemetry_payload()
    payload["machine_id"] = "bad id with spaces!"

    res = client.post("/api/telemetry/live", json=payload, headers={"X-API-KEY": "test-key"})
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_post_live_telemetry_invalid_sequence_number(app, client):
    """Negative sequence number or boolean sequence returns 400 VALIDATION_ERROR."""
    app.config["API_KEY"] = "test-key"
    payload = make_valid_telemetry_payload()
    payload["snapshot_sequence"] = -5

    res = client.post("/api/telemetry/live", json=payload, headers={"X-API-KEY": "test-key"})
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "VALIDATION_ERROR"

    payload["snapshot_sequence"] = True  # bool
    res = client.post("/api/telemetry/live", json=payload, headers={"X-API-KEY": "test-key"})
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_post_live_telemetry_missing_timestamps(app, client):
    """Missing original or ingest timestamp returns 400 VALIDATION_ERROR."""
    app.config["API_KEY"] = "test-key"
    payload = make_valid_telemetry_payload()
    payload["original_timestamp"] = ""

    res = client.post("/api/telemetry/live", json=payload, headers={"X-API-KEY": "test-key"})
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_post_live_telemetry_invalid_channels_structure(app, client):
    """Channels list not containing exactly 4 channels returns 400 VALIDATION_ERROR."""
    app.config["API_KEY"] = "test-key"
    payload = make_valid_telemetry_payload()
    payload["channels"] = payload["channels"][:3]  # Only 3 channels

    res = client.post("/api/telemetry/live", json=payload, headers={"X-API-KEY": "test-key"})
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_post_live_telemetry_missing_channel_metric(app, client):
    """Channel missing one of the 9 required metrics returns 400 VALIDATION_ERROR."""
    app.config["API_KEY"] = "test-key"
    payload = make_valid_telemetry_payload()
    del payload["channels"][0]["spectral_energy"]

    res = client.post("/api/telemetry/live", json=payload, headers={"X-API-KEY": "test-key"})
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "VALIDATION_ERROR"
    assert "spectral_energy" in res.get_json()["error"]["message"]


def test_post_live_telemetry_non_finite_channel_metric(app, client):
    """Channel metric containing non-numeric or NaN value returns 400 VALIDATION_ERROR."""
    app.config["API_KEY"] = "test-key"
    payload = make_valid_telemetry_payload()
    payload["channels"][0]["rms"] = "invalid_string"

    res = client.post("/api/telemetry/live", json=payload, headers={"X-API-KEY": "test-key"})
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "VALIDATION_ERROR"


# ============================================================
# 3. SEQUENCE ORDERING, DEDUPLICATION & REPLAY RESET
# ============================================================

def test_sequence_ordering_and_duplicate_suppression(app, client):
    """Verifies that sequences must be strictly increasing, rejecting duplicates and out-of-order snapshots."""
    app.config["API_KEY"] = "test-key"
    headers = {"X-API-KEY": "test-key"}

    # 1. Sequence 10 succeeds
    res10 = client.post("/api/telemetry/live", json=make_valid_telemetry_payload(sequence=10), headers=headers)
    assert res10.status_code == 200
    assert res10.get_json()["data"]["snapshot_sequence"] == 10

    # 2. Sequence 11 succeeds
    res11 = client.post("/api/telemetry/live", json=make_valid_telemetry_payload(sequence=11), headers=headers)
    assert res11.status_code == 200
    assert res11.get_json()["data"]["snapshot_sequence"] == 11

    # 3. Duplicate Sequence 11 returns 409 DUPLICATE_SEQUENCE
    res_dup = client.post("/api/telemetry/live", json=make_valid_telemetry_payload(sequence=11), headers=headers)
    assert res_dup.status_code == 409
    assert res_dup.get_json()["error"]["code"] == "DUPLICATE_SEQUENCE"

    # 4. Out-of-order older Sequence 5 returns 409 OUT_OF_ORDER_SEQUENCE
    res_old = client.post("/api/telemetry/live", json=make_valid_telemetry_payload(sequence=5), headers=headers)
    assert res_old.status_code == 409
    assert res_old.get_json()["error"]["code"] == "OUT_OF_ORDER_SEQUENCE"


def test_ingestion_metadata_flags_rejection(app, client):
    """Payloads pre-flagged as duplicate or out_of_order by ingestion consumer are rejected with 409."""
    app.config["API_KEY"] = "test-key"
    headers = {"X-API-KEY": "test-key"}

    # Pre-flagged duplicate
    p_dup = make_valid_telemetry_payload(sequence=20, is_duplicate=True)
    res_dup = client.post("/api/telemetry/live", json=p_dup, headers=headers)
    assert res_dup.status_code == 409
    assert res_dup.get_json()["error"]["code"] == "DUPLICATE_SEQUENCE"

    # Pre-flagged out_of_order
    p_ooo = make_valid_telemetry_payload(sequence=20, is_out_of_order=True)
    res_ooo = client.post("/api/telemetry/live", json=p_ooo, headers=headers)
    assert res_ooo.status_code == 409
    assert res_ooo.get_json()["error"]["code"] == "OUT_OF_ORDER_SEQUENCE"


def test_replay_reset_cycle_transition(app, client):
    """Legitimate sequence 0 with is_replay_reset=True successfully resets high sequence."""
    app.config["API_KEY"] = "test-key"
    headers = {"X-API-KEY": "test-key"}

    # Set initial high sequence 99
    res99 = client.post("/api/telemetry/live", json=make_valid_telemetry_payload(sequence=99), headers=headers)
    assert res99.status_code == 200

    # Sequence 0 without replay_reset flag is rejected
    res0_bad = client.post("/api/telemetry/live", json=make_valid_telemetry_payload(sequence=0, is_replay_reset=False), headers=headers)
    assert res0_bad.status_code == 409
    assert res0_bad.get_json()["error"]["code"] == "OUT_OF_ORDER_SEQUENCE"

    # Sequence 0 with is_replay_reset=True succeeds
    res0_ok = client.post("/api/telemetry/live", json=make_valid_telemetry_payload(sequence=0, is_replay_reset=True), headers=headers)
    assert res0_ok.status_code == 200
    assert res0_ok.get_json()["data"]["snapshot_sequence"] == 0

    # Next sequence 1 in new cycle succeeds
    res1_ok = client.post("/api/telemetry/live", json=make_valid_telemetry_payload(sequence=1), headers=headers)
    assert res1_ok.status_code == 200
    assert res1_ok.get_json()["data"]["snapshot_sequence"] == 1


# ============================================================
# 4. RETRIEVAL (GET /api/telemetry/latest)
# ============================================================

def test_get_latest_telemetry_not_found(client):
    """GET /api/telemetry/latest for machine with no telemetry returns 404 TELEMETRY_NOT_FOUND."""
    res = client.get("/api/telemetry/latest?machine_id=unseen_rig")
    assert res.status_code == 404
    data = res.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "TELEMETRY_NOT_FOUND"


def test_get_latest_telemetry_invalid_machine_id(client):
    """GET /api/telemetry/latest with invalid machine_id returns 400 VALIDATION_ERROR."""
    res = client.get("/api/telemetry/latest?machine_id=invalid%20name%21")
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_get_latest_telemetry_success(app, client):
    """GET /api/telemetry/latest returns the latest stored state in standard JSON response envelope."""
    app.config["API_KEY"] = "test-key"
    headers = {"X-API-KEY": "test-key"}

    payload = make_valid_telemetry_payload(machine_id="ims_set2_rig", sequence=42)
    post_res = client.post("/api/telemetry/live", json=payload, headers=headers)
    assert post_res.status_code == 200

    # Retrieve latest telemetry
    get_res = client.get("/api/telemetry/latest?machine_id=ims_set2_rig")
    assert get_res.status_code == 200
    body = get_res.get_json()
    assert body["success"] is True
    assert body["error"] is None

    data = body["data"]
    assert data["machine_id"] == "ims_set2_rig"
    assert data["snapshot_sequence"] == 42
    assert len(data["channels"]) == 4
    assert data["feature_count"] == 36
    # Ensure MongoDB internal _id is NOT in data
    assert "_id" not in data


# ============================================================
# 5. DATABASE FAILURE TESTS (FAIL SAFE)
# ============================================================

def test_database_unavailable_error(app, client):
    """When get_db returns None, API raises 503 DATABASE_UNAVAILABLE."""
    app.config["API_KEY"] = "test-key"
    headers = {"X-API-KEY": "test-key"}

    with patch("backend.services.telemetry_service.get_db", return_value=None):
        # Test write endpoint
        res_post = client.post(
            "/api/telemetry/live",
            json=make_valid_telemetry_payload(),
            headers=headers
        )
        assert res_post.status_code == 503
        assert res_post.get_json()["error"]["code"] == "DATABASE_UNAVAILABLE"

        # Test read endpoint
        res_get = client.get("/api/telemetry/latest?machine_id=ims_set2_rig")
        assert res_get.status_code == 503
        assert res_get.get_json()["error"]["code"] == "DATABASE_UNAVAILABLE"


def test_database_operational_error(app, client):
    """When PyMongo raises PyMongoError during operation, API returns 503 DATABASE_ERROR."""
    app.config["API_KEY"] = "test-key"
    headers = {"X-API-KEY": "test-key"}

    with patch("backend.services.telemetry_service.get_database") as mock_get_db:
        mock_db = mock_get_db.return_value
        mock_db.live_telemetry.find_one.side_effect = PyMongoError("Connection reset by peer")

        res = client.post(
            "/api/telemetry/live",
            json=make_valid_telemetry_payload(),
            headers=headers
        )
        assert res.status_code == 503
        assert res.get_json()["error"]["code"] == "DATABASE_ERROR"
