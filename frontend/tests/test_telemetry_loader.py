"""
test_telemetry_loader.py - Comprehensive tests for telemetry loading, validation, staleness, and local fallback.
"""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from frontend.services.api_client import ApiError
from frontend.services.telemetry_loader import (
    validate_telemetry_payload,
    load_local_telemetry_file,
    calculate_telemetry_age,
    fetch_telemetry_stream
)


def make_sample_telemetry(machine_id="ims_set2_rig", sequence=10, age_offset_seconds=5.0):
    """Helper creating a sample 36-feature telemetry state."""
    channels = []
    for ch in range(1, 5):
        channels.append({
            "channel": ch,
            "mean": 0.01 * ch,
            "std": 0.05 * ch,
            "rms": 0.08 * ch,
            "p2p": 0.5 * ch,
            "skewness": 0.02 * ch,
            "kurtosis": 3.0 * ch,
            "crest_factor": 3.1 * ch,
            "spectral_energy": 10.0 * ch,
            "spectral_centroid": 1500.0 * ch
        })

    updated_ts = time.time() - age_offset_seconds

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
            "is_duplicate": False,
            "is_out_of_order": False,
            "is_sequence_gap": False,
            "processing_latency_ms": 12.5,
            "is_replay_reset": False
        },
        "dashboard_updated_at": updated_ts
    }


# ============================================================
# 1. PAYLOAD VALIDATION & STALENESS TESTS
# ============================================================

def test_validate_telemetry_payload_valid():
    """Valid payload is accepted and returned."""
    sample = make_sample_telemetry()
    assert validate_telemetry_payload(sample) == sample


def test_validate_telemetry_payload_invalid_structure():
    """Non-dict, missing channels, bad channel count, or non-finite values return None."""
    assert validate_telemetry_payload("string") is None
    assert validate_telemetry_payload(None) is None

    # Missing sequence
    bad_seq = make_sample_telemetry()
    bad_seq["snapshot_sequence"] = -1
    assert validate_telemetry_payload(bad_seq) is None

    # Wrong channel count
    bad_ch = make_sample_telemetry()
    bad_ch["channels"] = bad_ch["channels"][:2]
    assert validate_telemetry_payload(bad_ch) is None

    # Missing metric in channel
    bad_metric = make_sample_telemetry()
    del bad_metric["channels"][0]["rms"]
    assert validate_telemetry_payload(bad_metric) is None

    # Non-finite metric
    bad_val = make_sample_telemetry()
    bad_val["channels"][0]["rms"] = "invalid"
    assert validate_telemetry_payload(bad_val) is None


def test_validate_telemetry_payload_float_coercion():
    """Verifies that float sequence numbers and channel IDs from JSON/DB are normalized to int."""
    sample = make_sample_telemetry()
    sample["snapshot_sequence"] = 99.0
    for ch in sample["channels"]:
        ch["channel"] = float(ch["channel"])

    validated = validate_telemetry_payload(sample)
    assert validated is not None
    assert isinstance(validated["snapshot_sequence"], int)
    assert validated["snapshot_sequence"] == 99
    for ch in validated["channels"]:
        assert isinstance(ch["channel"], int)



def test_calculate_telemetry_age():
    """Calculates age correctly and flags staleness beyond threshold."""
    now = 1000.0

    # Fresh (5 seconds old)
    fresh_sample = {"dashboard_updated_at": now - 5.0}
    is_stale, age, age_str, ts_str = calculate_telemetry_age(fresh_sample, stale_threshold=30.0, current_time=now)
    assert is_stale is False
    assert abs(age - 5.0) < 0.1
    assert "5.0s ago" in age_str

    # Stale (60 seconds old)
    stale_sample = {"dashboard_updated_at": now - 60.0}
    is_stale, age, age_str, ts_str = calculate_telemetry_age(stale_sample, stale_threshold=30.0, current_time=now)
    assert is_stale is True
    assert abs(age - 60.0) < 0.1
    assert "1.0m ago" in age_str


# ============================================================
# 2. STREAM RESOLVER & SOURCE-SELECTION POLICY TESTS
# ============================================================

@patch("frontend.services.telemetry_loader.get_latest_telemetry")
def test_remote_success_fresh(mock_get_latest):
    """When remote backend returns fresh telemetry, source is 'backend' and status is 'fresh'."""
    sample = make_sample_telemetry(sequence=42, age_offset_seconds=2.0)
    mock_get_latest.return_value = sample

    res = fetch_telemetry_stream(machine_id="ims_set2_rig", stale_threshold=30.0)
    assert res["telemetry"]["snapshot_sequence"] == 42
    assert res["source"] == "backend"
    assert res["status"] == "fresh"
    assert res["is_stale"] is False


@patch("frontend.services.telemetry_loader.get_latest_telemetry")
def test_remote_success_stale(mock_get_latest):
    """When remote backend returns telemetry older than stale_threshold, status is 'stale'."""
    sample = make_sample_telemetry(sequence=42, age_offset_seconds=120.0)
    mock_get_latest.return_value = sample

    res = fetch_telemetry_stream(machine_id="ims_set2_rig", stale_threshold=30.0)
    assert res["telemetry"]["snapshot_sequence"] == 42
    assert res["source"] == "backend"
    assert res["status"] == "stale"
    assert res["is_stale"] is True


@patch("frontend.services.telemetry_loader.get_latest_telemetry")
def test_remote_no_telemetry_with_no_local_fallback(mock_get_latest, tmp_path):
    """When backend returns None (404 TELEMETRY_NOT_FOUND) and no local file exists, status is 'no_telemetry'."""
    mock_get_latest.return_value = None
    empty_file = tmp_path / "nonexistent.json"

    res = fetch_telemetry_stream(machine_id="ims_set2_rig", local_file=empty_file)
    assert res["telemetry"] is None
    assert res["source"] is None
    assert res["status"] == "no_telemetry"
    assert "Waiting for live telemetry stream" in res["message"]


@patch("frontend.services.telemetry_loader.get_latest_telemetry")
def test_remote_no_telemetry_with_local_fallback(mock_get_latest, tmp_path):
    """When backend returns None (no data yet) but a local file exists, falls back to local file with 'local_fallback' status."""
    mock_get_latest.return_value = None

    local_file = tmp_path / "live_telemetry.json"
    local_sample = make_sample_telemetry(sequence=5)
    local_file.write_text(json.dumps(local_sample), encoding="utf-8")

    res = fetch_telemetry_stream(machine_id="ims_set2_rig", local_file=local_file)
    assert res["telemetry"]["snapshot_sequence"] == 5
    assert res["source"] == "local_fallback"
    assert res["status"] == "local_fallback"


@patch("frontend.services.telemetry_loader.get_latest_telemetry")
def test_backend_timeout_with_local_fallback(mock_get_latest, tmp_path):
    """When backend request times out and local file exists, returns local fallback with 'backend_unavailable_fallback_used'."""
    mock_get_latest.side_effect = ApiError("TIMEOUT_ERROR", "Request timed out")

    local_file = tmp_path / "live_telemetry.json"
    local_sample = make_sample_telemetry(sequence=8)
    local_file.write_text(json.dumps(local_sample), encoding="utf-8")

    res = fetch_telemetry_stream(machine_id="ims_set2_rig", local_file=local_file)
    assert res["telemetry"]["snapshot_sequence"] == 8
    assert res["source"] == "local_fallback"
    assert res["status"] == "backend_unavailable_fallback_used"
    assert "Backend temporarily unavailable" in res["message"]


@patch("frontend.services.telemetry_loader.get_latest_telemetry")
def test_backend_timeout_with_no_local_fallback(mock_get_latest, tmp_path):
    """When backend request times out and NO local file exists, returns 'backend_unavailable'."""
    mock_get_latest.side_effect = ApiError("TIMEOUT_ERROR", "Request timed out")
    empty_file = tmp_path / "nonexistent.json"

    res = fetch_telemetry_stream(machine_id="ims_set2_rig", local_file=empty_file)
    assert res["telemetry"] is None
    assert res["source"] is None
    assert res["status"] == "backend_unavailable"
    assert "currently unreachable" in res["message"]


@patch("frontend.services.telemetry_loader.get_latest_telemetry")
def test_backend_server_error_with_local_fallback(mock_get_latest, tmp_path):
    """When backend raises 503 DATABASE_UNAVAILABLE and local file exists, uses local fallback."""
    mock_get_latest.side_effect = ApiError("DATABASE_UNAVAILABLE", "MongoDB database connection is unavailable.")

    local_file = tmp_path / "live_telemetry.json"
    local_sample = make_sample_telemetry(sequence=99)
    local_file.write_text(json.dumps(local_sample), encoding="utf-8")

    res = fetch_telemetry_stream(machine_id="ims_set2_rig", local_file=local_file)
    assert res["telemetry"]["snapshot_sequence"] == 99
    assert res["source"] == "local_fallback"
    assert res["status"] == "backend_unavailable_fallback_used"


@patch("frontend.services.telemetry_loader.get_latest_telemetry")
def test_remote_malformed_response_falls_back_to_local(mock_get_latest, tmp_path):
    """When backend returns a malformed/partial dictionary, loader gracefully falls back to local file."""
    mock_get_latest.return_value = {"bad": "data"}

    local_file = tmp_path / "live_telemetry.json"
    local_sample = make_sample_telemetry(sequence=7)
    local_file.write_text(json.dumps(local_sample), encoding="utf-8")

    res = fetch_telemetry_stream(machine_id="ims_set2_rig", local_file=local_file)
    assert res["telemetry"]["snapshot_sequence"] == 7
    assert res["source"] == "local_fallback"


@patch("frontend.services.telemetry_loader.get_latest_telemetry")
def test_neither_remote_nor_local_available(mock_get_latest, tmp_path):
    """When backend connection fails and local file is missing, status is 'backend_unavailable' and telemetry is None."""
    mock_get_latest.side_effect = ApiError("CONNECTION_ERROR", "Backend unreachable")
    empty_file = tmp_path / "nonexistent.json"

    res = fetch_telemetry_stream(machine_id="ims_set2_rig", local_file=empty_file)
    assert res["telemetry"] is None
    assert res["source"] is None
    assert res["status"] == "backend_unavailable"


def test_get_telemetry_refresh_interval_ms():
    """Verifies safe autorefresh intervals for live, stale, and idle telemetry states."""
    from frontend.app import get_telemetry_refresh_interval_ms

    sample = make_sample_telemetry()

    # 1. Default (10 seconds) - live telemetry
    assert get_telemetry_refresh_interval_ms(telemetry=sample, is_stale=False, config_seconds=10) == 10000

    # 2. Default - stale telemetry (guaranteed >= 10s)
    assert get_telemetry_refresh_interval_ms(telemetry=sample, is_stale=True, config_seconds=10) == 10000

    # 3. Default - idle telemetry (no data, guaranteed >= 10s)
    assert get_telemetry_refresh_interval_ms(telemetry=None, is_stale=False, config_seconds=10) == 10000

    # 4. Configured higher interval (e.g. 15s)
    assert get_telemetry_refresh_interval_ms(telemetry=sample, is_stale=False, config_seconds=15) == 15000

    # 5. Overly aggressive config (e.g. 2s) is capped to minimum 5s for live data
    assert get_telemetry_refresh_interval_ms(telemetry=sample, is_stale=False, config_seconds=2) == 5000

    # 6. Overly aggressive config (e.g. 2s) is capped to minimum 10s for stale/idle data
    assert get_telemetry_refresh_interval_ms(telemetry=sample, is_stale=True, config_seconds=2) == 10000
    assert get_telemetry_refresh_interval_ms(telemetry=None, is_stale=False, config_seconds=2) == 10000
