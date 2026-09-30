"""
test_ml_service.py - Unit tests for ML integration layer and engine mapping.
"""

import pytest
import numpy as np

from backend.config import Config
from backend.services.ml_service import (
    init_ml_service,
    is_model_loaded,
    get_model_version,
    run_inference,
    _machine_engines
)


@pytest.fixture(autouse=True)
def setup_ml():
    """Initializes ML service using real models directory for unit testing."""
    _machine_engines.clear()
    success = init_ml_service(Config.MODEL_DIR, Config.ML_SERVICE_PATH)
    assert success is True, "Failed to initialize ML service with real models"


def test_ml_service_initialization():
    """Verifies that ML service reports loaded status and version."""
    assert is_model_loaded() is True
    assert isinstance(get_model_version(), str)
    assert len(get_model_version()) > 0


def test_run_inference_output_mapping():
    """Verifies that run_inference output matches the target API contract schema exactly."""
    # Generate a clean 20480x4 snapshot (zeros)
    snapshot = np.zeros((20480, 4), dtype=np.float64)

    result = run_inference(
        machine_id="test_machine_01",
        snapshot=snapshot,
        timestamp="2026-01-31T12:00:00Z",
        source_filename="2004.02.12.10.32.39"
    )

    # Check top-level contract structure
    assert result["machine_id"] == "test_machine_01"
    assert result["timestamp"] == "2026-01-31T12:00:00Z"
    assert result["source_filename"] == "2004.02.12.10.32.39"
    assert result["model_version"] == get_model_version()
    assert "prediction_id" in result

    # Check overall status object
    overall = result["overall"]
    assert overall["state"] in ["normal", "watch", "anomaly_detected"]
    assert isinstance(overall["snapshot_flagged"], bool)
    assert isinstance(overall["persistence_confirmed"], bool)
    assert overall["label"] in [
        "Normal",
        "Anomalous snapshot — awaiting persistence confirmation",
        "Vibration anomaly detected"
    ]

    # Check 4 channels
    channels = result["channels"]
    assert len(channels) == 4
    for idx, ch in enumerate(channels, start=1):
        assert ch["channel"] == idx
        assert ch["name"] == f"Channel {idx}"
        assert isinstance(ch["anomaly_score"], float)
        assert isinstance(ch["threshold"], float)
        assert isinstance(ch["snapshot_flagged"], bool)
        assert isinstance(ch["consecutive_flagged_count"], int)
        assert isinstance(ch["persistence_confirmed"], bool)
        assert ch["state"] in ["normal", "watch", "anomaly_detected"]

    # Check persistence metadata
    assert result["persistence"]["required_consecutive_snapshots"] == 3

    # Check alert structure default
    assert result["alert"]["created"] is False
    assert result["alert"]["alert_id"] is None


def test_per_machine_state_isolation():
    """Verifies that per-machine engines maintain independent exceedance state counters."""
    snapshot = np.zeros((20480, 4), dtype=np.float64)

    res_a = run_inference("machine_A", snapshot)
    res_b = run_inference("machine_B", snapshot)

    assert res_a["machine_id"] == "machine_A"
    assert res_b["machine_id"] == "machine_B"
    assert "machine_A" in _machine_engines
    assert "machine_B" in _machine_engines
    assert _machine_engines["machine_A"] is not _machine_engines["machine_B"]
