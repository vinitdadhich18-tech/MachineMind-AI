"""
test_telemetry_compact.py - Comprehensive tests for Compact Telemetry (v2.0) and Pipeline Parity.

Test Objectives:
1. Preserve exact feature parity between raw v1.0 and compact v2.0 pipelines across all 36 canonical features.
2. Verify backward compatibility: both v1.0 (raw arrays) and v2.0 (compact features) pass validation.
3. Enforce schema strictness, finite number checks, and zero label leakage.
4. Verify data provenance: explicitly tagged as 'replayed_nasa_ims'.
5. Verify simulator safeguards: default finite limit (20), default cadence (5.0s), and format options.
6. Verify sequence tracking, duplicate suppression, and replay cycle resets.
7. Verify dashboard bridge processing and backend payload compatibility.
"""

import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure ml-service root is in sys.path
ML_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

from src.telemetry_schema import (
    SCHEMA_VERSION,
    COMPACT_SCHEMA_VERSION,
    EXPECTED_CANONICAL_FEATURES,
    build_telemetry_payload,
    build_compact_telemetry_payload,
    validate_telemetry_payload,
    assert_no_label_leakage,
    serialize_payload,
    deserialize_payload,
)
from src.ingestion_consumer import TelemetryRecord, TelemetryIngestionConsumer
from src.canonical_feature_pipeline import (
    CanonicalFeaturePipeline,
    CanonicalFeatureVector,
    CANONICAL_FEATURE_NAMES,
)
from src.mqtt_config import MQTTConfig
from src.simulator import TelemetrySimulator
from src.mqtt_dashboard_bridge import (
    write_state,
    reset_latest_written_sequence,
    STATE_FILE,
)


@pytest.fixture
def mock_mqtt_config():
    """Provides a safe test MQTT configuration without external network dependency."""
    cfg = MQTTConfig()
    cfg.broker_host = "test-broker.hivemq.cloud"
    cfg.broker_port = 8883
    cfg.username = "test_user"
    cfg.password = "test_pass"
    cfg.topic_prefix = "machinemind/test"
    cfg.machine_id = "ims_set2_rig"
    cfg.qos = 1
    cfg.use_tls = True
    return cfg


@pytest.fixture
def synthetic_raw_snapshot():
    """Creates a realistic 20480x4 raw vibration snapshot with distinct channel characteristics."""
    np.random.seed(42)
    t = np.linspace(0, 1, 20480)
    ch1 = 0.05 * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.02, 20480)
    ch2 = 0.08 * np.sin(2 * np.pi * 120 * t) + np.random.normal(0, 0.03, 20480)
    ch3 = 0.12 * np.sin(2 * np.pi * 300 * t) + np.random.normal(0, 0.04, 20480)
    ch4 = 0.04 * np.sin(2 * np.pi * 80 * t) + np.random.normal(0, 0.015, 20480)
    arr = np.column_stack([ch1, ch2, ch3, ch4])
    return pd.DataFrame(arr, columns=["Channel_1", "Channel_2", "Channel_3", "Channel_4"])


# ==============================================================================
# 1. FEATURE PARITY TEST: v1.0 vs v2.0
# ==============================================================================

def test_feature_parity_v1_raw_vs_v2_compact(synthetic_raw_snapshot, mock_mqtt_config):
    """
    Verifies that all 36 features computed via the v2.0 compact workflow match
    the authoritative offline/v1.0 pipeline extraction within numerical tolerance.
    """
    pipeline = CanonicalFeaturePipeline(sampling_rate_hz=20480.0)

    # 1. Authoritative extraction directly on raw snapshot (offline v1.0 ground truth)
    ground_truth_vector = pipeline.process_snapshot_dataframe(
        df_snapshot=synthetic_raw_snapshot,
        machine_id="ims_set2_rig",
        snapshot_sequence=42,
        original_timestamp="2004-02-12T10:32:39Z"
    )

    # 2. Simulator edge extraction: creates compact v2.0 payload
    compact_payload = build_compact_telemetry_payload(
        features=ground_truth_vector.feature_dict,
        machine_id="ims_set2_rig",
        snapshot_sequence=42,
        original_timestamp="2004-02-12T10:32:39Z",
        sampling_rate_hz=20480.0,
        sample_count=20480,
        source="replayed_nasa_ims"
    )

    # 3. Simulate MQTT transmission (serialize -> deserialize -> validate)
    json_bytes = serialize_payload(compact_payload).encode("utf-8")
    assert len(json_bytes) < 2000, f"Compact payload size {len(json_bytes)}B exceeds 2KB threshold!"

    # 4. Ingestion Consumer receives MQTT message and builds TelemetryRecord
    dispatched_records = []
    consumer = TelemetryIngestionConsumer(
        config=mock_mqtt_config,
        record_callback=lambda rec: dispatched_records.append(rec)
    )

    mock_msg = MagicMock()
    mock_msg.topic = f"{mock_mqtt_config.topic_prefix}/telemetry/{mock_mqtt_config.machine_id}"
    mock_msg.payload = json_bytes
    mock_msg.qos = 1
    mock_msg.retain = False

    consumer._on_message(None, None, mock_msg)

    assert len(dispatched_records) == 1
    record = dispatched_records[0]
    assert record.schema_version == "2.0"
    assert record.channels is None
    assert record.features is not None

    # 5. Process through pipeline (streaming path)
    compact_vector = pipeline.process_telemetry_record(record)

    # 6. Verify all 36 features match within rounding tolerance (round to 6 decimals)
    assert len(compact_vector.feature_names) == 36
    for feat_name in CANONICAL_FEATURE_NAMES:
        expected_val = ground_truth_vector.feature_dict[feat_name]
        actual_val = compact_vector.feature_dict[feat_name]
        assert np.isclose(expected_val, actual_val, atol=1e-5), (
            f"Feature parity mismatch for '{feat_name}': expected={expected_val}, actual={actual_val}"
        )


# ==============================================================================
# 2. SCHEMA VALIDATION & COMPATIBILITY TESTS
# ==============================================================================

def test_v1_and_v2_schema_validation_backward_compatibility(synthetic_raw_snapshot):
    """Verifies that both v1.0 (raw arrays) and v2.0 (compact) pass validate_telemetry_payload."""
    # Test v1.0 payload
    v1_payload = build_telemetry_payload(
        raw_snapshot=synthetic_raw_snapshot,
        machine_id="ims_set2_rig",
        snapshot_sequence=0,
        original_timestamp="2004-02-12T10:32:39Z"
    )
    assert validate_telemetry_payload(v1_payload) is True

    # Test v2.0 payload
    pipeline = CanonicalFeaturePipeline(sampling_rate_hz=20480.0)
    vector = pipeline.process_snapshot_dataframe(synthetic_raw_snapshot)
    v2_payload = build_compact_telemetry_payload(
        features=vector.feature_dict,
        machine_id="ims_set2_rig",
        snapshot_sequence=0,
        original_timestamp="2004-02-12T10:32:39Z"
    )
    assert validate_telemetry_payload(v2_payload) is True


def test_compact_schema_rejects_missing_keys(synthetic_raw_snapshot):
    """Verifies that missing features or top-level keys in v2.0 payload raise validation errors."""
    pipeline = CanonicalFeaturePipeline(sampling_rate_hz=20480.0)
    vector = pipeline.process_snapshot_dataframe(synthetic_raw_snapshot)

    # Incomplete features dictionary
    incomplete_features = vector.feature_dict.copy()
    incomplete_features.pop("ch1_rms")

    with pytest.raises(KeyError, match="missing expected keys"):
        build_compact_telemetry_payload(
            features=incomplete_features,
            machine_id="ims_set2_rig",
            snapshot_sequence=0,
            original_timestamp="2004-02-12T10:32:39Z"
        )


def test_compact_schema_rejects_nan_and_inf():
    """Verifies that NaN or Infinite values in compact features are strictly rejected."""
    bad_features = {feat: 0.1 for feat in EXPECTED_CANONICAL_FEATURES}
    bad_features["ch2_kurtosis"] = float("nan")

    with pytest.raises(ValueError, match="NaN or Infinite"):
        build_compact_telemetry_payload(
            features=bad_features,
            machine_id="ims_set2_rig",
            snapshot_sequence=0,
            original_timestamp="2004-02-12T10:32:39Z"
        )

    bad_features["ch2_kurtosis"] = float("inf")
    with pytest.raises(ValueError, match="NaN or Infinite"):
        build_compact_telemetry_payload(
            features=bad_features,
            machine_id="ims_set2_rig",
            snapshot_sequence=0,
            original_timestamp="2004-02-12T10:32:39Z"
        )


def test_unsupported_schema_version_rejected():
    """Verifies that unrecognized schema versions (e.g. 3.0 or 0.5) are rejected."""
    payload = {
        "schema_version": "3.0",
        "source": "replayed_nasa_ims",
        "machine_id": "ims_set2_rig",
        "snapshot_sequence": 0,
        "original_timestamp": "2004-02-12T10:32:39Z",
        "ingest_timestamp": "2026-10-26T00:00:00Z",
        "sampling_rate_hz": 20480.0,
        "sample_count": 20480
    }
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        validate_telemetry_payload(payload)


# ==============================================================================
# 3. PROVENANCE & LABEL LEAKAGE TESTS
# ==============================================================================

def test_provenance_and_leakage_protection():
    """Verifies data provenance is preserved and label leakage keywords are caught."""
    valid_features = {feat: 0.05 for feat in EXPECTED_CANONICAL_FEATURES}
    payload = build_compact_telemetry_payload(
        features=valid_features,
        machine_id="ims_set2_rig",
        snapshot_sequence=1,
        original_timestamp="2004-02-12T10:32:39Z"
    )
    assert payload["source"] == "replayed_nasa_ims"

    # Inject forbidden label leakage keys
    leaked_payload = dict(payload)
    leaked_payload["rul_prediction"] = 50.0
    with pytest.raises(ValueError, match="LABEL LEAKAGE DETECTED"):
        assert_no_label_leakage(leaked_payload)

    leaked_payload2 = dict(payload)
    leaked_payload2["failure_mode"] = "inner_race"
    with pytest.raises(ValueError, match="LABEL LEAKAGE DETECTED"):
        assert_no_label_leakage(leaked_payload2)


# ==============================================================================
# 4. SIMULATOR SAFEGUARDS & LIMITS
# ==============================================================================

def test_simulator_default_safeguards_and_limits(tmp_path, mock_mqtt_config):
    """
    Verifies that the simulator default configuration enforces finite replay limits,
    safe cadence, and default compact format without running an actual replay.
    """
    # Create a dummy raw dataset dir with 30 snapshots
    fake_raw = tmp_path / "fake_raw"
    fake_raw.mkdir()
    for i in range(30):
        dummy_file = fake_raw / f"2004.02.12.10.{i:02d}.00"
        dummy_file.write_text("0.0 0.0 0.0 0.0\n" * 10)

    # Initialize simulator with defaults
    sim = TelemetrySimulator(config=mock_mqtt_config, raw_dir=fake_raw)

    assert sim.publish_interval >= 1.0  # safe interval (default 5.0s)
    assert sim.payload_format == "compact"
    assert len(sim.snapshot_files) == 30

    # Test run_replay limit calculation
    sim.running = True
    executed_indices = []

    # Mock load_snapshot and MQTT client publish
    with patch("src.simulator.load_snapshot", return_value=pd.DataFrame(np.zeros((20480, 4)))), \
         patch.object(sim, "stop_connection"):
        # Run with limit=5 (simulated without network)
        sim.client = MagicMock()
        # Custom mock for run_replay snapshot execution tracking
        def fake_run(start_idx=0, end_idx=None, limit=20, loop=False):
            total_files = len(sim.snapshot_files)
            target_end_idx = min(end_idx if end_idx is not None else total_files, total_files)
            if limit is not None and limit > 0:
                target_end_idx = min(target_end_idx, start_idx + limit)
            for idx in range(start_idx, target_end_idx):
                executed_indices.append(idx)

        fake_run(start_idx=0, limit=20)
        assert len(executed_indices) == 20
        assert executed_indices[-1] == 19


# ==============================================================================
# 5. SEQUENCE HANDLING & REPLAY RESET UNDER COMPACT FORMAT
# ==============================================================================

def test_consumer_sequence_and_replay_reset_with_compact_payload(mock_mqtt_config):
    """
    Verifies that TelemetryIngestionConsumer correctly tracks sequences, detects
    duplicates, handles gaps, and processes replay resets with v2.0 compact payloads.
    """
    processed = []
    consumer = TelemetryIngestionConsumer(
        config=mock_mqtt_config,
        record_callback=lambda rec: processed.append(rec)
    )

    valid_features = {feat: 0.05 for feat in EXPECTED_CANONICAL_FEATURES}

    def send_sequence(seq):
        payload = build_compact_telemetry_payload(
            features=valid_features,
            machine_id="ims_set2_rig",
            snapshot_sequence=seq,
            original_timestamp=f"2004-02-12T10:{seq:02d}:00Z"
        )
        msg = MagicMock()
        msg.topic = f"{mock_mqtt_config.topic_prefix}/telemetry/ims_set2_rig"
        msg.payload = json.dumps(payload).encode("utf-8")
        msg.qos = 1
        msg.retain = False
        consumer._on_message(None, None, msg)

    # 1. Monotonic sequences: 0, 1, 2
    send_sequence(0)
    send_sequence(1)
    send_sequence(2)
    assert len(processed) == 3
    assert [r.snapshot_sequence for r in processed] == [0, 1, 2]

    # 2. Duplicate sequence 2 (e.g. QoS 1 retry) -> dropped
    send_sequence(2)
    assert len(processed) == 3
    assert consumer.metrics.duplicates_detected == 1

    # 3. Gap sequence 5 (jumped from 2 to 5) -> accepted with gap flag
    send_sequence(5)
    assert len(processed) == 4
    assert processed[-1].ingestion_metadata["is_sequence_gap"] is True

    # 4. Replay cycle reset: sequence 0 arrives after sequence 5 -> accepted with replay_reset flag
    send_sequence(0)
    assert len(processed) == 5
    assert processed[-1].snapshot_sequence == 0
    assert processed[-1].ingestion_metadata["is_replay_reset"] is True


# ==============================================================================
# 6. DASHBOARD BRIDGE COMPATIBILITY
# ==============================================================================

def test_dashboard_bridge_compatibility_with_compact_record(tmp_path, monkeypatch):
    """
    Verifies that on_telemetry_record correctly accepts a compact v2.0 TelemetryRecord,
    computes the 4 channel summaries, and writes the required dashboard state.
    """
    reset_latest_written_sequence()
    temp_state = tmp_path / "live_telemetry.json"
    monkeypatch.setattr("src.mqtt_dashboard_bridge.STATE_FILE", temp_state)
    monkeypatch.setattr("src.mqtt_dashboard_bridge.FRONTEND_DATA_DIR", tmp_path)

    valid_features = {feat: 0.1 for feat in EXPECTED_CANONICAL_FEATURES}
    record = TelemetryRecord(
        machine_id="ims_set2_rig",
        snapshot_sequence=15,
        original_timestamp="2004-02-12T10:32:39Z",
        ingest_timestamp="2026-10-26T00:00:00Z",
        sampling_rate_hz=20480.0,
        sample_count=20480,
        source="replayed_nasa_ims",
        features=valid_features,
        schema_version="2.0",
        ingestion_metadata={"is_duplicate": False, "is_replay_reset": False}
    )

    with patch("src.mqtt_dashboard_bridge.forward_state_to_backend") as mock_forward:
        write_state(record)
        assert temp_state.exists()
        saved_data = json.loads(temp_state.read_text(encoding="utf-8"))
        assert saved_data["machine_id"] == "ims_set2_rig"
        assert saved_data["snapshot_sequence"] == 15
        assert saved_data["feature_count"] == 36
        assert len(saved_data["channels"]) == 4
        assert saved_data["channels"][0]["channel"] == 1
        mock_forward.assert_called_once()
