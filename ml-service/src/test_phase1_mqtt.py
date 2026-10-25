"""
test_phase1_mqtt.py - Unit & Integration Test Suite for Phase 1 MQTT Architecture.

Phase 1: Telemetry Simulator & MQTT Infrastructure
Target Dataset: NASA IMS Bearing Dataset (Set 2)
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure ml-service directory is in sys.path
ML_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

from src.mqtt_config import MQTTConfig, create_mqtt_client
from src.telemetry_schema import (
    build_telemetry_payload,
    validate_telemetry_payload,
    assert_no_label_leakage,
    serialize_payload,
    deserialize_payload,
    EXPECTED_SAMPLE_COUNT
)
from src.simulator import TelemetrySimulator
from src.data_loading import DEFAULT_RAW_SET2_PATH


def test_mqtt_config_loading_and_masking():
    """Verifies MQTTConfig loads defaults and masks credentials safely."""
    config = MQTTConfig()
    summary = config.get_masked_summary()

    assert "broker_host" in summary
    assert "broker_port" in summary
    assert "username" in summary
    assert summary["qos"] == 1
    # Ensure password is never in summary dictionary
    assert "password" not in summary
    assert "your_hivemq_password" not in str(summary)


def test_telemetry_schema_building_and_validation():
    """Tests building and validating a telemetry payload with synthetic 20480x4 signal."""
    np.random.seed(42)
    synthetic_arr = np.random.randn(EXPECTED_SAMPLE_COUNT, 4)

    payload = build_telemetry_payload(
        raw_snapshot=synthetic_arr,
        machine_id="test_rig_01",
        snapshot_sequence=42,
        original_timestamp="2004-02-12T10:32:39Z",
        sampling_rate_hz=20480.0
    )

    assert payload["schema_version"] == "1.0"
    assert payload["source"] == "replayed_nasa_ims"
    assert payload["machine_id"] == "test_rig_01"
    assert payload["snapshot_sequence"] == 42
    assert payload["sample_count"] == EXPECTED_SAMPLE_COUNT
    assert len(payload["channels"]["ch1"]) == EXPECTED_SAMPLE_COUNT
    assert len(payload["channels"]["ch4"]) == EXPECTED_SAMPLE_COUNT

    # Validate schema
    assert validate_telemetry_payload(payload) is True


def test_label_leakage_rejection():
    """Asserts that adding forbidden label/RUL keys raises ValueError."""
    np.random.seed(42)
    synthetic_arr = np.random.randn(EXPECTED_SAMPLE_COUNT, 4)

    payload = build_telemetry_payload(
        raw_snapshot=synthetic_arr,
        machine_id="test_rig_01",
        snapshot_sequence=1,
        original_timestamp="2004-02-12T10:32:39Z"
    )

    # Attempt injecting RUL key
    leakage_payload = payload.copy()
    leakage_payload["rul"] = 500

    with pytest.raises(ValueError, match="LABEL LEAKAGE DETECTED"):
        assert_no_label_leakage(leakage_payload)

    with pytest.raises(ValueError, match="LABEL LEAKAGE DETECTED"):
        validate_telemetry_payload(leakage_payload)

    # Attempt injecting failure label
    leakage_payload2 = payload.copy()
    leakage_payload2["failure_status"] = "FAILED"

    with pytest.raises(ValueError, match="LABEL LEAKAGE DETECTED"):
        assert_no_label_leakage(leakage_payload2)


def test_telemetry_payload_serialization_roundtrip():
    """Tests JSON serialization and deserialization roundtrip."""
    synthetic_arr = np.zeros((EXPECTED_SAMPLE_COUNT, 4))
    payload = build_telemetry_payload(
        raw_snapshot=synthetic_arr,
        machine_id="roundtrip_rig",
        snapshot_sequence=10,
        original_timestamp="2004-02-12T10:32:39Z"
    )

    json_str = serialize_payload(payload)
    assert isinstance(json_str, str)
    assert "roundtrip_rig" in json_str

    deserialized = deserialize_payload(json_str)
    assert deserialized["machine_id"] == "roundtrip_rig"
    assert deserialized["snapshot_sequence"] == 10
    assert len(deserialized["channels"]["ch1"]) == EXPECTED_SAMPLE_COUNT


def test_simulator_payload_generation_from_real_data():
    """Tests that TelemetrySimulator loads real IMS Set 2 files and creates valid payloads."""
    if not DEFAULT_RAW_SET2_PATH.exists():
        pytest.skip(f"Raw IMS data directory not found at {DEFAULT_RAW_SET2_PATH}")

    config = MQTTConfig()
    sim = TelemetrySimulator(config=config, raw_dir=DEFAULT_RAW_SET2_PATH, publish_interval=0.1)

    assert len(sim.snapshot_files) == 984

    # Test loading and building payload for first snapshot
    first_file = sim.snapshot_files[0]
    df_first = pd.read_csv(first_file, sep=r"\s+", header=None)

    payload = build_telemetry_payload(
        raw_snapshot=df_first,
        machine_id="ims_set2_rig",
        snapshot_sequence=0,
        original_timestamp="2004-02-12T10:32:39Z"
    )

    assert validate_telemetry_payload(payload) is True
    assert payload["sample_count"] == 20480
