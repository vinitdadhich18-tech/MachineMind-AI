"""
test_phase2_ingestion.py - Focused Unit Tests for Phase 2 Telemetry Ingestion Consumer.

Phase 2: MQTT Consumer / Ingestion Service
Target Dataset: NASA IMS Bearing Dataset (Set 2)

Note: All unit tests in this suite operate statelessly or using mocks.
They DO NOT depend on active HiveMQ Cloud network credentials.
"""

import os
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

from src.mqtt_config import MQTTConfig
from src.telemetry_schema import (
    build_telemetry_payload,
    serialize_payload,
    EXPECTED_SAMPLE_COUNT
)
from src.ingestion_consumer import (
    TelemetryRecord,
    ConsumerMetrics,
    TelemetryIngestionConsumer
)


def test_telemetry_record_structure():
    """Verifies TelemetryRecord dataclass initialization and summary output."""
    channels = {f"ch{i}": np.zeros(20480, dtype=np.float64) for i in range(1, 5)}
    record = TelemetryRecord(
        machine_id="test_rig",
        snapshot_sequence=5,
        original_timestamp="2004-02-12T10:32:39Z",
        ingest_timestamp="2026-10-01T11:00:00Z",
        sampling_rate_hz=20480.0,
        sample_count=20480,
        source="replayed_nasa_ims",
        channels=channels,
        ingestion_metadata={"is_duplicate": False, "is_out_of_order": False}
    )

    assert record.machine_id == "test_rig"
    assert record.snapshot_sequence == 5
    assert record.channels["ch1"].shape == (20480,)
    
    summary = record.to_summary_dict()
    assert summary["machine_id"] == "test_rig"
    assert summary["channels_present"] == ["ch1", "ch2", "ch3", "ch4"]


def test_consumer_metrics_counter():
    """Verifies operational metrics aggregation."""
    metrics = ConsumerMetrics()
    metrics.messages_received += 1
    metrics.messages_accepted += 1
    metrics.duplicates_detected += 1
    metrics.last_processed_sequence["rig1"] = 42

    d = metrics.to_dict()
    assert d["messages_received"] == 1
    assert d["messages_accepted"] == 1
    assert d["duplicates_detected"] == 1
    assert d["last_processed_sequence"]["rig1"] == 42


def test_valid_telemetry_message_acceptance():
    """Tests that a valid telemetry MQTT message is accepted and passed to record_callback."""
    callback_mock = MagicMock()
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(config=config, record_callback=callback_mock)

    arr = np.random.randn(EXPECTED_SAMPLE_COUNT, 4)
    payload = build_telemetry_payload(
        raw_snapshot=arr,
        machine_id="rig_acceptance",
        snapshot_sequence=0,
        original_timestamp="2004-02-12T10:32:39Z"
    )

    json_str = serialize_payload(payload)

    mqtt_msg = MagicMock()
    mqtt_msg.topic = "machinemind/v1/telemetry/rig_acceptance"
    mqtt_msg.payload = json_str.encode('utf-8')
    mqtt_msg.qos = 1
    mqtt_msg.retain = False

    # Simulate message arrival
    consumer._on_message(None, None, mqtt_msg)

    assert consumer.metrics.messages_received == 1
    assert consumer.metrics.messages_accepted == 1
    assert consumer.metrics.messages_rejected == 0
    assert callback_mock.call_count == 1

    rec: TelemetryRecord = callback_mock.call_args[0][0]
    assert rec.machine_id == "rig_acceptance"
    assert rec.snapshot_sequence == 0
    assert rec.original_timestamp == "2004-02-12T10:32:39Z"
    assert rec.channels["ch1"].shape == (20480,)


def test_malformed_json_rejection():
    """Tests that malformed non-JSON messages are safely rejected without crashing."""
    callback_mock = MagicMock()
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(config=config, record_callback=callback_mock)

    mqtt_msg = MagicMock()
    mqtt_msg.topic = "machinemind/v1/telemetry/rig_malformed"
    mqtt_msg.payload = b"NOT_A_VALID_JSON_STRING"
    mqtt_msg.qos = 1
    mqtt_msg.retain = False

    consumer._on_message(None, None, mqtt_msg)

    assert consumer.metrics.messages_received == 1
    assert consumer.metrics.messages_accepted == 0
    assert consumer.metrics.messages_rejected == 1
    assert callback_mock.call_count == 0


def test_invalid_schema_and_sample_count_rejection():
    """Tests that payload with wrong array shape or missing required keys is rejected."""
    callback_mock = MagicMock()
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(config=config, record_callback=callback_mock)

    # Invalid channel sample count (100 samples instead of 20480)
    bad_payload = {
        "schema_version": "1.0",
        "source": "replayed_nasa_ims",
        "machine_id": "rig_bad_schema",
        "snapshot_sequence": 1,
        "original_timestamp": "2004-02-12T10:32:39Z",
        "ingest_timestamp": "2026-10-01T11:00:00Z",
        "sampling_rate_hz": 20480.0,
        "sample_count": 20480,
        "channels": {
            "ch1": [0.0] * 100,  # Invalid size!
            "ch2": [0.0] * 20480,
            "ch3": [0.0] * 20480,
            "ch4": [0.0] * 20480,
        }
    }

    mqtt_msg = MagicMock()
    mqtt_msg.topic = "machinemind/v1/telemetry/rig_bad_schema"
    mqtt_msg.payload = json.dumps(bad_payload).encode('utf-8')
    mqtt_msg.qos = 1
    mqtt_msg.retain = False

    consumer._on_message(None, None, mqtt_msg)

    assert consumer.metrics.messages_accepted == 0
    assert consumer.metrics.messages_rejected == 1
    assert callback_mock.call_count == 0


def test_label_leakage_guard_rejection():
    """Tests that payloads containing label/RUL keywords are rejected by leakage guard."""
    callback_mock = MagicMock()
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(config=config, record_callback=callback_mock)

    arr = np.random.randn(EXPECTED_SAMPLE_COUNT, 4)
    payload = build_telemetry_payload(
        raw_snapshot=arr,
        machine_id="rig_leakage",
        snapshot_sequence=2,
        original_timestamp="2004-02-12T10:32:39Z"
    )

    # Inject forbidden label key
    payload["predicted_rul"] = 150

    mqtt_msg = MagicMock()
    mqtt_msg.topic = "machinemind/v1/telemetry/rig_leakage"
    mqtt_msg.payload = json.dumps(payload).encode('utf-8')
    mqtt_msg.qos = 1
    mqtt_msg.retain = False

    consumer._on_message(None, None, mqtt_msg)

    assert consumer.metrics.messages_accepted == 0
    assert consumer.metrics.messages_rejected == 1
    assert consumer.metrics.leakage_violations_detected == 1
    assert callback_mock.call_count == 0


def test_duplicate_message_idempotency_handling():
    """Tests QoS 1 retry duplicate payload detection (same sequence ID sent twice)."""
    callback_mock = MagicMock()
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(config=config, record_callback=callback_mock)

    arr = np.random.randn(EXPECTED_SAMPLE_COUNT, 4)
    payload = build_telemetry_payload(
        raw_snapshot=arr,
        machine_id="rig_dup",
        snapshot_sequence=10,
        original_timestamp="2004-02-12T10:32:39Z"
    )
    json_str = serialize_payload(payload)

    mqtt_msg = MagicMock()
    mqtt_msg.topic = "machinemind/v1/telemetry/rig_dup"
    mqtt_msg.payload = json_str.encode('utf-8')
    mqtt_msg.qos = 1
    mqtt_msg.retain = False

    # Send first time
    consumer._on_message(None, None, mqtt_msg)
    assert consumer.metrics.messages_accepted == 1
    assert consumer.metrics.duplicates_detected == 0
    assert callback_mock.call_count == 1

    # Send second time (duplicate sequence 10)
    consumer._on_message(None, None, mqtt_msg)
    assert consumer.metrics.messages_accepted == 2  # Total accepted messages logged
    assert consumer.metrics.duplicates_detected == 1
    # Callback should NOT be called second time for duplicate
    assert callback_mock.call_count == 1


def test_sequence_gap_and_out_of_order_detection():
    """Tests sequence ordering tracking (detecting gaps and out-of-order sequence IDs)."""
    callback_mock = MagicMock()
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(config=config, record_callback=callback_mock)

    arr = np.zeros((EXPECTED_SAMPLE_COUNT, 4))

    def _make_msg(seq: int):
        p = build_telemetry_payload(arr, "rig_seq", seq, "2004-02-12T10:32:39Z")
        m = MagicMock()
        m.topic = "machinemind/v1/telemetry/rig_seq"
        m.payload = serialize_payload(p).encode('utf-8')
        m.qos = 1
        m.retain = False
        return m

    # 1. Send seq=0
    consumer._on_message(None, None, _make_msg(0))
    assert consumer.metrics.out_of_order_detected == 0

    # 2. Send seq=5 (Gap from 0 -> 5)
    consumer._on_message(None, None, _make_msg(5))
    assert consumer.metrics.out_of_order_detected == 0  # Gap is logged as warning, not out_of_order

    # 3. Send seq=2 (Out of order: 2 arrived after 5!)
    consumer._on_message(None, None, _make_msg(2))
    assert consumer.metrics.out_of_order_detected == 1


def test_timestamp_preservation():
    """Verifies that original_timestamp is preserved exactly while ingest_timestamp is added."""
    callback_mock = MagicMock()
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(config=config, record_callback=callback_mock)

    arr = np.zeros((EXPECTED_SAMPLE_COUNT, 4))
    payload = build_telemetry_payload(
        raw_snapshot=arr,
        machine_id="rig_ts",
        snapshot_sequence=0,
        original_timestamp="2004-02-12T10:32:39Z"
    )

    mqtt_msg = MagicMock()
    mqtt_msg.topic = "machinemind/v1/telemetry/rig_ts"
    mqtt_msg.payload = serialize_payload(payload).encode('utf-8')
    mqtt_msg.qos = 1
    mqtt_msg.retain = False

    consumer._on_message(None, None, mqtt_msg)

    assert callback_mock.call_count == 1
    rec: TelemetryRecord = callback_mock.call_args[0][0]
    
    # Original timestamp preserved strictly
    assert rec.original_timestamp == "2004-02-12T10:32:39Z"
    # Ingest timestamp is present and contains ISO string format
    assert isinstance(rec.ingest_timestamp, str)
    assert len(rec.ingest_timestamp) > 10
