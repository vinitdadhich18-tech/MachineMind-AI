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


def test_out_of_order_does_not_regress_last_processed_sequence():
    """Verifies that out-of-order sequence (e.g. 38 after 99) does not regress sequence tracker."""
    callback_mock = MagicMock()
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(config=config, record_callback=callback_mock)

    arr = np.zeros((EXPECTED_SAMPLE_COUNT, 4))

    def _make_msg(seq: int):
        p = build_telemetry_payload(arr, "rig_ordering", seq, "2004-02-12T10:32:39Z")
        m = MagicMock()
        m.topic = "machinemind/v1/telemetry/rig_ordering"
        m.payload = serialize_payload(p).encode('utf-8')
        m.qos = 1
        m.retain = False
        return m

    # 1. Send seq=99
    consumer._on_message(None, None, _make_msg(99))
    assert consumer.metrics.last_processed_sequence["rig_ordering"] == 99
    assert consumer._last_sequences["rig_ordering"] == 99
    assert consumer.metrics.out_of_order_detected == 0

    # 2. Send seq=38 (arrives after 99)
    consumer._on_message(None, None, _make_msg(38))
    assert consumer.metrics.out_of_order_detected == 1
    # Last processed sequence must remain 99 (must NOT regress to 38)
    assert consumer.metrics.last_processed_sequence["rig_ordering"] == 99
    assert consumer._last_sequences["rig_ordering"] == 99


def test_consumer_clean_session_configuration():
    """Verifies that TelemetryIngestionConsumer configures clean_session properly."""
    config = MQTTConfig()

    # Default should be clean_session=True
    c1 = TelemetryIngestionConsumer(config=config)
    assert c1.clean_session is True

    # Explicit override to False
    c2 = TelemetryIngestionConsumer(config=config, clean_session=False)
    assert c2.clean_session is False

    # Check client factory gets clean_session parameter
    with patch("src.ingestion_consumer.create_mqtt_client") as mock_factory:
        mock_factory.return_value = MagicMock()
        c1.start()
        mock_factory.assert_called_once_with(config, client_id="machinemind-ingest-ims_set2_rig", clean_session=True)
        c1.stop()


def test_dashboard_bridge_write_state_regression_guard(tmp_path, monkeypatch):
    """Verifies that write_state prevents duplicate or out-of-order sequences from overwriting dashboard state."""
    from src import mqtt_dashboard_bridge
    test_state_file = tmp_path / "live_telemetry.json"
    monkeypatch.setattr(mqtt_dashboard_bridge, "STATE_FILE", test_state_file)
    monkeypatch.setattr(mqtt_dashboard_bridge, "FRONTEND_DATA_DIR", tmp_path)
    mqtt_dashboard_bridge.reset_latest_written_sequence()

    channels = {f"ch{i}": np.zeros(EXPECTED_SAMPLE_COUNT, dtype=np.float64) for i in range(1, 5)}

    def _build_record(seq: int, is_dup: bool = False, is_ooo: bool = False):
        return TelemetryRecord(
            machine_id="test_machine",
            snapshot_sequence=seq,
            original_timestamp="2004-02-12T10:32:39Z",
            ingest_timestamp="2026-10-25T15:00:00Z",
            sampling_rate_hz=20480.0,
            sample_count=EXPECTED_SAMPLE_COUNT,
            source="replayed_nasa_ims",
            channels=channels,
            ingestion_metadata={
                "is_duplicate": is_dup,
                "is_out_of_order": is_ooo,
                "is_sequence_gap": False,
                "processing_latency_ms": 5.0
            }
        )

    # 1. Write seq=99 -> should write successfully
    rec99 = _build_record(99)
    mqtt_dashboard_bridge.write_state(rec99)
    assert test_state_file.exists()
    data = json.loads(test_state_file.read_text(encoding="utf-8"))
    assert data["snapshot_sequence"] == 99

    # 2. Receive older seq=38 (out-of-order) -> state MUST remain 99 (no regression)
    rec38 = _build_record(38, is_ooo=True)
    mqtt_dashboard_bridge.write_state(rec38)
    data = json.loads(test_state_file.read_text(encoding="utf-8"))
    assert data["snapshot_sequence"] == 99

    # 3. Receive duplicate seq=99 -> state MUST remain 99
    rec99_dup = _build_record(99, is_dup=True)
    mqtt_dashboard_bridge.write_state(rec99_dup)
    data = json.loads(test_state_file.read_text(encoding="utf-8"))
    assert data["snapshot_sequence"] == 99

    # 4. Receive forward seq=100 -> state updates to 100
    rec100 = _build_record(100)
    mqtt_dashboard_bridge.write_state(rec100)
    data = json.loads(test_state_file.read_text(encoding="utf-8"))
    assert data["snapshot_sequence"] == 100


def test_replay_cycle_resets_seen_sequences_on_seq_zero():
    """Verifies that sequence 0..99 followed by a new cycle at sequence 0 resets state and succeeds."""
    callback_mock = MagicMock()
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(config=config, record_callback=callback_mock)

    arr = np.zeros((EXPECTED_SAMPLE_COUNT, 4))

    def _make_msg(seq: int):
        p = build_telemetry_payload(arr, "rig_replay", seq, "2004-02-12T10:32:39Z")
        m = MagicMock()
        m.topic = "machinemind/v1/telemetry/rig_replay"
        m.payload = serialize_payload(p).encode('utf-8')
        m.qos = 1
        m.retain = False
        return m

    # Cycle 1: seq 0 to 99
    for s in range(100):
        consumer._on_message(None, None, _make_msg(s))

    assert consumer.metrics.messages_accepted == 100
    assert consumer.metrics.duplicates_detected == 0
    assert consumer._last_sequences["rig_replay"] == 99
    assert callback_mock.call_count == 100

    # Start of Cycle 2: seq 0
    consumer._on_message(None, None, _make_msg(0))

    # Should be accepted as replay reset, NOT duplicate
    assert consumer.metrics.messages_accepted == 101
    assert consumer.metrics.duplicates_detected == 0
    assert consumer._last_sequences["rig_replay"] == 0
    assert callback_mock.call_count == 101

    rec_cycle2_0: TelemetryRecord = callback_mock.call_args[0][0]
    assert rec_cycle2_0.snapshot_sequence == 0
    assert rec_cycle2_0.ingestion_metadata["is_replay_reset"] is True
    assert rec_cycle2_0.ingestion_metadata["is_duplicate"] is False

    # Next sequence in Cycle 2: seq 1
    consumer._on_message(None, None, _make_msg(1))
    assert consumer.metrics.messages_accepted == 102
    assert consumer.metrics.duplicates_detected == 0
    assert consumer._last_sequences["rig_replay"] == 1
    assert callback_mock.call_count == 102


def test_duplicate_within_same_cycle_is_rejected():
    """Verifies that duplicate sequence within the same cycle is rejected without callback."""
    callback_mock = MagicMock()
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(config=config, record_callback=callback_mock)

    arr = np.zeros((EXPECTED_SAMPLE_COUNT, 4))

    def _make_msg(seq: int):
        p = build_telemetry_payload(arr, "rig_dup_cycle", seq, "2004-02-12T10:32:39Z")
        m = MagicMock()
        m.topic = "machinemind/v1/telemetry/rig_dup_cycle"
        m.payload = serialize_payload(p).encode('utf-8')
        m.qos = 1
        m.retain = False
        return m

    consumer._on_message(None, None, _make_msg(0))
    consumer._on_message(None, None, _make_msg(1))
    assert callback_mock.call_count == 2

    # Duplicate seq 1
    consumer._on_message(None, None, _make_msg(1))
    assert consumer.metrics.duplicates_detected == 1
    assert callback_mock.call_count == 2  # callback not invoked


def test_out_of_order_non_zero_does_not_reset_state():
    """Verifies that out-of-order non-zero sequence is rejected without resetting state."""
    callback_mock = MagicMock()
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(config=config, record_callback=callback_mock)

    arr = np.zeros((EXPECTED_SAMPLE_COUNT, 4))

    def _make_msg(seq: int):
        p = build_telemetry_payload(arr, "rig_ooo_no_reset", seq, "2004-02-12T10:32:39Z")
        m = MagicMock()
        m.topic = "machinemind/v1/telemetry/rig_ooo_no_reset"
        m.payload = serialize_payload(p).encode('utf-8')
        m.qos = 1
        m.retain = False
        return m

    consumer._on_message(None, None, _make_msg(10))
    consumer._on_message(None, None, _make_msg(20))
    assert consumer._last_sequences["rig_ooo_no_reset"] == 20

    # Non-zero out of order: seq 15
    consumer._on_message(None, None, _make_msg(15))
    assert consumer.metrics.out_of_order_detected == 1
    assert consumer._last_sequences["rig_ooo_no_reset"] == 20  # not regressed or reset


def test_delayed_seq_zero_without_prior_progress_does_not_reset():
    """Verifies that an immediate second seq 0 (last_seq == 0) is treated as duplicate, not replay reset."""
    callback_mock = MagicMock()
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(config=config, record_callback=callback_mock)

    arr = np.zeros((EXPECTED_SAMPLE_COUNT, 4))

    def _make_msg(seq: int):
        p = build_telemetry_payload(arr, "rig_immediate_zero", seq, "2004-02-12T10:32:39Z")
        m = MagicMock()
        m.topic = "machinemind/v1/telemetry/rig_immediate_zero"
        m.payload = serialize_payload(p).encode('utf-8')
        m.qos = 1
        m.retain = False
        return m

    # First seq 0
    consumer._on_message(None, None, _make_msg(0))
    assert consumer.metrics.messages_accepted == 1
    assert callback_mock.call_count == 1

    # Immediate second seq 0 (retry / delayed duplicate with last_seq=0)
    consumer._on_message(None, None, _make_msg(0))
    assert consumer.metrics.duplicates_detected == 1
    assert callback_mock.call_count == 1  # suppressed duplicate!


def test_dashboard_bridge_allows_verified_replay_reset_and_blocks_unverified(tmp_path, monkeypatch):
    """Verifies that dashboard bridge updates on verified replay reset, but ignores unverified seq 0."""
    from src import mqtt_dashboard_bridge
    test_state_file = tmp_path / "live_telemetry.json"
    monkeypatch.setattr(mqtt_dashboard_bridge, "STATE_FILE", test_state_file)
    monkeypatch.setattr(mqtt_dashboard_bridge, "FRONTEND_DATA_DIR", tmp_path)
    mqtt_dashboard_bridge.reset_latest_written_sequence()

    channels = {f"ch{i}": np.zeros(EXPECTED_SAMPLE_COUNT, dtype=np.float64) for i in range(1, 5)}

    def _build_record(seq: int, is_dup: bool = False, is_ooo: bool = False, is_reset: bool = False):
        return TelemetryRecord(
            machine_id="rig_dashboard_reset",
            snapshot_sequence=seq,
            original_timestamp="2004-02-12T10:32:39Z",
            ingest_timestamp="2026-10-25T15:00:00Z",
            sampling_rate_hz=20480.0,
            sample_count=EXPECTED_SAMPLE_COUNT,
            source="replayed_nasa_ims",
            channels=channels,
            ingestion_metadata={
                "is_duplicate": is_dup,
                "is_out_of_order": is_ooo,
                "is_sequence_gap": False,
                "is_replay_reset": is_reset,
                "processing_latency_ms": 5.0
            }
        )

    # 1. Dashboard at seq 99
    mqtt_dashboard_bridge.write_state(_build_record(99))
    data = json.loads(test_state_file.read_text(encoding="utf-8"))
    assert data["snapshot_sequence"] == 99

    # 2. Unverified seq 0 (is_replay_reset=False) -> MUST NOT regress dashboard
    mqtt_dashboard_bridge.write_state(_build_record(0, is_reset=False))
    data = json.loads(test_state_file.read_text(encoding="utf-8"))
    assert data["snapshot_sequence"] == 99

    # 3. Verified replay reset seq 0 (is_replay_reset=True) -> MUST update dashboard to 0
    mqtt_dashboard_bridge.write_state(_build_record(0, is_reset=True))
    data = json.loads(test_state_file.read_text(encoding="utf-8"))
    assert data["snapshot_sequence"] == 0


def test_fresh_consumer_with_persisted_state_recognizes_replay_reset(tmp_path, monkeypatch):
    """
    Verifies that when a fresh consumer is started with a persisted dashboard file
    holding sequence 99 (e.g. across container restarts), seeding the consumer
    with the persisted sequence allows it to recognize a new replay cycle at sequence 0,
    successfully update the dashboard, and reject stale/duplicate messages.
    """
    from src import mqtt_dashboard_bridge
    test_state_file = tmp_path / "live_telemetry.json"
    monkeypatch.setattr(mqtt_dashboard_bridge, "STATE_FILE", test_state_file)
    monkeypatch.setattr(mqtt_dashboard_bridge, "FRONTEND_DATA_DIR", tmp_path)
    mqtt_dashboard_bridge.reset_latest_written_sequence()

    # Pre-populate persisted state file simulating previous container run at seq 99
    initial_state = {
        "machine_id": "rig_persisted_test",
        "snapshot_sequence": 99,
        "original_timestamp": "2004-02-12T10:32:39Z",
        "ingest_timestamp": "2026-10-25T15:00:00Z",
        "sampling_rate_hz": 20480.0,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "source": "replayed_nasa_ims",
        "feature_count": 36,
        "channels": [],
        "ingestion": {"is_duplicate": False, "is_out_of_order": False, "is_sequence_gap": False}
    }
    test_state_file.write_text(json.dumps(initial_state), encoding="utf-8")

    # Read persisted sequence
    persisted_seq = mqtt_dashboard_bridge.get_latest_written_sequence("rig_persisted_test")
    assert persisted_seq == 99

    # Fresh consumer instantiated with seeded sequence
    config = MQTTConfig()
    consumer = TelemetryIngestionConsumer(
        config=config,
        record_callback=mqtt_dashboard_bridge.write_state,
        initial_sequences={"rig_persisted_test": persisted_seq}
    )

    arr = np.zeros((EXPECTED_SAMPLE_COUNT, 4))

    def _make_msg(seq: int):
        p = build_telemetry_payload(arr, "rig_persisted_test", seq, "2004-02-12T10:32:39Z")
        m = MagicMock()
        m.topic = "machinemind/v1/telemetry/rig_persisted_test"
        m.payload = serialize_payload(p).encode('utf-8')
        m.qos = 1
        m.retain = False
        return m

    # 1. Stale older sequence arriving (e.g. seq 50) -> should be rejected by out-of-order & dashboard guard
    consumer._on_message(None, None, _make_msg(50))
    data = json.loads(test_state_file.read_text(encoding="utf-8"))
    assert data["snapshot_sequence"] == 99  # NOT overwritten!

    # 2. Legitimate new replay cycle starting at sequence 0
    consumer._on_message(None, None, _make_msg(0))
    data = json.loads(test_state_file.read_text(encoding="utf-8"))
    assert data["snapshot_sequence"] == 0  # Successfully updated to 0!
    assert consumer._last_sequences["rig_persisted_test"] == 0

    # 3. Next message in cycle (seq 1) -> dashboard advances to 1
    consumer._on_message(None, None, _make_msg(1))
    data = json.loads(test_state_file.read_text(encoding="utf-8"))
    assert data["snapshot_sequence"] == 1
