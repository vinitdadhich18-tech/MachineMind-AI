"""
test_phase3_influx.py - Unit Test Suite for Phase 3 InfluxDB Integration.

Phase 3: InfluxDB Integration & Time-Series Schema
Target Dataset: NASA IMS Bearing Dataset (Set 2)

Note: All unit tests in this suite use mocks/stubs.
They DO NOT depend on active external InfluxDB connections.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Ensure ml-service root is in sys.path
ML_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

from src.ingestion_consumer import TelemetryRecord
from src.influx_writer import InfluxConfig, InfluxDBWriter


def test_influx_config_masking():
    """Verifies InfluxConfig masks auth token safely."""
    config = InfluxConfig()
    summary = config.get_masked_summary()

    assert "url" in summary
    assert "org" in summary
    assert "bucket" in summary
    assert "token" in summary
    assert "your_influxdb_token" not in str(summary["token"])


def test_duplicate_record_idempotency_skip():
    """Verifies duplicate TelemetryRecords skip InfluxDB writing."""
    write_api_mock = MagicMock()
    writer = InfluxDBWriter()
    writer.write_api = write_api_mock

    channels = {f"ch{i}": np.zeros(20480, dtype=np.float64) for i in range(1, 5)}
    duplicate_record = TelemetryRecord(
        machine_id="dup_rig",
        snapshot_sequence=5,
        original_timestamp="2004-02-12T10:32:39Z",
        ingest_timestamp="2026-10-01T11:00:00Z",
        sampling_rate_hz=20480.0,
        sample_count=20480,
        source="replayed_nasa_ims",
        channels=channels,
        ingestion_metadata={"is_duplicate": True}
    )

    success = writer.write_telemetry_record(duplicate_record)
    assert success is True
    # Write API must NOT be called for duplicate record
    assert write_api_mock.write.call_count == 0


def test_point_construction_schema_and_tags():
    """Verifies that point fields, tags, measurements, and timestamp semantics match schema."""
    write_api_mock = MagicMock()
    writer = InfluxDBWriter()
    writer.write_api = write_api_mock

    np.random.seed(42)
    channels = {f"ch{i}": np.random.randn(20480) for i in range(1, 5)}
    record = TelemetryRecord(
        machine_id="schema_rig",
        snapshot_sequence=12,
        original_timestamp="2004-02-12T10:32:39Z",
        ingest_timestamp="2026-10-01T11:30:00Z",
        sampling_rate_hz=20480.0,
        sample_count=20480,
        source="replayed_nasa_ims",
        channels=channels,
        ingestion_metadata={"is_duplicate": False, "processing_latency_ms": 11.5}
    )

    success = writer.write_telemetry_record(record)
    assert success is True
    assert write_api_mock.write.call_count == 1

    kwargs = write_api_mock.write.call_args[1]
    points = kwargs["record"]
    assert len(points) == 5  # 4 channels of vibration_features + 1 pipeline_health point

    # Check vibration_features point for Channel 1
    ch1_point = points[0]
    line_protocol = ch1_point.to_line_protocol()

    assert "vibration_features" in line_protocol
    assert "machine_id=schema_rig" in line_protocol
    assert "channel=ch1" in line_protocol
    assert "source=replayed_nasa_ims" in line_protocol
    assert "snapshot_sequence=12i" in line_protocol
    assert 'original_timestamp="2004-02-12T10:32:39Z"' in line_protocol

    # Confirm raw array is NOT stored in line protocol (only summary metrics stored)
    assert "ch1_raw" not in line_protocol
    assert "array" not in line_protocol

    # Check pipeline_health point
    health_point = points[4]
    health_lp = health_point.to_line_protocol()
    assert "pipeline_health" in health_lp
    assert "service_name=mqtt_ingestion_consumer" in health_lp
    assert "processing_latency_ms=11.5" in health_lp


def test_failed_write_exception_handling():
    """Tests that InfluxDB API write exception is handled gracefully without crashing."""
    write_api_mock = MagicMock()
    write_api_mock.write.side_effect = Exception("Connection Refused")

    writer = InfluxDBWriter()
    writer.write_api = write_api_mock

    channels = {f"ch{i}": np.zeros(20480, dtype=np.float64) for i in range(1, 5)}
    record = TelemetryRecord(
        machine_id="fail_rig",
        snapshot_sequence=1,
        original_timestamp="2004-02-12T10:32:39Z",
        ingest_timestamp="2026-10-01T11:00:00Z",
        sampling_rate_hz=20480.0,
        sample_count=20480,
        source="replayed_nasa_ims",
        channels=channels
    )

    success = writer.write_telemetry_record(record)
    assert success is False
