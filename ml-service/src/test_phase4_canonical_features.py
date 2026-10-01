"""
test_phase4_canonical_features.py - Unit & Integration Test Suite for Phase 4 Canonical Feature Pipeline.

Phase 4: Signal Processing & Feature Engineering Pipeline
Target Dataset: NASA IMS Bearing Dataset (Set 2)
"""

import os
import sys
import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

# Ensure ml-service root is in sys.path
ML_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

from src.data_loading import DEFAULT_RAW_SET2_PATH
from src.ingestion_consumer import TelemetryRecord, TelemetryIngestionConsumer
from src.mqtt_config import MQTTConfig
from src.simulator import TelemetrySimulator
from src.canonical_feature_pipeline import (
    CanonicalFeaturePipeline,
    CanonicalFeatureVector,
    FeatureScalerAdapter,
    CANONICAL_FEATURE_NAMES
)


def test_canonical_feature_names_dimension():
    """Verifies that CANONICAL_FEATURE_NAMES contains exactly 36 deterministic feature names."""
    assert len(CANONICAL_FEATURE_NAMES) == 36
    assert CANONICAL_FEATURE_NAMES[0] == "ch1_mean"
    assert CANONICAL_FEATURE_NAMES[-1] == "ch4_spectral_centroid"
    # Ensure all feature names are unique
    assert len(set(CANONICAL_FEATURE_NAMES)) == 36


def test_offline_online_parity():
    """
    CRITICAL REQUIREMENT:
    Asserts 100% mathematical parity between offline DataFrame feature extraction
    and online streaming TelemetryRecord feature extraction.
    """
    np.random.seed(42)
    raw_arr = np.random.randn(20480, 4)

    pipeline = CanonicalFeaturePipeline(sampling_rate_hz=20480.0)

    # 1. Offline path from DataFrame
    df_snapshot = pd.DataFrame(raw_arr, columns=["Channel_1", "Channel_2", "Channel_3", "Channel_4"])
    vector_offline = pipeline.process_snapshot_dataframe(
        df_snapshot=df_snapshot,
        machine_id="parity_rig",
        snapshot_sequence=7,
        original_timestamp="2004-02-12T10:32:39Z",
        ingest_timestamp="2026-10-01T12:00:00Z"
    )

    # 2. Online path from TelemetryRecord
    channels_dict = {f"ch{i}": raw_arr[:, i-1] for i in range(1, 5)}
    record_online = TelemetryRecord(
        machine_id="parity_rig",
        snapshot_sequence=7,
        original_timestamp="2004-02-12T10:32:39Z",
        ingest_timestamp="2026-10-01T12:00:00Z",
        sampling_rate_hz=20480.0,
        sample_count=20480,
        source="replayed_nasa_ims",
        channels=channels_dict
    )
    vector_online = pipeline.process_telemetry_record(record_online)

    # 3. Assert exact float identity
    assert vector_offline.feature_values.shape == (36,)
    assert vector_online.feature_values.shape == (36,)
    np.testing.assert_allclose(vector_offline.feature_values, vector_online.feature_values, rtol=1e-12, atol=1e-12)
    assert vector_offline.feature_dict == vector_online.feature_dict


def test_feature_vector_finite_and_repeatable():
    """Verifies feature values are finite and extraction is 100% deterministic."""
    pipeline = CanonicalFeaturePipeline()
    np.random.seed(123)
    raw_arr = np.random.randn(20480, 4) * 0.05
    df_snapshot = pd.DataFrame(raw_arr)

    v1 = pipeline.process_snapshot_dataframe(df_snapshot)
    v2 = pipeline.process_snapshot_dataframe(df_snapshot)

    assert v1.validate_finite() is True
    np.testing.assert_array_equal(v1.feature_values, v2.feature_values)


def test_feature_scaler_adapter_fit_transform():
    """Verifies FeatureScalerAdapter fit-on-training-only rules."""
    adapter = FeatureScalerAdapter(scaler_type="robust")
    assert adapter.is_fitted is False

    # Calling transform before fit must raise RuntimeError
    dummy_X = np.random.randn(10, 36)
    with pytest.raises(RuntimeError, match="must be fitted"):
        adapter.transform(dummy_X)

    # Fit on training data (100 snapshots x 36 features)
    X_train = np.random.randn(100, 36)
    adapter.fit(X_train)
    assert adapter.is_fitted is True

    # Transform test matrix
    X_test = np.random.randn(20, 36)
    X_scaled = adapter.transform(X_test)
    assert X_scaled.shape == (20, 36)


def test_no_future_data_leakage():
    """Verifies feature extraction for snapshot t is causally isolated from snapshot t+1."""
    pipeline = CanonicalFeaturePipeline()
    np.random.seed(42)

    arr_t = np.random.randn(20480, 4)
    arr_t_plus_1 = np.random.randn(20480, 4) * 100.0  # Huge future vibration spike

    # Compute features for snapshot t
    v_t_original = pipeline.process_snapshot_dataframe(pd.DataFrame(arr_t), snapshot_sequence=0)

    # Compute features for snapshot t+1
    v_t_plus_1 = pipeline.process_snapshot_dataframe(pd.DataFrame(arr_t_plus_1), snapshot_sequence=1)

    # Re-compute features for snapshot t
    v_t_recomputed = pipeline.process_snapshot_dataframe(pd.DataFrame(arr_t), snapshot_sequence=0)

    # Features at t must be strictly identical regardless of t+1 contents
    np.testing.assert_array_equal(v_t_original.feature_values, v_t_recomputed.feature_values)


def test_real_mqtt_to_canonical_feature_pipeline_integration():
    """
    Real Integration Test:
    NASA IMS snapshot -> TelemetrySimulator -> HiveMQ Cloud over TLS -> Consumer -> TelemetryRecord -> CanonicalFeaturePipeline
    """
    mqtt_cfg = MQTTConfig()
    if not mqtt_cfg.is_configured():
        pytest.skip("MQTTConfig unconfigured. Skipping live HiveMQ Cloud test.")

    pipeline = CanonicalFeaturePipeline()
    extracted_vectors: list[CanonicalFeatureVector] = []

    def on_record(record: TelemetryRecord):
        # Pass TelemetryRecord directly to CanonicalFeaturePipeline
        vector = pipeline.process_telemetry_record(record)
        vector.validate_finite()
        extracted_vectors.append(vector)

    # Initialize Consumer and Simulator
    consumer = TelemetryIngestionConsumer(config=mqtt_cfg, record_callback=on_record)
    if not consumer.start():
        pytest.skip("Failed to connect Consumer to HiveMQ Cloud. Skipping live test.")

    time.sleep(1.5)

    simulator = TelemetrySimulator(config=mqtt_cfg, publish_interval=0.5)
    if not simulator.start_connection():
        consumer.stop()
        pytest.skip("Failed to connect Simulator to HiveMQ Cloud. Skipping live test.")

    # Replay 2 snapshots
    simulator.run_replay(start_idx=0, end_idx=2, loop=False)
    time.sleep(5.0)

    consumer.stop()

    assert len(extracted_vectors) >= 2
    for vec in extracted_vectors:
        assert vec.feature_values.shape == (36,)
        assert vec.validate_finite() is True
        assert vec.machine_id == "ims_set2_rig"
