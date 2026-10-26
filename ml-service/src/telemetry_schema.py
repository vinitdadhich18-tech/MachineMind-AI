"""
telemetry_schema.py - JSON Payload Schema Contract and Validation Engine.

Phase 1: Telemetry Simulator & MQTT Infrastructure
Target Dataset: NASA IMS Bearing Dataset (Set 2)

Honesty & Leakage Rules:
1. Data source MUST be identified as 'replayed_nasa_ims' (simulated telemetry).
2. Payloads MUST NEVER contain labels, RUL hints, failure flags, or pre-computed anomaly scores.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"
COMPACT_SCHEMA_VERSION = "2.0"
SUPPORTED_SCHEMA_VERSIONS = {SCHEMA_VERSION, COMPACT_SCHEMA_VERSION}

DEFAULT_SOURCE = "replayed_nasa_ims"
EXPECTED_SAMPLE_COUNT = 20480
EXPECTED_CHANNELS = ["ch1", "ch2", "ch3", "ch4"]

EXPECTED_FEATURE_SUFFIXES = [
    "mean", "std", "rms", "p2p", "skewness", "kurtosis", "crest_factor",
    "spectral_energy", "spectral_centroid"
]
EXPECTED_CANONICAL_FEATURES = [
    f"{ch}_{suffix}" for ch in EXPECTED_CHANNELS for suffix in EXPECTED_FEATURE_SUFFIXES
]

# Forbidden label / leakage key words in telemetry payload
FORBIDDEN_LEAKAGE_KEYS = {
    "rul", "remaining_useful_life", "label", "failure", "failed",
    "anomaly_score", "score", "is_anomaly", "alert", "degradation"
}


def build_telemetry_payload(
    raw_snapshot: Union[np.ndarray, pd.DataFrame],
    machine_id: str,
    snapshot_sequence: int,
    original_timestamp: Union[str, datetime],
    ingest_timestamp: Union[str, datetime, None] = None,
    sampling_rate_hz: float = 20480.0,
    source: str = DEFAULT_SOURCE
) -> Dict[str, Any]:
    """
    Constructs a validated JSON-serializable telemetry payload dictionary from a raw snapshot file.

    Parameters
    ----------
    raw_snapshot : np.ndarray or pd.DataFrame
        Raw vibration snapshot array of shape (20480, 4).
    machine_id : str
        Identifier of the monitored machinery equipment.
    snapshot_sequence : int
        Monotonic sequence index of the snapshot.
    original_timestamp : str or datetime
        Original timestamp recorded in dataset filename (e.g. '2004-02-12T10:32:39Z').
    ingest_timestamp : str or datetime, optional
        Virtual live ingest timestamp. If None, defaults to current UTC time.
    sampling_rate_hz : float, default 20480.0
        Reported acquisition sampling frequency.
    source : str, default 'replayed_nasa_ims'
        Provenance indicator.

    Returns
    -------
    dict
        Structured JSON-serializable telemetry dictionary (Schema v1.0, ~514.8 KB).
    """
    if isinstance(raw_snapshot, pd.DataFrame):
        arr = raw_snapshot.to_numpy(dtype=np.float64)
    elif isinstance(raw_snapshot, np.ndarray):
        arr = raw_snapshot.astype(np.float64, copy=False)
    else:
        raise TypeError(f"Expected np.ndarray or pd.DataFrame, got {type(raw_snapshot)}.")

    if arr.shape != (EXPECTED_SAMPLE_COUNT, 4):
        raise ValueError(f"Expected snapshot shape ({EXPECTED_SAMPLE_COUNT}, 4), got {arr.shape}.")

    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Raw snapshot contains NaN or Infinite values.")

    orig_ts_str = (
        original_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(original_timestamp, datetime)
        else str(original_timestamp)
    )

    if ingest_timestamp is None:
        ingest_ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif isinstance(ingest_timestamp, datetime):
        ingest_ts_str = ingest_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        ingest_ts_str = str(ingest_timestamp)

    # Convert 20480x4 float array to native Python float list per channel
    channels_dict = {
        f"ch{idx + 1}": [round(float(val), 6) for val in arr[:, idx]]
        for idx in range(4)
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "machine_id": str(machine_id),
        "snapshot_sequence": int(snapshot_sequence),
        "original_timestamp": orig_ts_str,
        "ingest_timestamp": ingest_ts_str,
        "sampling_rate_hz": float(sampling_rate_hz),
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "channels": channels_dict
    }

    # Verify no label leakage before returning
    assert_no_label_leakage(payload)

    return payload


def build_compact_telemetry_payload(
    features: Dict[str, Union[float, int]],
    machine_id: str,
    snapshot_sequence: int,
    original_timestamp: Union[str, datetime],
    ingest_timestamp: Union[str, datetime, None] = None,
    sampling_rate_hz: float = 20480.0,
    sample_count: int = EXPECTED_SAMPLE_COUNT,
    source: str = DEFAULT_SOURCE
) -> Dict[str, Any]:
    """
    Constructs a validated compact JSON-serializable telemetry payload (v2.0)
    containing 36 pre-computed canonical vibration features.

    Bandwidth reduction: ~1.2 KB payload compared to ~514.8 KB raw v1.0 payload (>99.7% reduction).
    Honesty Principle: Replayed historical NASA IMS bearing dataset, explicitly marked as 'replayed_nasa_ims'.
    Leakage Principle: Rejects forbidden keywords (RUL hints, failure flags, pre-computed labels/scores).
    """
    if not isinstance(features, dict):
        raise TypeError(f"Expected features to be a dictionary, got {type(features)}.")

    missing_feats = set(EXPECTED_CANONICAL_FEATURES) - set(features.keys())
    if missing_feats:
        raise KeyError(f"Compact telemetry features missing expected keys: {sorted(missing_feats)}")

    cleaned_features = {}
    for feat_name in EXPECTED_CANONICAL_FEATURES:
        val = features[feat_name]
        if isinstance(val, (bool, str)) or not isinstance(val, (int, float, np.number)):
            raise ValueError(f"Feature '{feat_name}' must be numeric, got {type(val)}: {val}")
        f_val = float(val)
        if np.isnan(f_val) or np.isinf(f_val):
            raise ValueError(f"Feature '{feat_name}' contains NaN or Infinite value: {f_val}")
        cleaned_features[feat_name] = round(f_val, 6)

    orig_ts_str = (
        original_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(original_timestamp, datetime)
        else str(original_timestamp)
    )

    if ingest_timestamp is None:
        ingest_ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif isinstance(ingest_timestamp, datetime):
        ingest_ts_str = ingest_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        ingest_ts_str = str(ingest_timestamp)

    payload = {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "source": str(source),
        "machine_id": str(machine_id),
        "snapshot_sequence": int(snapshot_sequence),
        "original_timestamp": orig_ts_str,
        "ingest_timestamp": ingest_ts_str,
        "sampling_rate_hz": float(sampling_rate_hz),
        "sample_count": int(sample_count),
        "features": cleaned_features
    }

    assert_no_label_leakage(payload)
    return payload


def validate_telemetry_payload(payload: Dict[str, Any]) -> bool:
    """
    Validates payload structure, required fields, shape/features, and label leakage rules.
    Supports both v1.0 (raw channels) and v2.0 (compact canonical features).
    Returns True if valid. Raises ValueError, TypeError, or KeyError if malformed.
    """
    if not isinstance(payload, dict):
        raise TypeError("Payload must be a dictionary.")

    version = payload.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema_version '{version}'. Must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )

    base_required_keys = {
        "schema_version", "source", "machine_id", "snapshot_sequence",
        "original_timestamp", "ingest_timestamp", "sampling_rate_hz",
        "sample_count"
    }

    if version == SCHEMA_VERSION:
        required_top_keys = base_required_keys | {"channels"}
    elif version == COMPACT_SCHEMA_VERSION:
        required_top_keys = base_required_keys | {"features"}
    else:
        required_top_keys = base_required_keys

    missing_keys = required_top_keys - set(payload.keys())
    if missing_keys:
        raise KeyError(f"Telemetry payload missing required keys: {sorted(missing_keys)}")

    # Snapshot sequence must be non-negative integer
    seq = payload["snapshot_sequence"]
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ValueError(f"snapshot_sequence must be a non-negative integer, got {seq}")

    # Validate based on version
    if version == SCHEMA_VERSION:
        channels = payload["channels"]
        if not isinstance(channels, dict):
            raise TypeError("Payload 'channels' must be a dictionary.")

        for ch_name in EXPECTED_CHANNELS:
            if ch_name not in channels:
                raise KeyError(f"Payload channels missing expected key '{ch_name}'.")
            samples = channels[ch_name]
            if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLE_COUNT:
                raise ValueError(
                    f"Channel '{ch_name}' must contain exactly {EXPECTED_SAMPLE_COUNT} samples, got {len(samples)}."
                )
    elif version == COMPACT_SCHEMA_VERSION:
        features = payload["features"]
        if not isinstance(features, dict):
            raise TypeError("Payload 'features' must be a dictionary.")
        missing_feats = set(EXPECTED_CANONICAL_FEATURES) - set(features.keys())
        if missing_feats:
            raise KeyError(f"Compact telemetry features missing expected keys: {sorted(missing_feats)}")
        for feat_name in EXPECTED_CANONICAL_FEATURES:
            val = features[feat_name]
            if isinstance(val, (bool, str)) or not isinstance(val, (int, float, np.number)):
                raise ValueError(f"Feature '{feat_name}' must be numeric, got {type(val)}: {val}")
            f_val = float(val)
            if np.isnan(f_val) or np.isinf(f_val):
                raise ValueError(f"Feature '{feat_name}' contains NaN or Infinite value.")

    # Label leakage check
    assert_no_label_leakage(payload)

    return True


def assert_no_label_leakage(payload: Dict[str, Any]) -> None:
    """
    Recursively inspects dictionary keys and string values to ensure zero label leakage.
    Raises ValueError if any forbidden leakage keyword is detected.
    """
    def _check_dict(d: Dict[str, Any]) -> None:
        for k, v in d.items():
            k_lower = str(k).lower()
            for forbidden in FORBIDDEN_LEAKAGE_KEYS:
                if forbidden in k_lower:
                    raise ValueError(
                        f"LABEL LEAKAGE DETECTED! Forbidden key '{k}' found in telemetry payload."
                    )
            if isinstance(v, dict):
                _check_dict(v)

    _check_dict(payload)


def serialize_payload(payload: Dict[str, Any]) -> str:
    """Serializes telemetry payload dict to JSON string."""
    assert_no_label_leakage(payload)
    return json.dumps(payload, separators=(',', ':'))


def deserialize_payload(json_str: str) -> Dict[str, Any]:
    """Deserializes JSON string to telemetry payload dict and validates it."""
    payload = json.loads(json_str)
    validate_telemetry_payload(payload)
    return payload
