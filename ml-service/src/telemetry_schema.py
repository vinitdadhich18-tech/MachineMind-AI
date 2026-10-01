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
DEFAULT_SOURCE = "replayed_nasa_ims"
EXPECTED_SAMPLE_COUNT = 20480
EXPECTED_CHANNELS = ["ch1", "ch2", "ch3", "ch4"]

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
        Structured JSON-serializable telemetry dictionary.
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


def validate_telemetry_payload(payload: Dict[str, Any]) -> bool:
    """
    Validates payload structure, required fields, shape, and label leakage rules.
    Returns True if valid. Raises ValueError or KeyError if malformed.
    """
    if not isinstance(payload, dict):
        raise TypeError("Payload must be a dictionary.")

    required_top_keys = {
        "schema_version", "source", "machine_id", "snapshot_sequence",
        "original_timestamp", "ingest_timestamp", "sampling_rate_hz",
        "sample_count", "channels"
    }

    missing_keys = required_top_keys - set(payload.keys())
    if missing_keys:
        raise KeyError(f"Telemetry payload missing required keys: {missing_keys}")

    # Check channels dictionary
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
