"""
telemetry_loader.py - Remote telemetry fetching, response validation, staleness checking, and local-file fallback.
"""

import json
import time
import math
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from frontend.utils.config import (
    BACKEND_URL,
    TELEMETRY_TIMEOUT,
    TELEMETRY_STALE_SECONDS
)
from frontend.services.api_client import get_latest_telemetry, ApiError

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "live_telemetry.json"

REQUIRED_METRICS = (
    "mean", "std", "rms", "p2p", "skewness",
    "kurtosis", "crest_factor", "spectral_energy", "spectral_centroid"
)


def validate_telemetry_payload(data: Any) -> Optional[Dict[str, Any]]:
    """
    Validates that a telemetry state object conforms to the expected 36-feature schema.
    Returns cleaned dict if valid, or None if malformed.
    """
    if not isinstance(data, dict):
        return None

    # machine_id
    machine_id = data.get("machine_id")
    if not machine_id or not isinstance(machine_id, str):
        return None

    # snapshot_sequence
    seq = data.get("snapshot_sequence")
    if seq is None or isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        return None

    # channels
    channels = data.get("channels")
    if not isinstance(channels, list) or len(channels) != 4:
        return None

    for ch in channels:
        if not isinstance(ch, dict):
            return None
        ch_num = ch.get("channel")
        if isinstance(ch_num, bool) or not isinstance(ch_num, int) or ch_num not in (1, 2, 3, 4):
            return None
        for m in REQUIRED_METRICS:
            val = ch.get(m)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                return None

    return data


def load_local_telemetry_file(file_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Reads and validates local fallback telemetry JSON file."""
    target_path = file_path if file_path is not None else DEFAULT_LOCAL_DATA_FILE
    if not target_path.exists():
        return None

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return validate_telemetry_payload(raw)
    except Exception as err:
        logger.warning(f"Failed to read local telemetry fallback file {target_path}: {err}")
        return None


def calculate_telemetry_age(
    telemetry: Dict[str, Any],
    stale_threshold: float = TELEMETRY_STALE_SECONDS,
    current_time: Optional[float] = None
) -> Tuple[bool, float, str, str]:
    """
    Calculates age and staleness from dashboard_updated_at, updated_at, or ingest_timestamp.
    Returns:
        (is_stale, age_seconds, age_human_str, timestamp_display_str)
    """
    now = current_time if current_time is not None else time.time()
    ts_epoch: Optional[float] = None
    ts_display = "N/A"

    # 1. Check dashboard_updated_at epoch float
    raw_epoch = telemetry.get("dashboard_updated_at")
    if isinstance(raw_epoch, (int, float)) and math.isfinite(raw_epoch) and raw_epoch > 0:
        ts_epoch = float(raw_epoch)
        try:
            ts_display = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            ts_display = str(raw_epoch)

    # 2. Fallback to ISO updated_at or ingest_timestamp string
    if ts_epoch is None:
        for key in ("updated_at", "ingest_timestamp", "original_timestamp"):
            val = telemetry.get(key)
            if isinstance(val, str) and val.strip():
                ts_display = val.strip()
                try:
                    # Clean trailing Z for fromisoformat
                    clean_iso = val.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(clean_iso)
                    ts_epoch = dt.timestamp()
                    break
                except Exception:
                    continue

    if ts_epoch is None:
        return False, 0.0, "unknown", ts_display

    age_seconds = max(0.0, now - ts_epoch)
    is_stale = age_seconds > stale_threshold

    if age_seconds < 60:
        age_str = f"{age_seconds:.1f}s ago"
    elif age_seconds < 3600:
        age_str = f"{age_seconds / 60:.1f}m ago"
    else:
        age_str = f"{age_seconds / 3600:.1f}h ago"

    return is_stale, age_seconds, age_str, ts_display


def fetch_telemetry_stream(
    machine_id: str = "ims_set2_rig",
    local_file: Optional[Path] = None,
    stale_threshold: float = TELEMETRY_STALE_SECONDS,
    timeout: int = TELEMETRY_TIMEOUT
) -> Dict[str, Any]:
    """
    Centralized telemetry stream resolver.

    Source-Selection Policy:
    1. In deployed and local environments, always attempt to fetch fresh remote telemetry from
       the backend first via GET /api/telemetry/latest.
    2. If remote telemetry is found and valid, check staleness. Return with source='backend'.
    3. If remote backend returns 404 (TELEMETRY_NOT_FOUND, i.e. no data streamed yet):
       - If a valid local fallback file exists, use it (source='local_fallback') and indicate
         that backend has no records yet.
       - Otherwise return status='no_telemetry' (source=None).
    4. If remote backend is unreachable, times out, or fails with 5xx:
       - If a valid local fallback file exists, use it (source='local_fallback') with status
         'backend_unavailable_fallback_used'.
       - Otherwise return status='backend_unavailable' (source=None).
    5. If neither remote backend nor local file is available, return status='unavailable'.
    """
    # ----------------------------------------------------
    # Stage 1: Try Remote Backend
    # ----------------------------------------------------
    remote_error: Optional[str] = None
    remote_data: Optional[Dict[str, Any]] = None
    backend_responded: bool = False

    try:
        remote_data = get_latest_telemetry(machine_id=machine_id, timeout=timeout)
        backend_responded = True
    except ApiError as err:
        remote_error = err.message
        logger.warning(f"Backend telemetry request failed ({err.code}): {err.message}")
    except Exception as exc:
        remote_error = str(exc)
        logger.warning(f"Unexpected error contacting backend: {exc}")

    # Case A: Remote Telemetry Received
    if backend_responded and remote_data is not None:
        validated = validate_telemetry_payload(remote_data)
        if validated is not None:
            is_stale, age_sec, age_str, ts_display = calculate_telemetry_age(
                validated, stale_threshold=stale_threshold
            )
            return {
                "telemetry": validated,
                "source": "backend",
                "status": "stale" if is_stale else "fresh",
                "is_stale": is_stale,
                "age_seconds": age_sec,
                "age_str": age_str,
                "last_updated_str": ts_display,
                "message": "Live telemetry streaming from Render backend.",
                "error": None
            }
        else:
            logger.warning("Remote telemetry payload was malformed. Falling back to local state.")

    # ----------------------------------------------------
    # Stage 2: Fallback to Local Data File if Appropriate
    # ----------------------------------------------------
    local_data = load_local_telemetry_file(local_file)

    if local_data is not None:
        is_stale, age_sec, age_str, ts_display = calculate_telemetry_age(
            local_data, stale_threshold=stale_threshold
        )
        if backend_responded:
            status = "local_fallback"
            msg = "Backend currently has no telemetry snapshots. Displaying local file fallback."
        else:
            status = "backend_unavailable_fallback_used"
            msg = f"Backend temporarily unavailable ({remote_error}). Displaying local file fallback."

        return {
            "telemetry": local_data,
            "source": "local_fallback",
            "status": status,
            "is_stale": is_stale,
            "age_seconds": age_sec,
            "age_str": age_str,
            "last_updated_str": ts_display,
            "message": msg,
            "error": remote_error
        }

    # ----------------------------------------------------
    # Stage 3: No Data Available Anywhere
    # ----------------------------------------------------
    if backend_responded:
        return {
            "telemetry": None,
            "source": None,
            "status": "no_telemetry",
            "is_stale": False,
            "age_seconds": None,
            "age_str": None,
            "last_updated_str": None,
            "message": f"Connected to backend at {BACKEND_URL}. Waiting for live telemetry stream from MQTT bridge...",
            "error": None
        }
    elif remote_error:
        return {
            "telemetry": None,
            "source": None,
            "status": "backend_unavailable",
            "is_stale": False,
            "age_seconds": None,
            "age_str": None,
            "last_updated_str": None,
            "message": f"Flask backend at {BACKEND_URL} is currently unreachable ({remote_error}).",
            "error": remote_error
        }
    else:
        return {
            "telemetry": None,
            "source": None,
            "status": "unavailable",
            "is_stale": False,
            "age_seconds": None,
            "age_str": None,
            "last_updated_str": None,
            "message": "No active telemetry found from backend or local data file.",
            "error": None
        }
