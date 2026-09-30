"""
alert.py - Serialization and document utilities for Alert entity.
"""

from typing import Any, Dict, List
import uuid
from backend.utils.responses import get_utc_now_iso


def create_alert_doc(
    machine_id: str,
    prediction_id: str,
    timestamp: str,
    affected_channels: List[int]
) -> Dict[str, Any]:
    """
    Creates a new MongoDB alert document using approved scientific wording.
    Severity: 'warning' if 1 channel affected, 'high' if 2+ channels affected.
    """
    now_iso = get_utc_now_iso()
    severity = "high" if len(affected_channels) >= 2 else "warning"

    ch_str = ", ".join(f"Channel {ch}" for ch in sorted(affected_channels))
    msg = f"Confirmed abnormal vibration pattern: {ch_str} anomaly score exceeded threshold for 3 consecutive snapshots."

    return {
        "alert_id": str(uuid.uuid4()),
        "machine_id": machine_id,
        "prediction_id": prediction_id,
        "timestamp": timestamp,
        "alert_type": "persistent_vibration_anomaly",
        "severity": severity,
        "affected_channels": affected_channels,
        "message": msg,
        "status": "open",
        "created_at": now_iso
    }


def serialize_alert(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Serializes Mongo alert document for API responses (removes _id)."""
    return {
        "alert_id": doc["alert_id"],
        "machine_id": doc["machine_id"],
        "prediction_id": doc.get("prediction_id"),
        "timestamp": doc["timestamp"],
        "alert_type": doc.get("alert_type", "persistent_vibration_anomaly"),
        "severity": doc["severity"],
        "affected_channels": doc["affected_channels"],
        "message": doc["message"],
        "status": doc["status"],
        "created_at": doc.get("created_at", doc["timestamp"])
    }
