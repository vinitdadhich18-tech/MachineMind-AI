"""
prediction.py - Serialization and document utilities for Prediction entity.
"""

from typing import Any, Dict


def serialize_prediction(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Serializes Mongo prediction document for API responses (removes _id)."""
    return {
        "prediction_id": doc["prediction_id"],
        "machine_id": doc["machine_id"],
        "timestamp": doc["timestamp"],
        "model_version": doc.get("model_version", "v1"),
        "source_filename": doc.get("source_filename"),
        "overall": doc["overall"],
        "channels": doc["channels"],
        "persistence": doc.get("persistence", {"required_consecutive_snapshots": 3}),
        "alert": doc.get("alert", {"created": False, "alert_id": None, "severity": None})
    }
