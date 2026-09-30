"""
machine.py - Serialization and document utilities for Machine entity.
"""

from typing import Any, Dict, Optional
from backend.utils.responses import get_utc_now_iso


def create_machine_doc(
    machine_id: str,
    name: str,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """Creates a new machine MongoDB document."""
    now_iso = get_utc_now_iso()
    return {
        "machine_id": machine_id,
        "name": name.strip(),
        "description": description.strip() if description else "",
        "status": "no_data",
        "created_at": now_iso,
        "updated_at": now_iso
    }


def serialize_machine(doc: Dict[str, Any], latest_prediction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Serializes Mongo machine document for API responses (removes _id)."""
    res = {
        "machine_id": doc["machine_id"],
        "name": doc["name"],
        "description": doc.get("description", ""),
        "status": doc.get("status", "no_data"),
        "created_at": doc["created_at"],
        "updated_at": doc.get("updated_at", doc["created_at"])
    }
    if latest_prediction is not None:
        res["latest_prediction"] = {
            "prediction_id": latest_prediction.get("prediction_id"),
            "timestamp": latest_prediction.get("timestamp"),
            "overall": latest_prediction.get("overall")
        }
    return res
