"""
alert_service.py - Business logic and database operations for Alerts.
"""

import logging
from typing import Any, Dict, List, Optional
from flask import current_app

from backend.utils.db import get_db
from backend.utils.errors import APIError
from backend.models.alert import create_alert_doc, serialize_alert
from backend.services.machine_service import get_machine

logger = logging.getLogger(__name__)


def get_database():
    """Helper to retrieve database instance or raise 503 if unavailable."""
    db = get_db(current_app.config["MONGO_URI"], current_app.config["DATABASE_NAME"])
    if db is None:
        raise APIError("DATABASE_UNAVAILABLE", "MongoDB database connection is unavailable.", status_code=503)
    return db


def evaluate_and_create_alert(prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Evaluates prediction result for persistence-confirmed channel anomalies.
    Creates an alert in MongoDB ONLY when a channel transitions into persistence_confirmed.
    Prevents duplicate open alerts covering the same machine and affected channels.
    """
    db = get_database()
    machine_id = prediction["machine_id"]
    prediction_id = prediction["prediction_id"]
    timestamp = prediction["timestamp"]

    # Identify affected channels with confirmed persistence
    affected_channels = [
        ch["channel"]
        for ch in prediction["channels"]
        if ch.get("persistence_confirmed", False)
    ]

    if not affected_channels:
        return None

    # Check for existing open alert covering the same machine and affected channels
    existing_open = db.alerts.find_one({
        "machine_id": machine_id,
        "status": "open",
        "affected_channels": {"$all": affected_channels}
    })

    if existing_open:
        logger.info(f"Open alert already exists for machine_id='{machine_id}', skipping duplicate alert creation.")
        return None

    # Create new alert
    doc = create_alert_doc(
        machine_id=machine_id,
        prediction_id=prediction_id,
        timestamp=timestamp,
        affected_channels=affected_channels
    )

    db.alerts.insert_one(doc)
    logger.info(f"Created new alert_id='{doc['alert_id']}' for machine_id='{machine_id}', channels={affected_channels}")
    return serialize_alert(doc)


def list_alerts(
    machine_id: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Lists alerts filtered by machine_id, status, severity with pagination (newest first).
    """
    if machine_id:
        get_machine(machine_id)

    if not isinstance(limit, int) or limit < 1 or limit > 200:
        raise APIError("VALIDATION_ERROR", "Parameter 'limit' must be an integer between 1 and 200.", status_code=400)

    if not isinstance(offset, int) or offset < 0:
        raise APIError("VALIDATION_ERROR", "Parameter 'offset' must be a non-negative integer.", status_code=400)

    query: Dict[str, Any] = {}
    if machine_id:
        query["machine_id"] = machine_id

    if status:
        if status not in ["open", "acknowledged", "resolved"]:
            raise APIError("VALIDATION_ERROR", "Parameter 'status' must be one of: open, acknowledged, resolved.", status_code=400)
        query["status"] = status

    if severity:
        if severity not in ["warning", "high"]:
            raise APIError("VALIDATION_ERROR", "Parameter 'severity' must be one of: warning, high.", status_code=400)
        query["severity"] = severity

    db = get_database()
    total = db.alerts.count_documents(query)
    cursor = db.alerts.find(query, sort=[("timestamp", -1)]).skip(offset).limit(limit)

    alerts = [serialize_alert(doc) for doc in cursor]

    res = {
        "count": len(alerts),
        "total": total,
        "alerts": alerts
    }
    if machine_id:
        res["machine_id"] = machine_id

    return res


def update_alert_status(alert_id: str, new_status: str) -> Dict[str, Any]:
    """
    Updates status of an alert ('acknowledged' or 'resolved').
    """
    if new_status not in ["acknowledged", "resolved", "open"]:
        raise APIError("VALIDATION_ERROR", "Status must be 'acknowledged' or 'resolved'.", status_code=400)

    db = get_database()
    alert_doc = db.alerts.find_one({"alert_id": alert_id})
    if not alert_doc:
        raise APIError("NOT_FOUND", f"Alert with ID '{alert_id}' not found.", status_code=404)

    db.alerts.update_one({"alert_id": alert_id}, {"$set": {"status": new_status}})
    alert_doc["status"] = new_status
    logger.info(f"Updated alert_id='{alert_id}' status to '{new_status}'")
    return serialize_alert(alert_doc)
