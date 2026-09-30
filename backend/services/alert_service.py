"""
alert_service.py - Business logic and database operations for Alerts.
"""

import logging
from typing import Any, Dict, List, Optional
from flask import current_app

from backend.utils.db import get_db
from backend.utils.errors import APIError
from backend.models.alert import create_alert_doc, serialize_alert

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
