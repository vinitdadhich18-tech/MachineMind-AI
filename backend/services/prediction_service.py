"""
prediction_service.py - Business logic for storing predictions and updating machine status.
"""

import logging
from typing import Any, Dict
from flask import current_app

from backend.utils.db import get_db
from backend.utils.errors import APIError
from backend.models.prediction import serialize_prediction
from backend.services.machine_service import update_machine_status
from backend.services.alert_service import evaluate_and_create_alert

logger = logging.getLogger(__name__)


def get_database():
    """Helper to retrieve database instance or raise 503 if unavailable."""
    db = get_db(current_app.config["MONGO_URI"], current_app.config["DATABASE_NAME"])
    if db is None:
        raise APIError("DATABASE_UNAVAILABLE", "MongoDB database connection is unavailable.", status_code=503)
    return db


def process_and_save_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Saves prediction metadata to Mongo, updates machine status, and evaluates alerts.
    Does NOT store raw 20480x4 signal arrays.
    """
    db = get_database()
    machine_id = prediction["machine_id"]

    # 1. Evaluate alert creation
    alert_info = evaluate_and_create_alert(prediction)
    if alert_info:
        prediction["alert"] = {
            "created": True,
            "alert_id": alert_info["alert_id"],
            "severity": alert_info["severity"]
        }
    else:
        prediction["alert"] = {
            "created": False,
            "alert_id": None,
            "severity": None
        }

    # 2. Save prediction document to Mongo
    db.predictions.insert_one(dict(prediction))

    # 3. Update machine status
    overall_state = prediction["overall"]["state"]
    update_machine_status(machine_id, overall_state)

    logger.info(f"Saved prediction_id='{prediction['prediction_id']}' for machine_id='{machine_id}', state='{overall_state}'")
    return serialize_prediction(prediction)
