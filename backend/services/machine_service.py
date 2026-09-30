"""
machine_service.py - Business logic and database operations for Machines.
"""

import logging
from typing import Any, Dict, List, Optional
from flask import current_app

from backend.utils.db import get_db
from backend.utils.errors import APIError
from backend.utils.responses import get_utc_now_iso
from backend.services.data_service import validate_machine_id
from backend.models.machine import create_machine_doc, serialize_machine

logger = logging.getLogger(__name__)


def get_database():
    """Helper to retrieve database instance or raise 503 if unavailable."""
    db = get_db(current_app.config["MONGO_URI"], current_app.config["DATABASE_NAME"])
    if db is None:
        raise APIError("DATABASE_UNAVAILABLE", "MongoDB database connection is unavailable.", status_code=503)
    return db


def create_machine(machine_id: str, name: str, description: Optional[str] = None) -> Dict[str, Any]:
    """Creates a new machine in MongoDB."""
    clean_id = validate_machine_id(machine_id)

    if not name or not isinstance(name, str) or len(name.strip()) == 0:
        raise APIError("VALIDATION_ERROR", "Field 'name' is required.", status_code=400)

    db = get_database()

    existing = db.machines.find_one({"machine_id": clean_id})
    if existing:
        raise APIError("MACHINE_EXISTS", f"Machine with ID '{clean_id}' already exists.", status_code=409)

    doc = create_machine_doc(clean_id, name, description)
    db.machines.insert_one(doc)
    logger.info(f"Created new machine machine_id='{clean_id}'")
    return serialize_machine(doc)


def get_machine(machine_id: str) -> Dict[str, Any]:
    """Retrieves machine details along with latest prediction summary and open alert count."""
    clean_id = validate_machine_id(machine_id)
    db = get_database()

    doc = db.machines.find_one({"machine_id": clean_id})
    if not doc:
        raise APIError("MACHINE_NOT_FOUND", f"Machine '{clean_id}' not found.", status_code=404)

    latest_pred = db.predictions.find_one({"machine_id": clean_id}, sort=[("timestamp", -1)])
    open_alerts_count = db.alerts.count_documents({"machine_id": clean_id, "status": "open"})

    serialized = serialize_machine(doc, latest_prediction=latest_pred)
    serialized["open_alerts_count"] = open_alerts_count
    return serialized


def list_machines() -> List[Dict[str, Any]]:
    """Lists all registered machines with their latest prediction summaries."""
    db = get_database()
    cursor = db.machines.find({}, sort=[("created_at", -1)])

    machines = []
    for doc in cursor:
        clean_id = doc["machine_id"]
        latest_pred = db.predictions.find_one({"machine_id": clean_id}, sort=[("timestamp", -1)])
        machines.append(serialize_machine(doc, latest_prediction=latest_pred))

    return machines


def update_machine_status(machine_id: str, new_status: str) -> None:
    """Updates machine status in DB."""
    db = get_database()
    now_iso = get_utc_now_iso()
    db.machines.update_one(
        {"machine_id": machine_id},
        {"$set": {"status": new_status, "updated_at": now_iso}}
    )
