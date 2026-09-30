"""
health.py - Health check route for MachineMind AI Flask Backend.
"""

from flask import Blueprint, current_app
from backend.utils.responses import success_response, get_utc_now_iso
from backend.utils.db import check_db_health

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    GET /api/health
    Returns service, database, model status, and current UTC time.
    """
    db_status = check_db_health(
        current_app.config["MONGO_URI"],
        current_app.config["DATABASE_NAME"]
    )

    model_loaded = getattr(current_app, "model_loaded", False)
    model_version = getattr(current_app, "model_version", "v1")

    service_status = "ok" if (db_status == "connected" and model_loaded) else "degraded"

    data = {
        "service": "machinemind-backend",
        "status": service_status,
        "database": db_status,
        "model": {
            "loaded": model_loaded,
            "version": model_version
        },
        "time": get_utc_now_iso()
    }
    return success_response(data=data, status_code=200)
