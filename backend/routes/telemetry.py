"""
telemetry.py - REST route handlers for Live Telemetry ingestion and retrieval.
"""

import hmac
import logging
import os
from flask import Blueprint, request, current_app
from backend.utils.responses import success_response
from backend.utils.errors import APIError
from backend.services.telemetry_service import (
    save_live_telemetry,
    get_latest_telemetry
)

telemetry_bp = Blueprint("telemetry", __name__)
logger = logging.getLogger(__name__)


def require_api_key():
    """
    Validates X-API-KEY header against configured API_KEY.
    Fails closed if API_KEY is not configured in the application environment.
    Never logs or exposes the configured key or incoming request key.
    """
    configured_key = current_app.config.get("API_KEY", "")
    if not configured_key or not str(configured_key).strip():
        logger.error(
            "API_KEY environment variable is not configured on the backend server. "
            "Rejecting telemetry write request (failing closed)."
        )
        raise APIError(
            code="AUTH_CONFIGURATION_ERROR",
            message="Server telemetry write authentication is not configured.",
            status_code=500
        )

    auth_header = request.headers.get("X-API-KEY")
    if not auth_header:
        raise APIError(
            code="UNAUTHORIZED",
            message="Missing required 'X-API-KEY' authentication header.",
            status_code=401
        )

    if not hmac.compare_digest(auth_header.strip(), str(configured_key).strip()):
        raise APIError(
            code="UNAUTHORIZED",
            message="Invalid 'X-API-KEY' authentication credential.",
            status_code=401
        )


@telemetry_bp.route("/telemetry/live", methods=["POST"])
def post_live_telemetry():
    """
    POST /api/telemetry/live
    Ingests latest telemetry state from MQTT dashboard bridge.
    Requires X-API-KEY header matching server API_KEY.
    Guarantees sequence integrity and idempotent persistence in MongoDB Atlas.
    """
    require_api_key()

    if not request.is_json:
        raise APIError("VALIDATION_ERROR", "Request body must be a JSON object.", status_code=400)

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise APIError("VALIDATION_ERROR", "Request body must be a valid JSON object.", status_code=400)

    saved = save_live_telemetry(body)
    return success_response(data=saved, status_code=200)


@telemetry_bp.route("/telemetry/latest", methods=["GET"])
def get_latest_telemetry_endpoint():
    """
    GET /api/telemetry/latest
    Returns the latest recorded telemetry snapshot for the specified machine.
    Read-only public endpoint returning standard envelope.
    Query params: machine_id (optional, default: ims_set2_rig).
    """
    default_machine_id = os.getenv("MQTT_MACHINE_ID", "ims_set2_rig").strip()
    machine_id = request.args.get("machine_id", default_machine_id)

    telemetry = get_latest_telemetry(machine_id)
    return success_response(data=telemetry, status_code=200)
