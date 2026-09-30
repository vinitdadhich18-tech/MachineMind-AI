"""
alerts.py - REST route handlers for Alert endpoints.
"""

from flask import Blueprint, request
from backend.utils.responses import success_response
from backend.utils.errors import APIError
from backend.services.alert_service import (
    list_alerts,
    update_alert_status
)

alerts_bp = Blueprint("alerts", __name__)


def parse_int_param(val: str, default: int) -> int:
    """Helper to parse integer query parameters."""
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        raise APIError("VALIDATION_ERROR", f"Invalid integer parameter: '{val}'", status_code=400)


@alerts_bp.route("/alerts", methods=["GET"])
def get_all_alerts():
    """
    GET /api/alerts?status=open&severity=high&limit=100&offset=0
    Returns list of all alerts across all machines with optional filters.
    """
    status = request.args.get("status")
    severity = request.args.get("severity")
    limit = parse_int_param(request.args.get("limit"), default=100)
    offset = parse_int_param(request.args.get("offset"), default=0)

    result = list_alerts(status=status, severity=severity, limit=limit, offset=offset)
    return success_response(data=result, status_code=200)


@alerts_bp.route("/alerts/<machine_id>", methods=["GET"])
def get_machine_alerts(machine_id: str):
    """
    GET /api/alerts/<machine_id>?status=open&limit=100
    Returns alerts for a specific machine.
    """
    status = request.args.get("status")
    severity = request.args.get("severity")
    limit = parse_int_param(request.args.get("limit"), default=100)
    offset = parse_int_param(request.args.get("offset"), default=0)

    result = list_alerts(machine_id=machine_id, status=status, severity=severity, limit=limit, offset=offset)
    return success_response(data=result, status_code=200)


@alerts_bp.route("/alerts/<alert_id>", methods=["PATCH"])
def patch_alert_status(alert_id: str):
    """
    PATCH /api/alerts/<alert_id>
    Updates status of an alert (e.g. { "status": "acknowledged" } or { "status": "resolved" }).
    """
    if not request.is_json:
        raise APIError("VALIDATION_ERROR", "Request body must be JSON.", status_code=400)

    body = request.get_json() or {}
    new_status = body.get("status")

    if not new_status:
        raise APIError("VALIDATION_ERROR", "JSON field 'status' is required.", status_code=400)

    updated = update_alert_status(alert_id=alert_id, new_status=new_status)
    return success_response(data=updated, status_code=200)
