"""
predictions.py - REST route handlers for Prediction history endpoints.
"""

from flask import Blueprint, request
from backend.utils.responses import success_response
from backend.utils.errors import APIError
from backend.services.prediction_service import get_prediction_history

predictions_bp = Blueprint("predictions", __name__)


def parse_int_param(val: str, default: int) -> int:
    """Helper to parse integer query parameters."""
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        raise APIError("VALIDATION_ERROR", f"Invalid integer parameter: '{val}'", status_code=400)


@predictions_bp.route("/predictions/<machine_id>", methods=["GET"])
def get_machine_predictions(machine_id: str):
    """
    GET /api/predictions/<machine_id>?limit=50&offset=0
    Returns paginated prediction history for machine_id (newest first).
    """
    limit = parse_int_param(request.args.get("limit"), default=50)
    offset = parse_int_param(request.args.get("offset"), default=0)

    result = get_prediction_history(machine_id=machine_id, limit=limit, offset=offset)
    return success_response(data=result, status_code=200)
