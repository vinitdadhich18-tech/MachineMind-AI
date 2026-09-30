"""
inference.py - REST route handler for POST /api/inference.
"""

import logging
from flask import Blueprint, request
from werkzeug.utils import secure_filename

from backend.utils.responses import success_response
from backend.utils.errors import APIError
from backend.services.data_service import (
    validate_machine_id,
    parse_and_validate_snapshot_file,
    validate_json_snapshot_data
)
from backend.services.ml_service import run_inference, is_model_loaded

inference_bp = Blueprint("inference", __name__)
logger = logging.getLogger(__name__)


@inference_bp.route("/inference", methods=["POST"])
def post_inference():
    """
    POST /api/inference
    Validates upload or JSON payload -> runs ML inference -> responds with prediction record.

    Accepts:
    1. multipart/form-data (file, machine_id, optional snapshot_time)
    2. application/json ({ machine_id, snapshot_time, data })
    """
    if not is_model_loaded():
        raise APIError("MODEL_UNAVAILABLE", "ML inference model is loaded/unavailable.", status_code=503)

    machine_id = None
    snapshot_time = None
    source_filename = None
    arr_snapshot = None

    if request.content_type and "multipart/form-data" in request.content_type:
        machine_id = validate_machine_id(request.form.get("machine_id"))
        snapshot_time = request.form.get("snapshot_time")

        if "file" not in request.files:
            raise APIError("VALIDATION_ERROR", "Multipart request missing required file field 'file'.", status_code=400)

        file_obj = request.files["file"]
        if not file_obj or file_obj.filename == "":
            raise APIError("VALIDATION_ERROR", "No file uploaded in 'file' field.", status_code=400)

        source_filename = secure_filename(file_obj.filename) or "upload.csv"
        file_bytes = file_obj.read()
        arr_snapshot = parse_and_validate_snapshot_file(file_bytes, file_obj.filename)

    elif request.is_json:
        json_body = request.get_json() or {}
        machine_id = validate_machine_id(json_body.get("machine_id"))
        snapshot_time = json_body.get("snapshot_time")
        source_filename = json_body.get("source_filename")

        if "data" not in json_body:
            raise APIError("VALIDATION_ERROR", "JSON payload missing required field 'data'.", status_code=400)

        arr_snapshot = validate_json_snapshot_data(json_body["data"])

    else:
        raise APIError(
            "UNSUPPORTED_FILE_TYPE",
            "Content-Type must be 'multipart/form-data' or 'application/json'.",
            status_code=415
        )

    # Run inference pipeline
    prediction_record = run_inference(
        machine_id=machine_id,
        snapshot=arr_snapshot,
        timestamp=snapshot_time,
        source_filename=source_filename
    )

    return success_response(data=prediction_record, status_code=201)
