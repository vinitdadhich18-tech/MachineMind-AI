"""
data.py - REST route handler for POST /api/data/upload (validate-only endpoint).
"""

import logging
from flask import Blueprint, request
from werkzeug.utils import secure_filename
import numpy as np

from backend.utils.responses import success_response
from backend.utils.errors import APIError
from backend.services.data_service import parse_and_validate_snapshot_file

data_bp = Blueprint("data", __name__)
logger = logging.getLogger(__name__)


@data_bp.route("/data/upload", methods=["POST"])
def post_validate_upload():
    """
    POST /api/data/upload
    Validate-only endpoint: parses and validates file upload, returns shape and per-channel summary statistics.
    No ML inference and no database storage.
    """
    if "file" not in request.files:
        raise APIError("VALIDATION_ERROR", "Multipart request missing required file field 'file'.", status_code=400)

    file_obj = request.files["file"]
    if not file_obj or file_obj.filename == "":
        raise APIError("VALIDATION_ERROR", "No file uploaded in 'file' field.", status_code=400)

    filename = secure_filename(file_obj.filename) or "upload.csv"
    file_bytes = file_obj.read()

    # Parse and validate shape (20480 x 4) and finite numeric values
    arr = parse_and_validate_snapshot_file(file_bytes, file_obj.filename)

    channel_summary = []
    for idx in range(4):
        col_data = arr[:, idx]
        channel_summary.append({
            "channel": idx + 1,
            "min": float(np.min(col_data)),
            "max": float(np.max(col_data)),
            "mean": float(np.mean(col_data)),
            "std": float(np.std(col_data))
        })

    result = {
        "filename": filename,
        "rows": arr.shape[0],
        "columns": arr.shape[1],
        "valid": True,
        "channel_summary": channel_summary
    }

    return success_response(data=result, status_code=200)
