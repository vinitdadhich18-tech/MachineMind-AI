"""
data_service.py - Validation utilities for snapshot uploads and JSON payloads in MachineMind AI.
"""

import io
import re
import logging
from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

from backend.utils.errors import APIError

logger = logging.getLogger(__name__)

MACHINE_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ALLOWED_EXTENSIONS = {".csv", ".txt", ".tsv"}


def validate_machine_id(machine_id: Optional[str]) -> str:
    """
    Validates machine_id against regex ^[A-Za-z0-9_-]{1,64}$.
    Raises APIError 400 if invalid or missing.
    """
    if not machine_id or not isinstance(machine_id, str):
        raise APIError(
            code="VALIDATION_ERROR",
            message="Field 'machine_id' is required.",
            status_code=400
        )

    clean_id = machine_id.strip()
    if not MACHINE_ID_REGEX.match(clean_id):
        raise APIError(
            code="VALIDATION_ERROR",
            message="Field 'machine_id' must be 1-64 characters matching ^[A-Za-z0-9_-]+$.",
            status_code=400,
            details={"machine_id": machine_id}
        )

    return clean_id


def parse_and_validate_snapshot_file(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Parses and validates a file upload containing a 1-second raw vibration snapshot.

    Rules:
    - Allowed extension: .csv, .txt, .tsv
    - Delimiter auto-detected (comma, tab, whitespace)
    - Exactly 20480 rows x 4 columns
    - All values numeric and finite (no NaN / Inf)
    """
    if not filename:
        raise APIError(
            code="UNSUPPORTED_FILE_TYPE",
            message="Uploaded file filename is missing.",
            status_code=415
        )

    # Check extension or NASA IMS timestamp format (e.g. 2004.02.12.10.32.39)
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    is_nasa_format = bool(re.match(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$", filename))

    if ext not in ALLOWED_EXTENSIONS and not is_nasa_format:
        raise APIError(
            code="UNSUPPORTED_FILE_TYPE",
            message=f"File extension or format not supported. Allowed extensions: .csv, .txt, .tsv or NASA IMS timestamp filename.",
            status_code=415,
            details={"allowed_extensions": list(ALLOWED_EXTENSIONS), "provided_filename": filename}
        )

    if len(file_bytes) == 0:
        raise APIError(
            code="VALIDATION_ERROR",
            message="Uploaded snapshot file is empty.",
            status_code=400
        )

    # Attempt parsing delimiter (comma, tab, or whitespace)
    content_str = file_bytes.decode("utf-8", errors="replace")

    df = None
    parse_errors = []

    # Try standard delims
    for sep in [r"\s+", r",", r"\t"]:
        try:
            temp_df = pd.read_csv(
                io.StringIO(content_str),
                sep=sep,
                header=None,
                engine="python"
            )
            # If the first row contains string headers, try dropping it or auto header
            if temp_df.shape[1] == 4:
                # Check if row 0 has strings
                first_row = temp_df.iloc[0].values
                if any(isinstance(val, str) and not val.replace(".", "", 1).replace("-", "", 1).isdigit() for val in first_row):
                    temp_df = temp_df.iloc[1:].reset_index(drop=True)

            if temp_df.shape[1] == 4:
                df = temp_df
                break
        except Exception as e:
            parse_errors.append(str(e))

    if df is None:
        raise APIError(
            code="INVALID_SHAPE",
            message="Failed to parse file into 4 numeric vibration columns.",
            status_code=400,
            details={"parse_attempts": parse_errors}
        )

    # Convert to numeric array
    try:
        arr = df.to_numpy(dtype=np.float64)
    except Exception as e:
        raise APIError(
            code="INVALID_VALUES",
            message="Snapshot contains non-numeric characters or unparseable text.",
            status_code=400,
            details={"error": str(e)}
        )

    # Check shape exactly 20480 x 4
    actual_rows, actual_cols = arr.shape
    if actual_rows != 20480 or actual_cols != 4:
        raise APIError(
            code="INVALID_SHAPE",
            message=f"Expected 20480 rows x 4 columns, got {actual_rows} x {actual_cols}.",
            status_code=400,
            details={"expected_rows": 20480, "expected_cols": 4, "actual_rows": actual_rows, "actual_cols": actual_cols}
        )

    # Check NaN / Inf
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count > 0 or inf_count > 0:
        # Find first offending index
        invalid_mask = np.isnan(arr) | np.isinf(arr)
        first_idx = np.argwhere(invalid_mask)[0]
        first_row, first_col = int(first_idx[0]), int(first_idx[1])

        raise APIError(
            code="INVALID_VALUES",
            message=f"Snapshot contains non-finite values (NaN count: {nan_count}, Inf count: {inf_count}).",
            status_code=400,
            details={
                "nan_count": nan_count,
                "inf_count": inf_count,
                "first_invalid_location": {"row": first_row, "channel": first_col + 1}
            }
        )

    return arr


def validate_json_snapshot_data(data: Any) -> np.ndarray:
    """
    Validates JSON array payload data of shape (20480, 4).
    """
    if not isinstance(data, list):
        raise APIError(
            code="VALIDATION_ERROR",
            message="JSON field 'data' must be a 2D list of shape (20480, 4).",
            status_code=400
        )

    try:
        arr = np.array(data, dtype=np.float64)
    except Exception as e:
        raise APIError(
            code="INVALID_VALUES",
            message="JSON 'data' matrix contains non-numeric elements.",
            status_code=400,
            details={"error": str(e)}
        )

    if arr.ndim != 2 or arr.shape != (20480, 4):
        actual_rows = arr.shape[0] if arr.ndim >= 1 else 0
        actual_cols = arr.shape[1] if arr.ndim == 2 else 0
        raise APIError(
            code="INVALID_SHAPE",
            message=f"Expected 20480 rows x 4 columns, got {actual_rows} x {actual_cols}.",
            status_code=400,
            details={"expected_rows": 20480, "expected_cols": 4, "actual_rows": actual_rows, "actual_cols": actual_cols}
        )

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count > 0 or inf_count > 0:
        raise APIError(
            code="INVALID_VALUES",
            message=f"JSON data contains non-finite values (NaN: {nan_count}, Inf: {inf_count}).",
            status_code=400,
            details={"nan_count": nan_count, "inf_count": inf_count}
        )

    return arr
