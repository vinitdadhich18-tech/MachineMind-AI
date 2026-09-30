"""
test_validation.py - Unit tests for data validation rules and POST /api/data/upload endpoint.
"""

import io
import pytest
import numpy as np

from backend.config import BASE_DIR
from backend.services.data_service import (
    validate_machine_id,
    parse_and_validate_snapshot_file,
    validate_json_snapshot_data
)
from backend.utils.errors import APIError

REAL_SNAPSHOT_PATH = BASE_DIR / "ml-service" / "data" / "raw" / "IMS" / "2nd_test" / "2nd_test" / "2004.02.12.10.32.39"


def test_validate_machine_id():
    """Verifies regex validation for machine_id."""
    assert validate_machine_id("rig_1") == "rig_1"
    assert validate_machine_id("bearing-set2-ch1") == "bearing-set2-ch1"

    with pytest.raises(APIError) as exc1:
        validate_machine_id("invalid spaces id")
    assert exc1.value.code == "VALIDATION_ERROR"

    with pytest.raises(APIError) as exc2:
        validate_machine_id("")
    assert exc2.value.code == "VALIDATION_ERROR"


def test_validate_only_upload_endpoint(client):
    """Verifies POST /api/data/upload returns file summary without running inference or storing data."""
    assert REAL_SNAPSHOT_PATH.exists()

    with open(REAL_SNAPSHOT_PATH, "rb") as f:
        file_bytes = f.read()

    data = {
        "file": (io.BytesIO(file_bytes), "2004.02.12.10.32.39")
    }

    response = client.post(
        "/api/data/upload",
        data=data,
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["error"] is None

    result = payload["data"]
    assert result["rows"] == 20480
    assert result["columns"] == 4
    assert result["valid"] is True
    assert len(result["channel_summary"]) == 4

    for ch_sum in result["channel_summary"]:
        assert "min" in ch_sum
        assert "max" in ch_sum
        assert "mean" in ch_sum
        assert "std" in ch_sum
