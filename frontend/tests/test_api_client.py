"""
test_api_client.py - Unit tests for Streamlit api_client and csv_validation utilities.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from frontend.services.api_client import (
    get_health,
    list_machines,
    create_machine,
    ApiError
)
from frontend.utils.csv_validation import validate_snapshot_csv


def test_csv_validation_valid():
    """Verifies client-side CSV validation with a valid 20480x4 synthetic array."""
    # Generate 20480x4 float csv text
    arr = np.random.randn(20480, 4).astype(np.float64)
    csv_bytes = "\n".join([f"{r[0]},{r[1]},{r[2]},{r[3]}" for r in arr]).encode("utf-8")

    is_valid, err, details, df = validate_snapshot_csv(csv_bytes, "test.csv")
    assert is_valid is True
    assert err is None
    assert details["rows"] == 20480
    assert details["cols"] == 4
    assert df is not None


def test_csv_validation_invalid_shape():
    """Verifies that wrong row count fails validation cleanly."""
    arr = np.random.randn(100, 4)
    csv_bytes = "\n".join([f"{r[0]},{r[1]},{r[2]},{r[3]}" for r in arr]).encode("utf-8")

    is_valid, err, details, df = validate_snapshot_csv(csv_bytes, "test.csv")
    assert is_valid is False
    assert "Expected exactly 20480" in err


@patch("requests.request")
def test_api_client_success(mock_request):
    """Verifies api_client parses success response envelopes."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": {"service": "machinemind-backend", "status": "ok"},
        "error": None
    }
    mock_request.return_value = mock_resp

    res = get_health()
    assert res["service"] == "machinemind-backend"
    assert res["status"] == "ok"


@patch("requests.request")
def test_api_client_error_envelope(mock_request):
    """Verifies api_client raises ApiError on error envelopes."""
    mock_resp = MagicMock()
    mock_resp.status_code = 409
    mock_resp.json.return_value = {
        "success": False,
        "data": None,
        "error": {
            "code": "MACHINE_EXISTS",
            "message": "Machine already exists.",
            "details": {}
        }
    }
    mock_request.return_value = mock_resp

    with pytest.raises(ApiError) as exc:
        create_machine("dup_id", "Name")

    assert exc.value.code == "MACHINE_EXISTS"
    assert exc.value.message == "Machine already exists."
