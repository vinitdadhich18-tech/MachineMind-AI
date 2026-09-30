"""
test_inference.py - Integration and validation tests for POST /api/inference.
"""

import io
from pathlib import Path
import pytest
import numpy as np

from backend.config import Config, BASE_DIR

# Locate a real NASA IMS snapshot file in the repository (read-only)
REAL_SNAPSHOT_PATH = BASE_DIR / "ml-service" / "data" / "raw" / "IMS" / "2nd_test" / "2nd_test" / "2004.02.12.10.32.39"


def test_inference_with_real_nasa_snapshot_multipart(client):
    """
    Tests POST /api/inference with a real NASA IMS bearing snapshot file via multipart form-data.
    """
    assert REAL_SNAPSHOT_PATH.exists(), f"Real NASA IMS snapshot missing at {REAL_SNAPSHOT_PATH}"

    with open(REAL_SNAPSHOT_PATH, "rb") as f:
        file_bytes = f.read()

    data = {
        "machine_id": "bearing_set2_rig1",
        "snapshot_time": "2004-02-12T10:32:39Z",
        "file": (io.BytesIO(file_bytes), "2004.02.12.10.32.39.txt")
    }

    response = client.post(
        "/api/inference",
        data=data,
        content_type="multipart/form-data"
    )

    assert response.status_code == 201, f"Response failed with {response.get_json()}"
    payload = response.get_json()

    assert payload["success"] is True
    assert payload["error"] is None

    result = payload["data"]
    assert result["machine_id"] == "bearing_set2_rig1"
    assert result["timestamp"] == "2004-02-12T10:32:39Z"
    assert result["source_filename"] == "2004.02.12.10.32.39.txt"
    assert result["overall"]["state"] in ["normal", "watch", "anomaly_detected"]
    assert len(result["channels"]) == 4

    # Confirm per-channel scores are valid positive floats
    for ch in result["channels"]:
        assert isinstance(ch["anomaly_score"], float)
        assert isinstance(ch["threshold"], float)


def test_inference_with_json_payload(client):
    """
    Tests POST /api/inference with a JSON payload containing a valid (20480, 4) numpy matrix.
    """
    # Create valid zero snapshot
    matrix = np.zeros((20480, 4), dtype=float).tolist()

    json_data = {
        "machine_id": "test_rig_json",
        "snapshot_time": "2026-01-31T12:00:00Z",
        "data": matrix
    }

    response = client.post(
        "/api/inference",
        json=json_data
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["data"]["machine_id"] == "test_rig_json"


def test_inference_validation_failures(client):
    """
    Verifies that invalid input formats return standard JSON error envelopes with proper status codes.
    """
    # 1. Missing machine_id
    res1 = client.post("/api/inference", json={"data": []})
    assert res1.status_code == 400
    assert res1.get_json()["error"]["code"] == "VALIDATION_ERROR"

    # 2. Invalid shape (e.g. 100 rows x 4 cols)
    matrix_bad = np.zeros((100, 4)).tolist()
    res2 = client.post("/api/inference", json={"machine_id": "valid_id", "data": matrix_bad})
    assert res2.status_code == 400
    assert res2.get_json()["error"]["code"] == "INVALID_SHAPE"

    # 3. Unsupported file type (.pdf)
    data_bad_ext = {
        "machine_id": "test_id",
        "file": (io.BytesIO(b"fake content"), "test.pdf")
    }
    res3 = client.post("/api/inference", data=data_bad_ext, content_type="multipart/form-data")
    assert res3.status_code == 415
    assert res3.get_json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
