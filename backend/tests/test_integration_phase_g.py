"""
test_integration_phase_g.py - End-to-End Phase G Integration Test.

Tests the full flow:
1. Create machine
2. Post 3 consecutive real NASA IMS bearing snapshots
3. Verify 3-snapshot persistence rule (watch -> watch -> anomaly_detected)
4. Verify alert creation on 3rd snapshot transition
5. Query prediction history and alerts endpoints
6. Update alert status via PATCH
"""

import io
from pathlib import Path
import pytest

from backend.config import BASE_DIR

NASA_DATA_DIR = BASE_DIR / "ml-service" / "data" / "raw" / "IMS" / "2nd_test" / "2nd_test"
SNAP1_PATH = NASA_DATA_DIR / "2004.02.12.10.32.39"
SNAP2_PATH = NASA_DATA_DIR / "2004.02.12.10.42.39"
SNAP3_PATH = NASA_DATA_DIR / "2004.02.12.10.52.39"


def test_phase_g_end_to_end_flow(client):
    """Full Phase G end-to-end integration test."""
    assert SNAP1_PATH.exists(), f"Snapshot 1 missing at {SNAP1_PATH}"
    assert SNAP2_PATH.exists(), f"Snapshot 2 missing at {SNAP2_PATH}"
    assert SNAP3_PATH.exists(), f"Snapshot 3 missing at {SNAP3_PATH}"

    machine_id = "nasa_bearing_set2_rig1"

    # Step 1: Create Machine
    res_m = client.post("/api/machines", json={
        "machine_id": machine_id,
        "name": "NASA Bearing Set 2 Test Rig",
        "description": "Sequential IMS bearing replay"
    })
    assert res_m.status_code == 201
    assert res_m.get_json()["data"]["status"] == "no_data"

    # Step 2: Upload Snapshot 1 (count 1 -> state 'watch')
    with open(SNAP1_PATH, "rb") as f1:
        res1 = client.post(
            "/api/inference",
            data={
                "machine_id": machine_id,
                "snapshot_time": "2004-02-12T10:32:39Z",
                "file": (io.BytesIO(f1.read()), "2004.02.12.10.32.39")
            },
            content_type="multipart/form-data"
        )
    assert res1.status_code == 201
    p1 = res1.get_json()["data"]
    assert p1["overall"]["state"] == "watch"
    assert p1["overall"]["persistence_confirmed"] is False
    assert p1["alert"]["created"] is False

    # Verify Machine Status updated to 'watch'
    res_m1 = client.get(f"/api/machines/{machine_id}")
    assert res_m1.get_json()["data"]["status"] == "watch"

    # Step 3: Upload Snapshot 2 (count 2 -> state 'watch')
    with open(SNAP2_PATH, "rb") as f2:
        res2 = client.post(
            "/api/inference",
            data={
                "machine_id": machine_id,
                "snapshot_time": "2004-02-12T10:42:39Z",
                "file": (io.BytesIO(f2.read()), "2004.02.12.10.42.39")
            },
            content_type="multipart/form-data"
        )
    assert res2.status_code == 201
    p2 = res2.get_json()["data"]
    assert p2["overall"]["state"] == "watch"
    assert p2["overall"]["persistence_confirmed"] is False
    assert p2["alert"]["created"] is False

    # Step 4: Upload Snapshot 3 (count 3 -> state 'anomaly_detected', alert created!)
    with open(SNAP3_PATH, "rb") as f3:
        res3 = client.post(
            "/api/inference",
            data={
                "machine_id": machine_id,
                "snapshot_time": "2004-02-12T10:52:39Z",
                "file": (io.BytesIO(f3.read()), "2004.02.12.10.52.39")
            },
            content_type="multipart/form-data"
        )
    assert res3.status_code == 201
    p3 = res3.get_json()["data"]
    assert p3["overall"]["state"] == "anomaly_detected"
    assert p3["overall"]["persistence_confirmed"] is True
    assert p3["alert"]["created"] is True
    alert_id = p3["alert"]["alert_id"]
    assert alert_id is not None

    # Verify Machine Status updated to 'anomaly_detected'
    res_m3 = client.get(f"/api/machines/{machine_id}")
    m3_data = res_m3.get_json()["data"]
    assert m3_data["status"] == "anomaly_detected"
    assert m3_data["open_alerts_count"] == 1

    # Step 5: Query Prediction History
    res_hist = client.get(f"/api/predictions/{machine_id}?limit=50")
    assert res_hist.status_code == 200
    hist_data = res_hist.get_json()["data"]
    assert hist_data["total"] == 3
    assert hist_data["count"] == 3
    assert hist_data["predictions"][0]["timestamp"] == "2004-02-12T10:52:39Z"

    # Step 6: Query Alerts & Acknowledge
    res_alerts = client.get(f"/api/alerts/{machine_id}?status=open")
    assert res_alerts.status_code == 200
    alerts_list = res_alerts.get_json()["data"]["alerts"]
    assert len(alerts_list) == 1
    assert alerts_list[0]["alert_id"] == alert_id
    assert alerts_list[0]["severity"] in ["warning", "high"]

    # Patch Alert to 'acknowledged'
    res_patch = client.patch(f"/api/alerts/{alert_id}", json={"status": "acknowledged"})
    assert res_patch.status_code == 200
    assert res_patch.get_json()["data"]["status"] == "acknowledged"

    # Re-check open alert count on machine -> 0
    res_m4 = client.get(f"/api/machines/{machine_id}")
    assert res_m4.get_json()["data"]["open_alerts_count"] == 0
