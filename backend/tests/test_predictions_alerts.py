"""
test_predictions_alerts.py - Integration tests for prediction history, alerts listing, filtering, and status updates.
"""

import io
import numpy as np

from backend.config import BASE_DIR

REAL_SNAPSHOT_PATH = BASE_DIR / "ml-service" / "data" / "raw" / "IMS" / "2nd_test" / "2nd_test" / "2004.02.12.10.32.39"


def test_persistence_flow_and_alert_trigger(client, app):
    """
    Submits a sequence of real flagged snapshots to verify:
    1. consecutive count increments (1 -> 2 -> 3)
    2. machine status transitions (no_data -> watch -> anomaly_detected)
    3. alert is created ONCE on 3rd snapshot transition
    4. prediction documents stored in Mongo do NOT contain the raw 20480x4 array
    """
    # 1. Create machine
    res_m = client.post("/api/machines", json={"machine_id": "persistence_rig_01", "name": "Persistence Rig 01"})
    assert res_m.status_code == 201

    with open(REAL_SNAPSHOT_PATH, "rb") as f:
        file_bytes = f.read()

    # Snapshot 1: count 1 -> state 'watch'
    res1 = client.post(
        "/api/inference",
        data={"machine_id": "persistence_rig_01", "file": (io.BytesIO(file_bytes), "snap1.txt")},
        content_type="multipart/form-data"
    )
    assert res1.status_code == 201
    p1 = res1.get_json()["data"]
    assert p1["overall"]["state"] == "watch"
    assert p1["overall"]["persistence_confirmed"] is False
    assert p1["alert"]["created"] is False

    # Check machine status in DB -> 'watch'
    m1 = client.get("/api/machines/persistence_rig_01").get_json()["data"]
    assert m1["status"] == "watch"

    # Snapshot 2: count 2 -> state 'watch'
    res2 = client.post(
        "/api/inference",
        data={"machine_id": "persistence_rig_01", "file": (io.BytesIO(file_bytes), "snap2.txt")},
        content_type="multipart/form-data"
    )
    assert res2.status_code == 201
    p2 = res2.get_json()["data"]
    assert p2["overall"]["state"] == "watch"
    assert p2["overall"]["persistence_confirmed"] is False
    assert p2["alert"]["created"] is False

    # Snapshot 3: count 3 -> state 'anomaly_detected', alert created!
    res3 = client.post(
        "/api/inference",
        data={"machine_id": "persistence_rig_01", "file": (io.BytesIO(file_bytes), "snap3.txt")},
        content_type="multipart/form-data"
    )
    assert res3.status_code == 201
    p3 = res3.get_json()["data"]
    assert p3["overall"]["state"] == "anomaly_detected"
    assert p3["overall"]["persistence_confirmed"] is True
    assert p3["alert"]["created"] is True
    assert p3["alert"]["alert_id"] is not None

    # Check machine status in DB -> 'anomaly_detected'
    m3 = client.get("/api/machines/persistence_rig_01").get_json()["data"]
    assert m3["status"] == "anomaly_detected"
    assert m3["open_alerts_count"] == 1

    # Snapshot 4: subsequent flagged snapshot -> alert NOT re-created (duplicate prevention)
    res4 = client.post(
        "/api/inference",
        data={"machine_id": "persistence_rig_01", "file": (io.BytesIO(file_bytes), "snap4.txt")},
        content_type="multipart/form-data"
    )
    assert res4.status_code == 201
    p4 = res4.get_json()["data"]
    assert p4["alert"]["created"] is False

    # Verify prediction document in Mongo does NOT contain raw signal array
    with app.app_context():
        from backend.utils.db import get_db
        db = get_db(app.config["MONGO_URI"], app.config["DATABASE_NAME"])
        doc = db.predictions.find_one({"machine_id": "persistence_rig_01"})
        assert "data" not in doc
        assert "raw_snapshot" not in doc
        assert "signal" not in doc


def test_prediction_history_endpoint(client):
    """Verifies GET /api/predictions/<machine_id> pagination and parameters."""
    client.post("/api/machines", json={"machine_id": "hist_rig", "name": "Hist Rig"})

    # Submit 2 predictions using JSON payload
    matrix = np.zeros((20480, 4), dtype=float).tolist()
    client.post("/api/inference", json={"machine_id": "hist_rig", "data": matrix})
    client.post("/api/inference", json={"machine_id": "hist_rig", "data": matrix})

    res = client.get("/api/predictions/hist_rig?limit=10&offset=0")
    assert res.status_code == 200
    payload = res.get_json()["data"]

    assert payload["machine_id"] == "hist_rig"
    assert payload["total"] == 2
    assert payload["count"] == 2
    assert len(payload["predictions"]) == 2


def test_alerts_endpoints_and_patch_status(client):
    """Verifies GET /api/alerts, GET /api/alerts/<machine_id>, and PATCH /api/alerts/<alert_id>."""
    client.post("/api/machines", json={"machine_id": "alert_rig", "name": "Alert Rig"})
    with open(REAL_SNAPSHOT_PATH, "rb") as f:
        file_bytes = f.read()

    # Trigger 3 snapshots to create 1 alert
    for i in range(3):
        client.post(
            "/api/inference",
            data={"machine_id": "alert_rig", "file": (io.BytesIO(file_bytes), f"snap{i}.txt")},
            content_type="multipart/form-data"
        )

    # 1. GET /api/alerts
    res_alerts = client.get("/api/alerts?status=open")
    assert res_alerts.status_code == 200
    alerts_data = res_alerts.get_json()["data"]
    assert alerts_data["total"] == 1
    alert_item = alerts_data["alerts"][0]
    assert alert_item["status"] == "open"
    assert alert_item["machine_id"] == "alert_rig"
    alert_id = alert_item["alert_id"]

    # 2. GET /api/alerts/<machine_id>
    res_m_alerts = client.get("/api/alerts/alert_rig")
    assert res_m_alerts.status_code == 200
    assert res_m_alerts.get_json()["data"]["count"] == 1

    # 3. PATCH /api/alerts/<alert_id> to 'acknowledged'
    res_patch = client.patch(f"/api/alerts/{alert_id}", json={"status": "acknowledged"})
    assert res_patch.status_code == 200
    assert res_patch.get_json()["data"]["status"] == "acknowledged"

    # Verify status changed to acknowledged
    res_open = client.get("/api/alerts?status=open")
    assert res_open.get_json()["data"]["total"] == 0

    res_ack = client.get("/api/alerts?status=acknowledged")
    assert res_ack.get_json()["data"]["total"] == 1
