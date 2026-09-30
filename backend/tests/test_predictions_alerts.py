"""
test_predictions_alerts.py - Integration tests for prediction persistence, status updates, and alert rules.
"""

import io
from pathlib import Path
import numpy as np

from backend.config import BASE_DIR
from backend.utils.db import set_test_db, get_db
from flask import current_app

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

    # Snapshot 4: subequent flagged snapshot -> alert NOT re-created (duplicate prevention)
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
