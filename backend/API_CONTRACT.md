# MachineMind AI — Flask Backend API Contract

> **Source of Truth for Frontend Integration**  
> **Backend Version:** 1.0.0  
> **Environment:** Python 3.14.2 / Flask REST API  
> **Status:** Frozen & Verified  

---

## 1. Scientific & Terminology Boundaries

MachineMind AI is an **unsupervised vibration anomaly-detection research prototype** for rotating machinery (trained/evaluated on the NASA IMS bearing dataset). It is **not** a physical failure predictor, RUL predictor, or fault classifier.

### Mandatory UI & API Wording Rules:
* **Approved Terms:** `"Normal"`, `"Vibration anomaly detected"`, `"Potential abnormal vibration pattern"`, `"Channel X anomaly score exceeded threshold"`, `"Anomalous snapshot — awaiting persistence confirmation"`.
* **Forbidden Terms:** `"Bearing failure"`, `"Machine will fail"`, `"Failure in N hours"`, `"RUL"`, accuracy percentages (e.g. `"99% accurate"`), or latency promises.

---

## 2. Standard Response Envelope

Every endpoint returns a consistent JSON envelope with top-level keys `success`, `data`, and `error`.

### Success Envelope (`HTTP 200 / 201`):
```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

### Error Envelope (`HTTP 400 / 404 / 409 / 413 / 415 / 422 / 500 / 503`):
```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "INVALID_SHAPE",
    "message": "Expected 20480 rows x 4 columns, got 1000 x 4.",
    "details": {
      "expected_rows": 20480,
      "actual_rows": 1000
    }
  }
}
```

---

## 3. Standard Error Codes (`error.code`)

| Code | HTTP Status | Description |
| :--- | :--- | :--- |
| `VALIDATION_ERROR` | 400 / 422 | Missing or malformed parameters (e.g., bad `machine_id`). |
| `INVALID_SHAPE` | 400 | Array shape is not exactly $20480 \times 4$. |
| `INVALID_VALUES` | 400 | Snapshot contains non-numeric, `NaN`, or `Inf` values. |
| `UNSUPPORTED_FILE_TYPE` | 415 | File extension is not `.csv`, `.txt`, `.tsv` or NASA IMS format. |
| `PAYLOAD_TOO_LARGE` | 413 | File upload exceeds `MAX_UPLOAD_MB` limit. |
| `MACHINE_NOT_FOUND` | 404 | Target `machine_id` does not exist in database. |
| `MACHINE_EXISTS` | 409 | Duplicate `machine_id` during creation. |
| `MODEL_UNAVAILABLE` | 503 | ML model artifacts missing or failed to load. |
| `DATABASE_UNAVAILABLE` | 503 | MongoDB connection unreachable. |
| `NOT_FOUND` | 404 | Endpoint route or resource ID not found. |
| `INTERNAL_ERROR` | 500 | Unexpected server error. |

---

## 4. Endpoints & Verified Payloads

### 4.1 Health Check — `GET /api/health`

**Description:** Returns service operational status, MongoDB connectivity, ML model loading status, and UTC timestamp.

**Response (`200 OK`):**
```json
{
  "success": true,
  "data": {
    "service": "machinemind-backend",
    "status": "ok",
    "database": "connected",
    "model": {
      "loaded": true,
      "version": "v1"
    },
    "time": "2026-10-01T00:30:00Z"
  },
  "error": null
}
```

---

### 4.2 List Machines — `GET /api/machines`

**Description:** Lists all registered machinery along with latest prediction summaries.

**Response (`200 OK`):**
```json
{
  "success": true,
  "data": {
    "machines": [
      {
        "machine_id": "bearing_set2_rig1",
        "name": "Test Rig A",
        "description": "NASA IMS Set 2 Bearing Test Rig",
        "status": "watch",
        "created_at": "2026-10-01T00:00:00Z",
        "updated_at": "2026-10-01T00:15:00Z",
        "latest_prediction": {
          "prediction_id": "87ee2231-8f6e-4632-bc5a-575134199390",
          "timestamp": "2004-02-12T10:32:39Z",
          "overall": {
            "state": "watch",
            "snapshot_flagged": true,
            "persistence_confirmed": false,
            "label": "Anomalous snapshot — awaiting persistence confirmation"
          }
        }
      }
    ]
  },
  "error": null
}
```

---

### 4.3 Get Machine Details — `GET /api/machines/<machine_id>`

**Description:** Retrieves machine details, latest prediction summary, and open alert count.

**Response (`200 OK`):**
```json
{
  "success": true,
  "data": {
    "machine_id": "bearing_set2_rig1",
    "name": "Test Rig A",
    "description": "NASA IMS Set 2 Bearing Test Rig",
    "status": "watch",
    "created_at": "2026-10-01T00:00:00Z",
    "updated_at": "2026-10-01T00:15:00Z",
    "open_alerts_count": 0,
    "latest_prediction": {
      "prediction_id": "87ee2231-8f6e-4632-bc5a-575134199390",
      "timestamp": "2004-02-12T10:32:39Z",
      "overall": {
        "state": "watch",
        "snapshot_flagged": true,
        "persistence_confirmed": false,
        "label": "Anomalous snapshot — awaiting persistence confirmation"
      }
    }
  },
  "error": null
}
```

---

### 4.4 Create Machine — `POST /api/machines`

**Description:** Registers a new machine. `machine_id` must match `^[A-Za-z0-9_-]{1,64}$`.

**Request Body (`application/json`):**
```json
{
  "machine_id": "bearing_set2_rig1",
  "name": "Test Rig A",
  "description": "NASA IMS Set 2 Bearing Test Rig"
}
```

**Response (`201 Created`):**
```json
{
  "success": true,
  "data": {
    "machine_id": "bearing_set2_rig1",
    "name": "Test Rig A",
    "description": "NASA IMS Set 2 Bearing Test Rig",
    "status": "no_data",
    "created_at": "2026-10-01T00:00:00Z",
    "updated_at": "2026-10-01T00:00:00Z"
  },
  "error": null
}
```

---

### 4.5 Run Inference — `POST /api/inference`

**Description:** Validates raw 1-second vibration snapshot ($20480 \times 4$), executes Isolation Forest feature scoring, updates per-machine persistence state, updates machine status, and returns complete analysis.

#### Input Mode 1: `multipart/form-data` (Primary Streamlit upload)
* `file`: `.csv`, `.txt`, `.tsv`, or NASA IMS filename ($20480 \times 4$ numeric signal)
* `machine_id`: registered machine ID
* `snapshot_time` *(optional)*: ISO-8601 UTC timestamp string

#### Input Mode 2: `application/json`
```json
{
  "machine_id": "bearing_set2_rig1",
  "snapshot_time": "2004-02-12T10:32:39Z",
  "data": [
    [-0.098, 0.046, 0.088, 0.034],
    ... 20480 rows total ...
  ]
}
```

**Response (`201 Created`):**
```json
{
  "success": true,
  "data": {
    "prediction_id": "87ee2231-8f6e-4632-bc5a-575134199390",
    "machine_id": "bearing_set2_rig1",
    "timestamp": "2004-02-12T10:32:39Z",
    "model_version": "v1",
    "source_filename": "2004.02.12.10.32.39",
    "overall": {
      "state": "watch",
      "snapshot_flagged": true,
      "persistence_confirmed": false,
      "label": "Anomalous snapshot — awaiting persistence confirmation"
    },
    "channels": [
      {
        "channel": 1,
        "name": "Channel 1",
        "anomaly_score": 0.7561814294792415,
        "threshold": 0.5131679214172415,
        "snapshot_flagged": true,
        "consecutive_flagged_count": 1,
        "persistence_confirmed": false,
        "state": "watch"
      },
      {
        "channel": 2,
        "name": "Channel 2",
        "anomaly_score": 0.7747685116568014,
        "threshold": 0.4725104927977178,
        "snapshot_flagged": true,
        "consecutive_flagged_count": 1,
        "persistence_confirmed": false,
        "state": "watch"
      },
      {
        "channel": 3,
        "name": "Channel 3",
        "anomaly_score": 0.7382172255831723,
        "threshold": 0.6225579020689226,
        "snapshot_flagged": true,
        "consecutive_flagged_count": 1,
        "persistence_confirmed": false,
        "state": "watch"
      },
      {
        "channel": 4,
        "name": "Channel 4",
        "anomaly_score": 0.7007143321980186,
        "threshold": 0.5066280904227561,
        "snapshot_flagged": true,
        "consecutive_flagged_count": 1,
        "persistence_confirmed": false,
        "state": "watch"
      }
    ],
    "persistence": {
      "required_consecutive_snapshots": 3
    },
    "alert": {
      "created": false,
      "alert_id": null,
      "severity": null
    }
  },
  "error": null
}
```

---

### 4.6 Get Prediction History — `GET /api/predictions/<machine_id>`

**Description:** Retrieves paginated prediction history for a specific machine (newest first).

**Query Parameters:**
* `limit`: int (default `50`, max `200`)
* `offset`: int (default `0`)

**Response (`200 OK`):**
```json
{
  "success": true,
  "data": {
    "machine_id": "bearing_set2_rig1",
    "count": 1,
    "total": 1,
    "predictions": [
      {
        "prediction_id": "87ee2231-8f6e-4632-bc5a-575134199390",
        "timestamp": "2004-02-12T10:32:39Z",
        "model_version": "v1",
        "source_filename": "2004.02.12.10.32.39",
        "overall": {
          "state": "watch",
          "snapshot_flagged": true,
          "persistence_confirmed": false,
          "label": "Anomalous snapshot — awaiting persistence confirmation"
        },
        "channels": [ ... ],
        "persistence": {
          "required_consecutive_snapshots": 3
        },
        "alert": {
          "created": false,
          "alert_id": null,
          "severity": null
        }
      }
    ]
  },
  "error": null
}
```

---

### 4.7 List System Alerts — `GET /api/alerts`

**Description:** Lists alerts across all machines with optional filtering.

**Query Parameters:**
* `status`: `open` | `acknowledged` | `resolved`
* `severity`: `warning` | `high`
* `limit`: int (default `100`, max `200`)
* `offset`: int (default `0`)

**Response (`200 OK`):**
```json
{
  "success": true,
  "data": {
    "count": 1,
    "total": 1,
    "alerts": [
      {
        "alert_id": "454cefd0-af05-4026-b822-6583f6948ee4",
        "machine_id": "bearing_set2_rig1",
        "prediction_id": "ba2ca558-e169-48ee-92b4-4b5502e56f5b",
        "timestamp": "2004-02-12T10:52:39Z",
        "alert_type": "persistent_vibration_anomaly",
        "severity": "high",
        "affected_channels": [1, 2, 3, 4],
        "message": "Potential abnormal vibration pattern: Channel 1, Channel 2, Channel 3, Channel 4 anomaly score exceeded threshold for 3 consecutive snapshots.",
        "status": "open",
        "created_at": "2026-10-01T00:30:00Z"
      }
    ]
  },
  "error": null
}
```

---

### 4.8 Get Machine Alerts — `GET /api/alerts/<machine_id>`

**Description:** Retrieves alerts filtered by a specific machine.

**Response (`200 OK`):**
```json
{
  "success": true,
  "data": {
    "machine_id": "bearing_set2_rig1",
    "count": 1,
    "total": 1,
    "alerts": [ ... ]
  },
  "error": null
}
```

---

### 4.9 Update Alert Status — `PATCH /api/alerts/<alert_id>`

**Description:** Updates the status of an alert (`acknowledged` or `resolved`).

**Request Body (`application/json`):**
```json
{
  "status": "acknowledged"
}
```

**Response (`200 OK`):**
```json
{
  "success": true,
  "data": {
    "alert_id": "454cefd0-af05-4026-b822-6583f6948ee4",
    "machine_id": "bearing_set2_rig1",
    "prediction_id": "ba2ca558-e169-48ee-92b4-4b5502e56f5b",
    "timestamp": "2004-02-12T10:52:39Z",
    "alert_type": "persistent_vibration_anomaly",
    "severity": "high",
    "affected_channels": [1, 2, 3, 4],
    "message": "Potential abnormal vibration pattern: Channel 1, Channel 2, Channel 3, Channel 4 anomaly score exceeded threshold for 3 consecutive snapshots.",
    "status": "acknowledged",
    "created_at": "2026-10-01T00:30:00Z"
  },
  "error": null
}
```
