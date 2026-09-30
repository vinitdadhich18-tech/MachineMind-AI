# MachineMind AI — Flask Backend Application Layer

Unsupervised vibration anomaly detection REST API built with Flask, PyMongo, and Python 3.14.2, wrapping the frozen `AnomalyInferenceEngine` ML pipeline.

---

## 1. Overview

The backend serves as the orchestration layer between the Streamlit frontend, MongoDB database, and the frozen ML inference artifacts under `ml-service/`.

```
Streamlit Frontend
      │
      ▼ HTTP REST (JSON / multipart form-data)
Flask Backend  (port 5000)
  │            │
  ▼            ▼
MongoDB     Existing ML Inference Engine (ml-service/src/inference.py)
                 │
           IF + PCA model artifacts (ml-service/models/)
```

---

## 2. Requirements & Setup

### Environment Requirements:
* **Python:** 3.14.2 (or 3.10+)
* **MongoDB:** Local MongoDB server running on `mongodb://localhost:27017` (or `mongomock` during unit testing)

### Installation:
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Configuration Variables (`.env`)

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `DATABASE_NAME` | `machinemind` | Target MongoDB database name |
| `ML_SERVICE_PATH` | `ml-service` | Path to `ml-service/` package directory |
| `MODEL_DIR` | `ml-service/models` | Folder containing `iforest_pipeline_v1.joblib` & `model_metadata.json` |
| `CORS_ORIGINS` | `http://localhost:8501` | Allowed CORS origins for Streamlit dashboard |
| `MAX_UPLOAD_MB` | `10` | Max file upload limit in megabytes |
| `PORT` | `5000` | Server listening port |
| `FLASK_DEBUG` | `0` | Debug mode toggle |

---

## 4. Running the Server & Tests

### Start Server:
```bash
python app.py
```
The API server will listen on `http://0.0.0.0:5000/api`.

### Run Test Suite:
```bash
python -m pytest backend/tests
```

---

## 5. Endpoints & `curl` Examples

### Health Check:
```bash
curl -X GET http://localhost:5000/api/health
```

### Register Machine:
```bash
curl -X POST http://localhost:5000/api/machines \
  -H "Content-Type: application/json" \
  -d '{"machine_id": "bearing_rig_01", "name": "NASA Test Rig 1", "description": "Set 2 replay"}'
```

### List Machines:
```bash
curl -X GET http://localhost:5000/api/machines
```

### Get Single Machine:
```bash
curl -X GET http://localhost:5000/api/machines/bearing_rig_01
```

### Upload Snapshot for Inference (`multipart/form-data`):
```bash
curl -X POST http://localhost:5000/api/inference \
  -F "machine_id=bearing_rig_01" \
  -F "file=@../ml-service/data/raw/IMS/2nd_test/2nd_test/2004.02.12.10.32.39"
```

### Get Prediction History:
```bash
curl -X GET "http://localhost:5000/api/predictions/bearing_rig_01?limit=50&offset=0"
```

### List System Alerts:
```bash
curl -X GET "http://localhost:5000/api/alerts?status=open&severity=high"
```

### Acknowledge Alert:
```bash
curl -X PATCH http://localhost:5000/api/alerts/<ALERT_ID> \
  -H "Content-Type: application/json" \
  -d '{"status": "acknowledged"}'
```
