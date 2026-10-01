# MachineMind AI — Phase 0: Repository Audit & Industrial Stack Migration Design Report

**Branch:** `feature/industrial-stack-migration`  
**Baseline Commit:** `e1c4f18 fix(ui): align anomaly monitoring states and telemetry metadata`  
**Date:** October 1, 2026  
**Author:** Antigravity (AI Coding Assistant)  
**Status:** PHASE 0 COMPLETE — AWAITING OWNER APPROVAL BEFORE PHASE 1  

---

## Executive Summary

This report establishes the Phase 0 audit and target architecture design for migrating **MachineMind AI** from a REST/MongoDB/Streamlit prototype into an industrial-grade predictive maintenance platform built natively around **MQTT + InfluxDB + XGBoost + Grafana**.

All findings in this document are grounded in direct inspection of the repository files, dataset contents, and executed test suites. **No production code has been modified during Phase 0.**

---

## 1. Current Architecture (Based on Actual Repository Code)

The current repository is structured as a decoupled 3-tier architecture with an offline ML model packaging pipeline:

```
[Raw NASA IMS Snapshot Files (20480x4)]
             │
             ▼
[ml-service/src/inference.py] ── AnomalyInferenceEngine (Stateful, k=3 persistence)
             │
             ▼
[backend/services/ml_service.py] ── Flask API Layer (Port 5000)
             │
      ┌──────┴────────────────────────┐
      ▼                               ▼
[MongoDB (Port 27017)]      [frontend/services/api_client.py]
 (machines, predictions,              │
  alerts collections)                 ▼
                            [frontend/app.py] (Streamlit Dashboard, Port 8501)
```

### Measured Baseline Code Metrics & Components
- **Dataset**: NASA IMS Set 2 raw files (984 snapshots, 20,480 rows x 4 channels, 10-minute uniform intervals).
- **ML Service**: 
  - [`ml-service/src/feature_extraction.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/feature_extraction.py): Extracts 7 time-domain features (`mean`, `std`, `rms`, `p2p`, `skewness`, `kurtosis`, `crest_factor`) per channel = 28 features total per snapshot.
  - [`ml-service/src/models.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/models.py): `RobustScaler` + `IsolationForest` / `PCA` fit on baseline snapshots 0..159.
  - [`ml-service/src/inference.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/inference.py): Packaging engine that computes anomaly scores, raw flags (`score > threshold_P99`), and stateful persistence (`k=3` consecutive exceedances).
- **Backend**: Flask REST API in `backend/app.py` exposes `/api/health`, `/api/inference`, `/api/machines`, `/api/predictions`, `/api/alerts`, `/api/data`. Uses PyMongo for database operations.
- **Frontend**: Streamlit multi-page dashboard (`frontend/app.py`, `frontend/pages/1_Machines.py` through `5_Model_Info.py`) polling Flask backend via `requests`.

---

## 2. Actual End-to-End Data Flow

1. **User Action / Script**: User uploads a 20,480 x 4 CSV snapshot via Streamlit `2_Upload_Analyze.py` or script calls `POST /api/inference`.
2. **Backend Processing**:
   - `backend/routes/inference.py` receives CSV upload.
   - `backend/utils/validation.py` validates file shape (20480, 4), non-null, numeric bounds.
   - `backend/services/ml_service.py` fetches or initializes the machine's `AnomalyInferenceEngine` instance.
3. **ML Inference**:
   - `extract_time_features` extracts 28 time-domain features.
   - For each channel (`ch1`..`ch4`), `RobustScaler.transform` and `IsolationForest.decision_function` compute anomaly score $S = 0.5 - \text{decision\_function}$.
   - If $S > T_{\text{channel\_P99}}$, `consecutive_exceedances[ch] += 1` else reset to `0`.
   - If `consecutive_exceedances[ch] >= 3`, `sustained_flag = True`.
4. **Persistence & Response**:
   - `backend/services/prediction_service.py` saves prediction document to MongoDB `predictions` collection.
   - If `overall.persistence_confirmed == True`, `alert_service.py` inserts/updates an `open` alert in MongoDB `alerts` collection.
   - Backend returns HTTP 200 JSON envelope to Streamlit.
5. **UI Render**: Streamlit updates status badges (`NORMAL`, `WATCH`, `ANOMALY DETECTED`), channel condition cards, and historical Altair charts.

---

## 3. Keep / Replace / Migrate / Deprecate Matrix

| Module / Component | Current Location | Target Action | Justification & Target Destination |
|---|---|---|---|
| **NASA IMS Dataset (Set 2)** | `ml-service/data/raw/IMS/2nd_test` | **KEEP & REPLAY** | Primary training source and input for MQTT Replay Simulator. |
| **Feature Extraction (Time-Domain)** | `ml-service/src/feature_extraction.py` | **KEEP & UNIFY** | 7 time-domain features per channel are domain-proven and mathematically clean. Single implementation will serve offline training & online streaming. |
| **Feature Extraction (Frequency-Domain)** | `ml-service/src/feature_extraction.py` | **REFACTOR & UNIFY** | Hann-windowed spectral energy and spectral centroid; unify for streaming consumer. |
| **Isolation Forest & PCA** | `ml-service/src/models.py` | **MIGRATE TO BASELINE** | Retain as offline reference baseline models; replace primary live inference model with XGBoost RUL regressor. |
| **RobustScaler Pipeline** | `ml-service/src/models.py` | **KEEP & REFACTOR** | Fit on healthy training segment only; preserve for scaling features before XGBoost/IF. |
| **Flask Backend (Telemetry Path)** | `backend/routes/inference.py` | **DEPRECATE / REPLACE** | Remove Flask from live data path. MQTT consumer will directly process telemetry, extract features, infer XGBoost, and write to InfluxDB. |
| **Flask Backend (Admin / Meta)** | `backend/app.py` | **RETAIN (REDUCED)** | Retain optional lightweight management/replay control endpoint if needed. |
| **MongoDB** | `backend/utils/db.py` | **DEPRECATE / REMOVE** | Replace completely with InfluxDB for time-series features, predictions, and state events. Remove in Phase 9. |
| **Streamlit Dashboard** | `frontend/app.py` | **DEPRECATE / REMOVE** | Primary industrial monitoring UI replaced by provisioned Grafana dashboards. Remove in Phase 9. |
| **Persistence Logic (k=3)** | `backend/services/ml_service.py` | **REDESIGN & MIGRATE** | Re-derive state machine for XGBoost RUL/risk threshold breach with audited state transitions in InfluxDB. |
| **MQTT Broker** | *None* (New) | **ADD (PHASE 1)** | Mosquitto MQTT broker running on `localhost:1883` for decoupled real-time streaming. |
| **InfluxDB** | *None* (New) | **ADD (PHASE 3)** | InfluxDB v2 time-series database storing features, predictions, system state, and pipeline telemetry. |
| **XGBoost RUL Regressor** | *None* (New) | **ADD (PHASE 5)** | Supervised RUL regression model with piecewise linear target ($RUL_{cap} = 500$ snapshots). |
| **Grafana Industrial UI** | *None* (New) | **ADD (PHASE 7)** | Code-provisioned Grafana monitoring console querying InfluxDB. |

---

## 4. MQTT Design Draft

### Role & Transport Strategy (Updated per Owner Specification)
- **Broker**: HiveMQ Cloud (MQTT over TLS on port 8883).
- **Publisher**: `ml-service/src/simulator.py` (reads NASA IMS raw snapshot files chronologically and emits JSON telemetry payloads).
- **Consumer**: `ingestion/consumer.py` (subscribes to MQTT topic, parses/validates payloads, feeds feature pipeline and InfluxDB).
- **QoS Policy**: QoS 1 (At Least Once) for telemetry; duplicate messages handled idempotently via InfluxDB point timestamps.
- **Session Policy**: Clean session = `False`, Client ID = `machinemind-ingestion-consumer` to ensure zero message loss during temporary consumer restarts.
- **Security & TLS**: TLS v1.2/v1.3 with TLS SNI host verification enabled. Credentials loaded strictly from environment variables (`MQTT_USERNAME`, `MQTT_PASSWORD`, `MQTT_BROKER_HOST`, `MQTT_BROKER_PORT`). Zero secrets committed.

---

## 5. MQTT Topic and Payload Proposal

### Topic Structure
```
machinemind/v1/telemetry/{machine_id}
machinemind/v1/status/{machine_id}
```

### JSON Telemetry Payload Contract
```json
{
  "schema_version": "1.0",
  "source": "replayed_nasa_ims",
  "machine_id": "ims_set2_rig",
  "snapshot_sequence": 1,
  "original_timestamp": "2004-02-12T10:32:39Z",
  "ingest_timestamp": "2026-10-01T10:30:00Z",
  "sampling_rate_hz": 20480.0,
  "sample_count": 20480,
  "channels": {
    "ch1": [-0.049, -0.042, 0.015, -0.051],
    "ch2": [-0.071, -0.073, 0.000, 0.020],
    "ch3": [-0.132, -0.007, 0.007, -0.002],
    "ch4": [-0.010, -0.105, 0.000, 0.100]
  }
}
```

---

## 6. InfluxDB Schema Proposal

Organized under bucket `machinemind_telemetry`:

### Measurement 1: `vibration_features`
- **Tags**: `machine_id`, `channel`, `source`
- **Fields**: `mean`, `std`, `rms`, `p2p`, `skewness`, `kurtosis`, `crest_factor`, `spectral_energy`, `spectral_centroid`
- **Timestamp**: Ingest timestamp (or virtual live timeline)

### Measurement 2: `model_predictions`
- **Tags**: `machine_id`, `model_type` ("xgboost"), `model_version` ("v1.0")
- **Fields**: `predicted_rul_snapshots`, `predicted_rul_hours`, `degradation_index`, `failure_risk_pct`

### Measurement 3: `machine_state`
- **Tags**: `machine_id`, `status` ("NORMAL", "WATCH", "CRITICAL_RUL_ALERT")
- **Fields**: `persistence_count`, `threshold_breached` (bool), `status_code` (0=Normal, 1=Watch, 2=Critical)

### Measurement 4: `pipeline_health`
- **Tags**: `service_name` ("mqtt_consumer")
- **Fields**: `messages_received`, `messages_rejected`, `inference_latency_ms`, `db_write_latency_ms`

---

## 7. Timestamp / Replay Strategy

**Hybrid Time Strategy**:
- `original_timestamp` (e.g. `2004-02-12 10:32:39`) is preserved in message metadata and stored as a field in InfluxDB.
- `ingest_timestamp` (mapped to virtual live timeline relative to `now()` maintaining original 10-minute intervals) is used as the InfluxDB point `time`.
- **Grafana Effect**: Visualizes data in relative real-time ("Last 24 hours") during replay while maintaining full historical fidelity.

---

## 8. XGBoost Target Proposal

### Supervised RUL Definition
For NASA IMS Set 2 (984 total 10-minute snapshots, run-to-failure at snapshot 984):
$$\text{Raw RUL}(t) = 984 - t \quad (\text{snapshots})$$

### Piecewise Linear Capped RUL (Target Definition)
$$\text{Target RUL}(t) = \min(500, \, 984 - t)$$
- **Rationale**: Bearing health is stationary during early life (snapshots 0..484). Capping RUL at 500 snapshots prevents tree models from attempting to regress arbitrary high values during the healthy phase where signals are statistically identical.

---

## 9. XGBoost Training & Evaluation Strategy

1. **Split Strategy**:
   - **Train**: Snapshots 0 to 600 (Healthy + early degradation initiation).
   - **Validation (Embargo Gap)**: Snapshots 601 to 750.
   - **Test (Held-Out End of Life)**: Snapshots 751 to 984 (Bearing 1 outer race failure propagation).
2. **Evaluation Metrics**:
   - **RMSE** & **MAE** (in snapshots and hours).
   - **PHM Asymmetric Penalty Score**: Penalizes late predictions (predicting high RUL when machine is about to fail) much heavier than early predictions.
3. **Baselines**:
   - Naive Constant Baseline (predict mean/last RUL).
   - Ridge Linear Regression.
   - Unsupervised Isolation Forest Anomaly Score trajectory (reference comparison).

---

## 10. Data Leakage Prevention Controls

1. **Strict Chronological Splitting**: Zero random K-fold splits.
2. **Train-Only Preprocessing**: `RobustScaler` fit strictly on train split (snapshots 0..600).
3. **Causal Rolling Features**: Rolling feature windows utilize only $t' \le t$. Zero centered windows.
4. **Target Isolation**: Target $RUL(t)$ used strictly as label during `.fit()`. No time/snapshot index used as input feature.
5. **Automated Leakage Unit Tests**:
   - `test_shuffle_future_invariance()`: Modifying $t+10$ snapshot does not alter features/predictions at $t$.
   - `test_scaler_leakage()`: Asserts test set statistics do not influence scaler parameters.

---

## 11. Persistence & Status Redesign

- **Quantity Monitored**: XGBoost Predicted RUL ($\widehat{RUL}$).
- **Threshold**: $\widehat{RUL} \le 100 \text{ snapshots}$ ($\sim 16.6 \text{ hours}$).
- **State Machine Vocabulary**:
  - `NORMAL`: $\widehat{RUL} > 100$ snapshots.
  - `WATCH`: $\widehat{RUL} \le 100$ snapshots for 1 or 2 consecutive snapshots.
  - `CRITICAL_RUL_ALERT`: $\widehat{RUL} \le 100$ snapshots for $k \ge 3$ consecutive snapshots.
- **State Auditability**: State transitions logged as events into InfluxDB `machine_state` measurement.

---

## 12. Grafana Migration Plan

- **Provisioning**: Environment-driven Grafana provisioning files (`grafana/provisioning/datasources/influxdb.yaml` and `grafana/provisioning/dashboards/machinemind.json`).
- **Dashboard Layout**:
  1. Equipment Overview & Status Banner (`NORMAL` / `WATCH` / `CRITICAL_RUL_ALERT`).
  2. Live Telemetry & RMS Trends (Channels 1–4).
  3. Predicted RUL & Degradation Trajectory (XGBoost vs Threshold).
  4. Spectral Feature Heatmaps / Channel Condition.
  5. Ingestion Pipeline Performance (MQTT msg/sec, processing lag).

---

## 13. Streamlit Migration Mapping

| Streamlit View / Page | Current Function | Target Grafana Panel | Action |
|---|---|---|---|
| `app.py` Header & KPI | Machine status & counts | State Banner & KPI stat panels | **MIGRATE TO GRAFANA** |
| `app.py` History Chart | Anomaly score line chart | Predicted RUL & Feature Trend Panel | **MIGRATE TO GRAFANA** |
| `1_Machines.py` | Equipment registration & list | Grafana Machine Dropdown Variable | **MIGRATE TO GRAFANA** |
| `2_Upload_Analyze.py` | Manual CSV file upload | Replay Simulator CLI / API | **MIGRATE TO REPLAY SIMULATOR** |
| `3_History.py` | Prediction history table | InfluxDB Table Panel | **MIGRATE TO GRAFANA** |
| `4_Alerts.py` | Alert list & status update | Grafana Alert / Incident History | **MIGRATE TO GRAFANA** |
| `5_Model_Info.py` | Model metadata display | Grafana Model Metadata Panel | **MIGRATE TO GRAFANA** |

---

## 14. Flask Migration Mapping

| Flask Endpoint | Current Path | Target Action | Replacement |
|---|---|---|---|
| `GET /api/health` | `backend/routes/health.py` | **RETAIN (OPTIONAL)** | Retain for ingestion service liveness check. |
| `POST /api/inference` | `backend/routes/inference.py` | **DEPRECATE / REMOVE** | Replaced by direct MQTT Ingestion Consumer. |
| `GET/POST /api/machines` | `backend/routes/machines.py` | **DEPRECATE / REMOVE** | Replaced by InfluxDB machine tag discovery in Grafana. |
| `GET /api/predictions` | `backend/routes/predictions.py` | **DEPRECATE / REMOVE** | Replaced by Flux queries in Grafana. |
| `GET/PATCH /api/alerts` | `backend/routes/alerts.py` | **DEPRECATE / REMOVE** | Replaced by InfluxDB alert state history. |

---

## 15. MongoDB Migration Mapping

- **Current Collections**: `machines`, `predictions`, `alerts`.
- **Target Strategy**: 
  - Migrate time-series predictions and alerts to InfluxDB measurements `model_predictions` and `machine_state`.
  - Backfill/regenerate historical dataset predictions deterministically from NASA IMS raw files into InfluxDB.
  - Remove MongoDB dependency entirely in Phase 9.

---

## 16. Isolation Forest / PCA Migration Decision

- **Decision**: Demote Isolation Forest and PCA from live decision path to **Offline Reference / Baseline Comparison Models**.
- **Live Path Model**: XGBoost RUL Regressor becomes the sole primary model on the live telemetry path.
- **Naming Rule**: Isolation Forest outputs remain strictly named `iforest_anomaly_score`. XGBoost outputs are strictly named `predicted_rul_snapshots` and `failure_risk_pct`.

---

## 17. Local Infrastructure Plan

All infrastructure will run locally via a single, reproducible `docker-compose.yml` (or local native service binaries if Docker is unavailable):
1. **MQTT Broker**: Eclipse Mosquitto (`localhost:1883`)
2. **Time-Series DB**: InfluxDB v2 (`localhost:8086`)
3. **Visualization**: Grafana (`localhost:3000`)

---

## 18. Risk Register

| Risk ID | Description | Severity | Mitigation Strategy |
|---|---|---|---|
| **R-01** | Single run-to-failure sequence limits generalizability | High | Clearly document dataset scope in Model Card and report. |
| **R-02** | Time-series data leakage in XGBoost features | High | Enforce strict chronological splits, embargo gaps, and automated unit tests. |
| **R-03** | High payload size for raw 20480x4 snapshot over MQTT | Medium | Package JSON payload efficiently; extract features in consumer. |
| **R-04** | Timestamp mismatch between 2004 dataset and live Grafana | Medium | Implement hybrid timestamp strategy (ingest time as point timestamp, original time as field). |
| **R-05** | InfluxDB tag cardinality explosion | Low | Use fixed tags (`machine_id`, `channel`, `model_version`). Keep numeric values in fields. |

---

## 19. Baseline Test & Application Execution Results

### 19.1 Verification Executed in Phase 0
- **Git Branch**: Created `feature/industrial-stack-migration` from `main` (commit `e1c4f18`).
- **Dataset Audit**: 
  - Verified 984 raw snapshot files in `ml-service/data/raw/IMS/2nd_test/2nd_test`.
  - Shape: 20,480 samples x 4 channels per snapshot.
  - Interval: Exactly 600.0s (10 min) between all 983 consecutive file pairs. Zero nulls.
- **Backend Test Suite**:
  - Command: `python -m pytest backend/tests`
  - Output: `16 passed in 5.05s`
- **Frontend Test Suite**:
  - Command: `python -m pytest frontend/tests`
  - Output: `11 passed in 2.04s`
- **Backend HTTP Liveness**:
  - `/api/health` -> HTTP 200 OK (`{"status": "ok", "database": "connected", "model": {"loaded": true, "version": "v1"}}`).

---

## 20. Proposed Phase 1 Implementation Plan

Upon approval of this Phase 0 report, **Phase 1** will execute as follows:

1. **Infrastructure**: Stand up local Mosquitto MQTT broker on `localhost:1883`.
2. **Payload Spec**: Formalize JSON payload schema module in `ml-service/src/telemetry_schema.py`.
3. **Replay Simulator**: Build `ml-service/src/simulator.py` to iterate through IMS Set 2 snapshots chronologically, publishing to `machinemind/v1/telemetry/ims_set2_rig` at configurable speed multiplier.
4. **Verification Subscriber**: Implement a standalone verification subscriber `ml-service/src/verify_mqtt_subscriber.py` to confirm message sequence, checksum, and zero leakage.
5. **Phase 1 Tests**:
   - `test_simulator_ordering()`
   - `test_payload_contract()`
   - `test_no_label_leakage()`
6. **Commit**: `feat(mqtt): add NASA IMS telemetry publisher and local broker setup`

---

### STOP & WAIT FOR APPROVAL
*Phase 0 Audit and Design Report is complete. No Phase 1 implementation code will be written until owner approval is received.*
