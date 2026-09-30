# BACKEND_PROMPT.md — MachineMind AI Flask Backend

> **Audience:** Antigravity IDE (coding agent).
> **Task:** Build the Flask REST API (application/service layer) for MachineMind AI.
> **Companion file:** `FRONTEND_PROMPT.md` (Streamlit dashboard). The frontend depends on the API contract defined here.

---

## 0. Read This First

MachineMind AI is an **unsupervised vibration anomaly-detection research/learning prototype** for rotating machinery, trained/evaluated on the NASA IMS bearing dataset.

The ML pipeline (Phases 1–11) is **complete and frozen**. Your job is ONLY the application layer around it:

```
Streamlit Frontend
      | HTTP REST
      v
Flask Backend  (this task)
  |            |
  v            v
MongoDB     Existing ML Inference Engine (ml-service/src/inference.py)
                 |
           IF + PCA model artifacts (ml-service/models/)
```

This is a time-constrained project. **A working, explainable MVP beats a complex one.** The code will be explained by the author in an interview/presentation, so keep it simple, readable, and commented where non-obvious.

---

## 1. Scientific Boundary (Non-Negotiable)

The system is an **unsupervised vibration anomaly detector**. It is NOT:

- a failure predictor, an RUL (remaining useful life) predictor,
- a supervised fault classifier,
- an industrially certified predictive-maintenance system.

Therefore, in every API field, message, log, alert text, and doc:

- **Use:** "Normal", "Vibration anomaly detected", "Potential abnormal vibration pattern detected", "Channel 2 anomaly score exceeded threshold".
- **Never use:** "Bearing failure detected/predicted", "Machine will fail", "Failure in N hours", "RUL", failure probabilities, accuracy percentages, or real-time latency claims.
- Do not invent metrics. Only expose values that the inference engine actually produces.

---

## 2. Mandatory Rules of Engagement

1. **Inspect the repository before modifying anything.**
2. **Read these files first** (in this order) and summarize what you learned before coding:
   - `ml-service/README.md`
   - `ml-service/ML_PROGRESS.md`
   - `ml-service/reports/model_card.md`
   - `ml-service/reports/integration_interface.md`
   - `ml-service/src/inference.py` (and `ml-service/src/models.py`, `ml-service/models/model_metadata.json`)
3. **Do NOT modify** (unless absolutely necessary — and if so, STOP and explain first, wait for approval):
   - trained artifacts in `ml-service/models/`
   - inference logic in `ml-service/src/`
   - raw NASA data in `ml-service/data/`
   - notebooks and Phase 1–11 reports
4. **Do NOT** retrain, retune, refit the scaler, change the 28 features, change thresholds (P99), change persistence (3 snapshots), or change OR aggregation.
5. **Do NOT duplicate ML logic.** No feature extraction, scaling, IF scoring, PCA, thresholding, or persistence re-implemented in the backend. Call the existing `AnomalyInferenceEngine`.
6. **No fake data presented as real.** If demo/synthetic data is ever needed, it must be clearly labeled `DEMO / SYNTHETIC` in filenames, docstrings, and any API field.
7. **No secrets in code.** Use environment variables and a committed `.env.example`.
8. **Work incrementally.** After each stage (Section 13): run tests, show the output, summarize what changed, then **STOP and wait for confirmation** before the next stage. Do not generate the whole app in one step.

> **Do not assume the inference engine's API.** The method names, input types, output keys, and how persistence state is held are defined by `inference.py` and `integration_interface.md`. Read them, then adapt the mapping layer (Section 6) to reality. If the real interface differs from this document, keep the **public API contract in Section 8 stable** and note the mapping in `backend/API_CONTRACT.md`.

---

## 3. Known ML Context (for orientation only — the code in `ml-service/` is authoritative)

- **Input:** one snapshot = `20480 × 4` numeric array (20480 samples, 4 vibration channels).
- **Preprocessing:** DC mean-centering → 28 time-domain features (7 per channel: mean, std, RMS, peak-to-peak, skewness, kurtosis, crest factor) → `RobustScaler` (fit on snapshots 0–159).
- **Models:** Isolation Forest (`n_estimators=100`, `max_samples="auto"`, `random_state=42`) and PCA reconstruction error (`n_components=3`, `random_state=42`).
- **Thresholds:** P99, calibrated on snapshots 160–179.
- **Decision logic:** anomaly must persist for **3 consecutive snapshots**; channels combined via **logical OR**.
- **Artifacts:** `iforest_pipeline_v1.joblib`, `pca_pipeline_v1.joblib`, `model_metadata.json`.

### Critical implication: persistence is stateful
A single snapshot can only produce a **snapshot-level anomaly flag**. The **persistence-confirmed** state needs ≥3 consecutive flagged snapshots **from the same machine, in order**. Investigate in Stage B:

1. How does `AnomalyInferenceEngine` hold persistence state (internal counters? history list? a stateless function taking prior flags)?
2. Is there one engine instance per machine, or can state be passed in/out?
3. Which output fields correspond to "raw snapshot flag" vs "persistence-confirmed"?

Recommended approach (adapt to what you find):
- Load the models **once at startup** (singleton service).
- Keep persistence state **per `machine_id`**.
- Persist each prediction's **per-channel raw flag and consecutive count** in MongoDB so state can be **rehydrated after a server restart** from the last N predictions (only if the engine supports state injection without modifying it).
- If the engine cannot support per-machine or restorable state without modifying `ml-service/src/`, **STOP and present options** (e.g., in-memory per-machine engine instances + documented limitation vs. a minimal wrapper class in `backend/services/` that does not alter ML logic). Do not silently reimplement persistence.

The UI must be able to distinguish **"snapshot flagged"** from **"anomaly confirmed by persistence"** — the API contract exposes both.

---

## 4. Project Structure

Create `backend/` at the repo root (sibling of `ml-service/`). Keep it flat and simple. Controllers are merged into route handlers (thin routes calling services) to save time.

```
backend/
├── app.py                  # app factory + entrypoint
├── config.py               # env-driven config
├── requirements.txt
├── .env.example
├── README.md               # how to run + test
├── API_CONTRACT.md         # final, verified endpoint contract (generated in Stage F)
├── routes/
│   ├── __init__.py
│   ├── health.py
│   ├── machines.py
│   ├── inference.py
│   ├── predictions.py
│   ├── alerts.py
│   └── data.py
├── services/
│   ├── ml_service.py       # ONLY place that imports/uses AnomalyInferenceEngine
│   ├── machine_service.py
│   ├── prediction_service.py
│   ├── alert_service.py
│   └── data_service.py     # CSV/JSON parsing + validation
├── models/                 # Mongo document schemas / serializers (plain Python)
│   ├── machine.py
│   ├── prediction.py
│   └── alert.py
├── utils/
│   ├── responses.py        # success()/error() helpers, envelope
│   ├── errors.py           # custom exceptions + global handlers
│   └── db.py               # MongoClient + index creation
└── tests/
    ├── conftest.py
    ├── test_health.py
    ├── test_validation.py
    ├── test_machines.py
    ├── test_inference.py
    ├── test_predictions_alerts.py
    └── fixtures/           # tiny helpers only; see Section 12
```

Stack: **Python 3.10+, Flask, flask-cors, pymongo, python-dotenv, numpy, pandas, pytest** (+ `mongomock` for fast tests). Add scikit-learn/joblib versions **matching `ml-service/requirements.txt`** (loading joblib artifacts with a different sklearn version is unsafe — check and align).

---

## 5. Configuration (Environment Variables)

`config.py` reads from environment (via `python-dotenv`). Provide `backend/.env.example` (no real secrets):

| Variable | Purpose | Example |
|---|---|---|
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017` |
| `DATABASE_NAME` | DB name | `machinemind` |
| `ML_SERVICE_PATH` | Path to `ml-service/` (added to `sys.path` so `src/inference.py` can be imported; the folder name has a hyphen, so it is not a normal package) | `../ml-service` |
| `MODEL_DIR` | Folder with model artifacts | `../ml-service/models` |
| `CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:8501` |
| `MAX_UPLOAD_MB` | Upload size limit | `10` |
| `FLASK_ENV` / `FLASK_DEBUG` | Dev toggles | `development` / `0` |
| `PORT` | Server port | `5000` |
| `API_KEY` | *Optional* simple shared-secret auth (leave empty to disable) | *(empty)* |

Never hardcode credentials. Fail with a clear error message if a required variable is missing.

---

## 6. ML Integration Layer (`services/ml_service.py`)

- This is the **only** module that touches `AnomalyInferenceEngine`.
- Add `ML_SERVICE_PATH` to `sys.path`, import the engine, load models **once** at app startup (lazy singleton is fine).
- Expose one function, e.g. `run_inference(machine_id, snapshot: np.ndarray) -> dict`, that:
  1. calls the existing engine with the validated `(20480, 4)` float array,
  2. maps engine output → the internal result dict used by the API (Section 8),
  3. returns per-channel scores, thresholds, raw flags, consecutive counts, confirmed states, overall state, and the model version (from engine/`model_metadata.json`).
- Do not expose internal implementation details (file paths, tracebacks, raw feature vectors) in API responses. The 28 scaled features may be omitted from responses.
- If models fail to load, the app should still start; `/api/health` reports `model.loaded = false`, and `/api/inference` returns `503` with a clear message.

---

## 7. Validation Rules (`services/data_service.py`)

Validate **before** calling the ML engine. Return `400`/`413`/`415`/`422` with a clean error.

**File upload (multipart):**
- Field name `file`; extension must be `.csv` (also accept `.txt`/`.tsv` — NASA IMS snapshot files are whitespace/tab-delimited); reject others with `415`.
- Enforce `MAX_CONTENT_LENGTH` (→ `413`).
- Auto-detect the delimiter (comma / tab / whitespace). Tolerate an optional single header row.
- No `eval`, no pickle, no path handling from user-supplied filenames (sanitize with `secure_filename`; never write uploads to disk unless required, and never to a user-controlled path).

**Data shape/content:**
- Exactly **4 columns** and exactly **20480 rows** (report actual vs expected in `details`).
- Every value numeric and **finite** (reject NaN/Inf/non-numeric; report count and first offending location).
- Convert to `float64` `np.ndarray` of shape `(20480, 4)`.

**Request fields:**
- `machine_id` required, must match `^[A-Za-z0-9_-]{1,64}$`, and must refer to an existing machine (`404` otherwise).
- `snapshot_time` optional ISO-8601 string (e.g., timestamp of the NASA IMS file); if absent, use server UTC time. Reject unparseable values with `422`.

**Error codes (`error.code`):** `VALIDATION_ERROR`, `INVALID_SHAPE`, `INVALID_VALUES`, `UNSUPPORTED_FILE_TYPE`, `PAYLOAD_TOO_LARGE`, `MACHINE_NOT_FOUND`, `MACHINE_EXISTS`, `MODEL_UNAVAILABLE`, `DATABASE_UNAVAILABLE`, `NOT_FOUND`, `INTERNAL_ERROR`.

---

## 8. API Contract (Target — keep it stable)

### 8.1 Response envelope (ALL endpoints)

Success:
```json
{ "success": true, "data": { }, "error": null }
```
Error:
```json
{ "success": false, "data": null,
  "error": { "code": "INVALID_SHAPE", "message": "Expected 20480 rows x 4 columns, got 1000 x 4.", "details": { "expected_rows": 20480, "actual_rows": 1000 } } }
```
Register global error handlers so **no HTML error pages and no stack traces** ever leak; log the traceback server-side only. Timestamps: ISO-8601 UTC strings (`2026-01-31T12:00:00Z`). Never return raw Mongo `_id`/`ObjectId` — serialize to string ids or omit.

### 8.2 Endpoints

| Method | Path | Purpose | Success |
|---|---|---|---|
| GET | `/api/health` | Service, DB, model status | 200 |
| GET | `/api/machines` | List machines (+ latest prediction summary) | 200 |
| GET | `/api/machines/<machine_id>` | One machine (+ latest prediction, open alert count) | 200 |
| POST | `/api/machines` | Create machine | 201 |
| POST | `/api/inference` | Validate → infer → store → respond | 201 |
| GET | `/api/predictions/<machine_id>` | Prediction history (`?limit=50&offset=0`, newest first, max limit 200) | 200 |
| GET | `/api/alerts` | All alerts (`?status=&severity=&limit=`) | 200 |
| GET | `/api/alerts/<machine_id>` | Alerts for one machine | 200 |
| PATCH | `/api/alerts/<alert_id>` | *(Optional, low priority)* set status `acknowledged`/`resolved` | 200 |
| POST | `/api/data/upload` | *(Optional, low priority)* validate-only: parse + validate a file, return shape/summary, **no inference, no storage** | 200 |

Unknown machine → `404 MACHINE_NOT_FOUND`. Duplicate `machine_id` → `409 MACHINE_EXISTS`.

### 8.3 `GET /api/health`
```json
{ "success": true, "data": {
  "service": "machinemind-backend",
  "status": "ok",                      // "ok" | "degraded"
  "database": "connected",             // "connected" | "unavailable"
  "model": { "loaded": true, "version": "v1" },
  "time": "2026-01-31T12:00:00Z" }, "error": null }
```
Return 200 whenever the process is up; use `"degraded"` if DB or model is unavailable.

### 8.4 `POST /api/machines`
Request: `{ "machine_id": "bearing_set2_ch1", "name": "Test Rig A", "description": "NASA IMS Set 2 replay" }` (`description` optional).
Response `201`: machine object:
```json
{ "machine_id": "...", "name": "...", "description": "...", "status": "no_data", "created_at": "..." }
```
**Machine `status` values:**
- `no_data` — no predictions yet
- `normal` — latest snapshot not flagged
- `watch` — latest snapshot flagged, persistence not yet satisfied
- `anomaly_detected` — persistence-confirmed anomaly (≥3 consecutive flagged snapshots)

(These four states are an application-layer convention derived from the engine's outputs; document them as such.)

### 8.5 `POST /api/inference`
Two accepted input modes:
1. **multipart/form-data** (primary; used by Streamlit): `file` (CSV) + `machine_id` (+ optional `snapshot_time`).
2. **application/json**: `{ "machine_id": "...", "snapshot_time": "...", "data": [[c1,c2,c3,c4], ... 20480 rows ...] }`.

Response `201`:
```json
{ "success": true, "data": {
  "prediction_id": "...",
  "machine_id": "bearing_set2_ch1",
  "timestamp": "2026-01-31T12:00:00Z",
  "model_version": "v1",
  "source_filename": "2004.02.12.10.32.39",
  "overall": {
    "state": "normal",                        // "normal" | "watch" | "anomaly_detected"
    "snapshot_flagged": false,                // any channel flagged in this snapshot (before persistence)
    "persistence_confirmed": false,           // any channel confirmed by the 3-snapshot rule
    "label": "Normal"                         // human-readable, follows Section 1 wording
  },
  "channels": [
    { "channel": 1, "name": "Channel 1",
      "isolation_forest_score": 0.0,          // map real engine outputs; names may be adapted
      "pca_reconstruction_error": 0.0,
      "isolation_forest_threshold": 0.0,
      "pca_threshold": 0.0,
      "snapshot_flagged": false,
      "consecutive_flagged_count": 0,
      "persistence_confirmed": false,
      "state": "normal" }                     // "normal" | "watch" | "anomaly_detected"
    // ... channels 2, 3, 4
  ],
  "persistence": { "required_consecutive_snapshots": 3 },
  "alert": { "created": false, "alert_id": null, "severity": null }
}, "error": null }
```
**Adapt the score/threshold field names to what the engine truly returns** (e.g., if the pipeline yields one combined score per channel, expose that instead and remove fields that don't exist). Never fabricate a field to fit this template. Record the final shape in `API_CONTRACT.md`.

### 8.6 `GET /api/predictions/<machine_id>`
```json
{ "data": { "machine_id": "...", "count": 2, "total": 37,
  "predictions": [ { "prediction_id": "...", "timestamp": "...", "model_version": "v1",
                     "overall": { ... same as above ... }, "channels": [ ... same as above ... ] } ] } }
```
Newest first. This feeds the frontend's history charts, so include per-channel scores and flags in every item.

### 8.7 Alerts
Alert object:
```json
{ "alert_id": "...", "machine_id": "...", "timestamp": "...",
  "alert_type": "persistent_vibration_anomaly",
  "severity": "warning",                       // "warning" | "high"
  "affected_channels": [2],
  "message": "Potential abnormal vibration pattern: Channel 2 anomaly score exceeded threshold for 3 consecutive snapshots.",
  "status": "open" }                           // "open" | "acknowledged" | "resolved"
```
Response: `{ "data": { "count": N, "alerts": [ ... ] } }`, newest first.

**Alert rules (keep simple, document as application-layer convention):**
- Create an alert **only when a channel transitions into persistence-confirmed anomaly** (not on every subsequent flagged snapshot — avoid duplicates; if an `open` alert for the same machine already covers the same channels, don't create another).
- Severity: `warning` = 1 affected channel; `high` = 2+ affected channels. This is a UI-level heuristic, **not** a physical severity or failure-risk claim.
- Snapshot-level flags that haven't met persistence do **not** create alerts; they appear in prediction history and as machine status `watch`.
- Messages must use the wording in Section 1.

### 8.8 `POST /api/data/upload` (optional)
Validate-only. Returns `{ "filename", "rows", "columns", "delimiter", "has_header", "valid": true, "channel_summary": [{"channel":1,"min":..,"max":..,"mean":..}, ...] }` or a validation error. Implement last, only if time permits.

---

## 9. MongoDB Design

Use `pymongo`. Database name from `DATABASE_NAME`. Collections and documents:

**`machines`**
`machine_id` (unique), `name`, `description`, `status`, `created_at`, `updated_at`

**`predictions`**
`machine_id`, `timestamp` (snapshot time), `created_at` (server time), `model_version`, `source_filename`, per-channel: `anomaly scores`, `thresholds`, `snapshot flag`, `consecutive_flagged_count`, `persistence_confirmed`, `state`; plus `overall_state`, `snapshot_flagged`, `persistence_confirmed`.

**`alerts`**
`machine_id`, `timestamp`, `alert_type`, `severity`, `affected_channels`, `message`, `status`, `prediction_id` (reference), `created_at`

**Rules:**
- **Do NOT store the raw 20480×4 array.** Store only results/metadata (a checksum of the upload is optional).
- Indexes (created at startup in `utils/db.py`): `machines.machine_id` unique; `predictions (machine_id, timestamp desc)`; `alerts (machine_id, timestamp desc)`, `alerts.status`.
- The app must start even if MongoDB is temporarily down; DB-dependent routes return `503 DATABASE_UNAVAILABLE`.
- Use a short `serverSelectionTimeoutMS` (e.g., 3000) so failures surface fast.

---

## 10. Security & Hardening (basic, quick)

- Request validation on every endpoint (Section 7).
- `MAX_CONTENT_LENGTH` from `MAX_UPLOAD_MB`; extension allow-list; `secure_filename`.
- CORS restricted to `CORS_ORIGINS` (no wildcard in the default config).
- Global JSON error handlers (400/404/405/413/415/422/500/503).
- Parameter bounds on pagination (`limit` ≤ 200; integers only).
- Never build Mongo queries directly from raw request dicts (avoid operator injection — validate types; treat `machine_id` as a plain string).
- Logging: log request path/status and server-side exceptions; **do not log vibration data or secrets**.
- **Authentication is optional.** If time allows, add an optional `X-API-Key` check enabled only when `API_KEY` is set. **Do not let this delay the core path.**

---

## 11. Code Quality Guidelines

- App-factory pattern (`create_app()`), blueprints per route file, thin routes, logic in services.
- Type hints and short docstrings; comments explaining *why*, not *what*.
- No dead code, no speculative abstractions, no unneeded dependencies.
- Pin versions in `requirements.txt`.

---

## 12. Testing Requirements

Use `pytest`. Use `mongomock` (or a dedicated test database) so tests don't touch real data.

**Must-have tests:**
1. `/api/health` returns the envelope and correct fields.
2. Machine create/list/get, duplicate → 409, invalid id → 422/400, unknown → 404.
3. Validation: wrong column count, wrong row count, non-numeric, NaN/Inf, empty file, wrong extension, oversize file, missing `machine_id`, unknown machine, malformed JSON.
4. Inference **with a real NASA IMS snapshot** (see below) returns 201 and the contract shape.
5. Prediction saved to Mongo and returned by `GET /api/predictions/<machine_id>` (ordering and pagination).
6. Persistence flow: submit a sequence of snapshots for one machine and confirm counters and the confirmed state behave as the engine defines; confirm an alert is created **once** on transition.
7. The stored prediction document does **not** contain the raw array.
8. Model-unavailable and DB-unavailable paths return the right error codes (mock/patch).

**Test data:**
- For real inference tests, locate a real snapshot from `ml-service/data/` (read-only; **do not modify or copy raw data into the repo** unless it's small and the license allows — otherwise reference it by path/env var like `TEST_SNAPSHOT_PATH` and `pytest.skip` if absent).
- Validation-only tests may use small synthetic arrays; label helpers/fixtures clearly as **DEMO / SYNTHETIC**.
- Never present synthetic data as a real sensor recording.

---

## 13. Implementation Stages (Follow in Order — STOP after each)

For each stage: implement → run tests → show output → summarize changes → **wait for user confirmation**.

**PHASE A — Minimal Flask backend.**
Skeleton, config, envelope helpers, error handlers, `/api/health`, CORS, `requirements.txt`, `.env.example`. Test health.

**PHASE B — Connect to existing ML engine.**
Implement `ml_service.py` (load once, map output). Report exactly how the engine's interface and persistence state work and **which design you chose for per-machine state** (Section 3). Unit-test the mapping. Do not touch `ml-service/`.

**PHASE C — Test `/api/inference` with a real NASA IMS snapshot.**
Implement `data_service.py` validation + `/api/inference` (without DB yet; machine lookup may be stubbed/in-memory temporarily). Run against a real snapshot and show the JSON. Include validation-failure tests.

**PHASE D — MongoDB.**
Connect Mongo, indexes, machines CRUD, store predictions, create alerts on confirmed transitions, machine status updates, rehydrate persistence state (if supported). Add the persistence/alert tests.

**PHASE E — History & alerts endpoints.**
`GET /api/predictions/<machine_id>`, `GET /api/alerts`, `GET /api/alerts/<machine_id>`, pagination/filter tests.

**PHASE F — Contract freeze & docs.**
Generate `backend/API_CONTRACT.md` with the **actual verified** request/response examples (captured from real runs). Write `backend/README.md` (setup, env vars, run, test, curl examples). The frontend agent will treat `API_CONTRACT.md` as the source of truth. Optionally: alert `PATCH`, `/api/data/upload`, optional API key.

**PHASE G — Final backend integration test.**
End-to-end with a real Mongo instance: create machine → upload several consecutive real snapshots (as a demonstration of the persistence rule) → check predictions, alerts, machine status. Report any deviation from the contract.

> Streamlit work (Phases E–H of the overall project) begins only after backend Stage F is confirmed. See `FRONTEND_PROMPT.md`.

---

## 14. End-to-End Flow the Backend Must Support

```
CSV upload
  ↓
Streamlit
  ↓
POST /api/inference (multipart: file + machine_id [+ snapshot_time])
  ↓
Flask: validate file, shape (20480×4), numeric/finite, machine exists
  ↓
ml_service → AnomalyInferenceEngine (existing, unmodified)
  ↓
Isolation Forest + PCA reconstruction error (P99 thresholds, 3-snapshot persistence, OR across channels)
  ↓
result → MongoDB (prediction; alert if newly confirmed; machine status update)
  ↓
Flask JSON envelope response
  ↓
Streamlit visualization
```

---

## 15. Definition of Done

- [ ] All Section 8 core endpoints (health, machines, inference, predictions, alerts) work against real MongoDB.
- [ ] A real NASA IMS snapshot posted to `/api/inference` returns a contract-conformant result produced by the **unmodified** `AnomalyInferenceEngine`.
- [ ] Persistence-confirmed vs snapshot-flagged states are distinguishable in responses.
- [ ] Alerts are created once per confirmed transition, using approved wording.
- [ ] No raw vibration arrays in MongoDB; no secrets in code; `.env.example` present.
- [ ] `pytest` passes; `API_CONTRACT.md` and `README.md` exist and reflect verified behavior.
- [ ] Nothing under `ml-service/` was modified (verify with `git status`/`git diff`).
- [ ] No unsupported claims (accuracy %, failure prediction, RUL, latency) appear anywhere.
