# FRONTEND_PROMPT.md — MachineMind AI Streamlit Dashboard

> **Audience:** Antigravity IDE (coding agent).
> **Task:** Build the Streamlit dashboard for MachineMind AI. It talks ONLY to the Flask backend over HTTP.
> **Companion file:** `BACKEND_PROMPT.md`. Once the backend exists, **`backend/API_CONTRACT.md` is the source of truth** for request/response shapes. If it conflicts with this document, follow `API_CONTRACT.md` and tell the user.

---

## 0. Read This First

MachineMind AI is an **unsupervised vibration anomaly-detection research/learning prototype** for rotating machinery (NASA IMS bearing dataset). The ML pipeline is complete and frozen; the Flask backend wraps it. Your job is the **presentation layer only**.

```
Streamlit Frontend (this task)
      | HTTP REST (requests)
      v
Flask Backend  →  MongoDB
               →  Existing ML Inference Engine
```

Time is very limited. Build a **working, explainable, clean MVP**. No excessive animation, no heavy custom CSS. The author must be able to explain every file in an interview/presentation.

---

## 1. Scientific Boundary & UI Language (Non-Negotiable)

The system is an **unsupervised vibration anomaly detector**. It is NOT a failure predictor, RUL predictor, supervised fault classifier, or certified industrial system.

**Use this wording:**
- "Normal"
- "Vibration anomaly detected"
- "Potential abnormal vibration pattern"
- "Channel 2 anomaly score exceeded threshold"
- "Anomalous snapshot — awaiting persistence confirmation"

**Never show or write:** "Bearing failure detected/predicted", "Machine will fail", "100% accurate" or any accuracy %, "Failure in N hours", "RUL", failure probabilities, real-time latency claims.

Do not compute or display any metric the backend did not return. Add a small persistent footer/caption: *"Research prototype. Detects statistical vibration anomalies; it does not confirm physical faults or predict failure."*

---

## 2. Mandatory Rules of Engagement

1. **Inspect the repository first.** Read `README.md`/`ML_PROGRESS.md`/`reports/model_card.md` in `ml-service/` for context (wording, limitations), and `backend/API_CONTRACT.md` + `backend/README.md` for the API.
2. The frontend **must NOT**:
   - connect to MongoDB,
   - load joblib models or import anything from `ml-service/`,
   - perform ML inference, feature extraction, thresholding, or persistence logic,
   - contain training code.
3. **Do NOT modify** `ml-service/`, `backend/` (except to report a contract bug to the user), model artifacts, raw NASA data, notebooks, or Phase 1–11 reports.
4. **No fake data presented as real.** If demo/synthetic data is used for UI testing, it must be clearly labeled `DEMO / SYNTHETIC` in the UI (visible banner) and code. Prefer testing with real NASA IMS snapshots.
5. No secrets in code. Backend URL comes from an environment variable.
6. **Work incrementally.** After each stage (Section 12): run/test, show what works (describe or screenshot-equivalent output), summarize changes, then **STOP and wait for confirmation**. Don't generate the whole app in one step.
7. Keep code simple: small functions, clear names, brief comments explaining *why*.

---

## 3. Project Structure

Create `frontend/` at the repo root (sibling of `backend/` and `ml-service/`).

```
frontend/
├── app.py                      # Overview / dashboard (Streamlit entrypoint)
├── pages/
│   ├── 1_Machines.py           # machine list, selection, create machine
│   ├── 2_Upload_Analyze.py     # CSV upload → inference → results
│   ├── 3_History.py            # prediction history charts/table
│   ├── 4_Alerts.py             # alert table + filters
│   └── 5_Model_Info.py         # explanation + limitations
├── components/
│   ├── status.py               # status badges/labels/colors (single source of wording)
│   ├── kpi.py                  # KPI card helpers
│   ├── channel_cards.py        # 4-channel analysis cards
│   └── charts.py               # chart builders (score bars, waveform, history)
├── services/
│   └── api_client.py           # ALL HTTP calls to Flask (only place that uses requests)
├── utils/
│   ├── config.py               # BACKEND_URL, timeouts
│   ├── csv_validation.py       # client-side CSV pre-check (shape/numeric)
│   └── formatting.py           # timestamp/number formatting
├── requirements.txt            # streamlit, requests, pandas, numpy, altair (pin versions)
├── .env.example                # BACKEND_URL=http://localhost:5000
├── README.md                   # how to run
└── tests/
    └── test_api_client.py      # mocked-HTTP unit tests (see Section 11)
```

Streamlit's `pages/` directory gives sidebar navigation automatically. Use `st.set_page_config(page_title="MachineMind AI", layout="wide")` on every page (first Streamlit call).

---

## 4. Configuration

- `BACKEND_URL` env var (default `http://localhost:5000`), read in `utils/config.py`. Provide `frontend/.env.example`.
- Request timeouts: ~10 s for reads, ~60 s for `/api/inference`.
- Optional: if the backend has `API_KEY` enabled, send an `X-API-Key` header from an env var `API_KEY` (only if set).

---

## 5. API Client (`services/api_client.py`)

The only module that performs HTTP. Requirements:

- One function per endpoint, returning parsed `data` on success or raising a small `ApiError(code, message, details)`.
- Handle all failure modes gracefully: connection refused/timeouts (→ "Backend unreachable at {BACKEND_URL}"), non-JSON responses, and the standard envelope `{success, data, error}`.
- Pages catch `ApiError` and show `st.error(...)` with a human-readable message and (in an expander) the `details`. **Never** show raw stack traces.

Functions (paths per BACKEND_PROMPT.md §8):

| Function | Endpoint |
|---|---|
| `get_health()` | `GET /api/health` |
| `list_machines()` | `GET /api/machines` |
| `get_machine(machine_id)` | `GET /api/machines/<id>` |
| `create_machine(machine_id, name, description)` | `POST /api/machines` |
| `run_inference(machine_id, file_bytes, filename, snapshot_time=None)` | `POST /api/inference` (multipart: `file`, `machine_id`, optional `snapshot_time`) |
| `get_predictions(machine_id, limit, offset)` | `GET /api/predictions/<id>` |
| `list_alerts(status=None, severity=None, limit=100)` | `GET /api/alerts` |
| `get_machine_alerts(machine_id)` | `GET /api/alerts/<id>` |
| `update_alert_status(alert_id, status)` | `PATCH /api/alerts/<id>` *(only if backend implements it)* |

Use `st.cache_data(ttl=5–10)` for read calls (health, machines, predictions, alerts) so navigation is snappy; **never cache `run_inference`**. After a successful inference, clear the relevant caches (`st.cache_data.clear()`) so dashboards refresh.

---

## 6. Expected Data Shapes (verify against `backend/API_CONTRACT.md`)

**Envelope:** `{ "success": bool, "data": {...}|null, "error": {"code","message","details"}|null }`

**Machine:** `machine_id, name, description, status, created_at, latest_prediction?`
Machine `status`: `no_data` | `normal` | `watch` | `anomaly_detected`.

**Inference / prediction item:**
```
prediction_id, machine_id, timestamp, model_version, source_filename,
overall: { state: normal|watch|anomaly_detected, snapshot_flagged, persistence_confirmed, label },
channels: [ { channel (1-4), name, <scores>, <thresholds>, snapshot_flagged,
              consecutive_flagged_count, persistence_confirmed, state } x4 ],
persistence: { required_consecutive_snapshots: 3 },
alert: { created, alert_id, severity }
```
Score/threshold field names (e.g., `isolation_forest_score`, `pca_reconstruction_error`, `*_threshold`) **may differ** in the final contract — read them from `API_CONTRACT.md`, and **centralize field names as constants** in one place (e.g., top of `components/channel_cards.py` or a `utils/fields.py`) so they're easy to change.

**Alert:** `alert_id, machine_id, timestamp, alert_type, severity (warning|high), affected_channels[], message, status (open|acknowledged|resolved)`

### Key concept the UI must communicate
- **Snapshot flagged** = this single snapshot exceeded a threshold on at least one channel.
- **Anomaly confirmed (persistence)** = the flag persisted for **3 consecutive snapshots** on a channel. Only then does the system report "Vibration anomaly detected" and create an alert.

A single uploaded file is **one snapshot**, so it will usually show "Normal" or "Anomalous snapshot — awaiting persistence confirmation". Show a `st.info` explaining this, plus "N of 3 consecutive flagged snapshots" per channel. Don't hide this behind confusing wording.

---

## 7. Status Wording & Colors (`components/status.py`)

Single source of truth for labels, used everywhere:

| State | Label | Indicator |
|---|---|---|
| `no_data` | "No data yet" | ⚪ gray |
| `normal` | "Normal" | 🟢 green |
| `watch` | "Anomalous snapshot — awaiting persistence confirmation" (short: "Watch") | 🟡 amber |
| `anomaly_detected` | "Vibration anomaly detected" | 🔴 red |

Alert severity: `warning` 🟠, `high` 🔴. Prefer emoji/`st.success/warning/error` and `st.metric` over custom CSS. Each alert/status display should never imply physical failure.

---

## 8. Pages & Features

### 8.1 Overview (`app.py`)
- Title, one-line description, the research-prototype disclaimer.
- **System status:** from `GET /api/health` → backend reachable, database, model loaded/version (badge).
- **KPI row (`st.metric`):** number of machines; machines in `normal` / `watch` / `anomaly_detected`; open alerts.
- **Current machine state:** table of machines with name, ID, status badge, latest prediction time.
- **Latest anomaly state:** for the selected machine (or most recent overall), show overall label + timestamp.
- **Recent alerts:** last ~5 alerts (timestamp, machine, channels, severity, message, status).
- If the backend is unreachable: a single clear error banner with the URL and hint to start the backend, then `st.stop()`.

### 8.2 Machines (`pages/1_Machines.py`)
- List machines; **select a machine** via `st.selectbox`; store `selected_machine_id` in `st.session_state` and ALSO show the selection in the sidebar on all pages (shared helper).
- Show for the selected machine: name, machine ID, description, status badge, latest prediction summary (time, overall label, per-channel state).
- **Create machine** form (`st.form`: machine_id, name, description) → `POST /api/machines`; show backend validation errors (e.g., duplicate ID, invalid ID format).
- Empty state: guide the user to create a machine first.

### 8.3 Upload & Analyze (`pages/2_Upload_Analyze.py`) — CORE FEATURE
Flow:
1. Require a selected machine (else prompt to select/create).
2. `st.file_uploader` for CSV/TXT/TSV (single file). Optional text input for `snapshot_time` (ISO-8601; helper text: "Optional — e.g., time of the NASA IMS file. Defaults to server time.").
3. **Client-side pre-validation** (`utils/csv_validation.py`, pandas): auto-detect delimiter (comma/tab/whitespace), tolerate one optional header row, check **exactly 20480 rows × 4 columns**, all numeric and finite. Show clear errors, e.g. *"Expected 20480 rows × 4 columns; your file has 1000 × 4."* / *"Non-numeric values found in column 3 (row 512)."* If valid: `st.success` + shape + small preview (`st.dataframe(head)`) + per-channel min/max/mean. This is UX only — the backend re-validates and is authoritative.
4. Waveform preview (see 8.5) of the uploaded data **even before** running analysis (it's the user's own file; no inference here).
5. **"Run analysis" button** → `api_client.run_inference(...)` with a `st.spinner`. Do not run inference in Streamlit. Disable the button if pre-validation failed.
6. Display results from the response: overall status banner, 4 channel cards (8.4), score chart (8.5), and if `alert.created` show an alert notice with severity + message.
7. Store the last result in `st.session_state` so it survives reruns.
8. Surface backend errors (`INVALID_SHAPE`, `MACHINE_NOT_FOUND`, `MODEL_UNAVAILABLE`, `DATABASE_UNAVAILABLE`, …) with friendly text.

**Optional stretch (only if time remains): "Sequence replay"** — allow multiple files (`accept_multiple_files=True`) sorted by filename (NASA IMS filenames are chronological), sent **one by one in order** to `/api/inference`, with a progress bar, to demonstrate the 3-snapshot persistence rule. Show a compact per-file result table. Real files only; do not simulate.

### 8.4 Channel Analysis (`components/channel_cards.py`)
Render four columns (Channel 1–4). For each channel show:
- Anomaly score(s) and threshold(s) exactly as returned (use `st.metric`; show delta vs threshold if it helps readability, e.g., "score / threshold").
- State badge (normal / watch / anomaly_detected).
- "Consecutive flagged snapshots: n / 3".
- A plain-language line, e.g., "Channel 2 anomaly score exceeded threshold." only when actually flagged.

### 8.5 Visualizations (`components/charts.py`) — keep to these four
Use Streamlit-native charts or **Altair** (bundled with Streamlit). Keep them readable.
1. **Anomaly score by channel** — grouped bar chart of score vs threshold per channel (threshold as a rule/line or second series). One chart per score type if the backend returns two (IF score and PCA error), in tabs.
2. **Vibration waveform** — the uploaded file's 4 channels, **downsampled** for performance (≤ ~2000 points per channel; simple stride or min/max decimation), channel selector or `st.tabs`. Label as "Uploaded snapshot (downsampled for display)". The x-axis is sample index unless a sampling rate is documented in `model_card.md` — do not invent time units.
3. **Prediction history** — from `GET /api/predictions/<id>`: line chart of per-channel scores over time, with threshold reference where practical, and markers/table for flagged/confirmed points. Limit to last 50–100 points.
4. **Alert history** — table (and optionally a small count-over-time chart) from alerts endpoints.

### 8.6 History (`pages/3_History.py`)
Selected machine → prediction history chart(s) + table (timestamp, overall label, per-channel states/flags, model version). Simple controls: number of points (`st.slider` 10–200). Empty state when no predictions.

### 8.7 Alerts (`pages/4_Alerts.py`)
- Table with: timestamp, machine, affected channel(s), severity, message, status.
- Filters: machine (all/selected), status, severity.
- Summary counts (open / acknowledged / resolved).
- Optional: acknowledge button per alert **only if** the backend `PATCH` endpoint exists.
- Empty state: "No alerts. Alerts are created only when an anomaly persists for 3 consecutive snapshots."

### 8.8 Model Information (`pages/5_Model_Info.py`)
Short, honest explanation (use `st.expander`s). Content must be consistent with `ml-service/reports/model_card.md`:
- **Purpose:** unsupervised vibration anomaly detection (not failure prediction / RUL / fault classification).
- **Input:** one snapshot = 20480 samples × 4 channels.
- **28 time-domain features:** 7 per channel — mean, standard deviation, RMS, peak-to-peak, skewness, kurtosis, crest factor (after DC mean-centering, then RobustScaler).
- **Isolation Forest:** isolates unusual feature patterns; lower "normality" / higher anomaly score means more unusual (describe according to the actual score convention in the model card).
- **PCA reconstruction error:** how poorly the learned normal-behavior subspace reconstructs a snapshot's features.
- **Thresholds:** P99 of a calibration window of healthy-period snapshots.
- **Persistence rule:** anomaly must persist for 3 consecutive snapshots; channels combined with logical OR.
- **Limitations:** research dataset (NASA IMS); statistical anomalies ≠ confirmed physical faults; not an industrially certified system. Pull limitation wording from `model_card.md` — do not invent.
- Show `model_version` and backend health info if available. **No accuracy numbers unless they appear verbatim in the existing final reports** (if included, cite the report name; otherwise omit).

---

## 9. UX Guidelines

- Sidebar: app name, selected machine, backend status dot, link-style navigation (auto from `pages/`).
- `st.columns` for KPI cards and channel cards; `st.tabs`/`st.expander` for technical details.
- Consistent status badges from `components/status.py`.
- Friendly empty states and loading spinners; never a blank page or a traceback.
- Format timestamps consistently (`utils/formatting.py`), show UTC explicitly.
- Keep custom CSS to a minimum (a few lines at most, or none).
- Don't expose internal IDs unnecessarily; show `machine_id`, `alert_id` only where useful.

---

## 10. Data Flow the Frontend Must Implement

```
User selects machine
  ↓
Uploads CSV (Streamlit file_uploader)
  ↓
Client-side pre-check: 20480 × 4, numeric  (UX only)
  ↓
POST /api/inference  (multipart: file + machine_id [+ snapshot_time])   ← via api_client
  ↓
Flask validates → AnomalyInferenceEngine → IF + PCA → MongoDB → JSON
  ↓
Streamlit renders: overall status, 4 channel cards, score chart,
                   alert notice, updated history/alerts on other pages
```

---

## 11. Testing Requirements (keep light but real)

- `tests/test_api_client.py` (pytest + `unittest.mock`/`responses`): success envelope parsing, error envelope → `ApiError`, connection error → friendly `ApiError`, non-JSON response handling.
- Unit tests for `utils/csv_validation.py`: valid file, wrong rows, wrong columns, non-numeric, NaN, header row, tab-delimited. Synthetic arrays here are fine (label as `DEMO / SYNTHETIC` in test helpers).
- Optionally use `streamlit.testing.v1.AppTest` for a smoke test that pages load with a mocked API client.
- **Manual end-to-end check** (Stage H) with the real backend and a real NASA IMS snapshot; report results.

---

## 12. Implementation Stages (Follow in Order — STOP after each)

The backend must be at least at **its Stage F** (contract frozen, `API_CONTRACT.md` exists) before frontend Stage 2 begins. If the backend isn't ready, you may do Stage 1 only.

For each stage: implement → test/run → summarize → **wait for confirmation**.

**PHASE E — Minimal Streamlit dashboard.**
Project skeleton, `config.py`, `requirements.txt`, `.env.example`, `status.py`, `app.py` showing title, disclaimer, and an empty/placeholder overview. App starts cleanly.

**PHASE F — Connect Streamlit to Flask.**
Implement `api_client.py` (all read functions + `run_inference`), health indicator, machines list/selection/creation, KPI cards, and the **Upload & Analyze page** with client-side validation and the full round trip to `POST /api/inference`. Show overall status + 4 channel cards. Verify with a real snapshot. *This completes the core end-to-end path.*

**PHASE G — Charts, history, alerts.**
Score chart, waveform preview, History page, Alerts page, recent alerts on Overview, Model Info page, empty states, error polish. (Optional: sequence replay, alert acknowledge.)

**PHASE H — Final integration testing.**
Full run: start MongoDB → backend → frontend; create machine; analyze real snapshots (including several consecutive real files to exhibit the persistence rule); verify status/alerts/history update; test error paths (wrong shape, bad file type, backend down). Run frontend tests. Review all wording against Section 1. Update `frontend/README.md` with run instructions and a short "demo script" for the presentation.

---

## 13. Definition of Done

- [ ] `streamlit run app.py` (from `frontend/`) works with `BACKEND_URL` set; sidebar navigation shows all pages.
- [ ] A real NASA IMS snapshot uploaded on the Upload & Analyze page yields: validated shape, POST to `/api/inference`, overall status, four channel cards with scores/thresholds/states, and a score chart.
- [ ] History, Alerts, Overview KPIs and Model Info pages work with real backend data and have empty states.
- [ ] Snapshot-flagged vs persistence-confirmed is clearly explained and visible.
- [ ] Frontend contains **no** direct MongoDB access, model loading, or ML logic (verify by searching for `pymongo`, `joblib`, `sklearn`, `ml-service` imports).
- [ ] All UI text follows Section 1; no accuracy %, failure prediction, RUL, or latency claims.
- [ ] Any synthetic/demo data is visibly labeled `DEMO / SYNTHETIC`.
- [ ] Nothing under `ml-service/` or `backend/` was modified.
- [ ] Tests pass; `frontend/README.md` documents setup and a demo walkthrough.
