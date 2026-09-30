# MachineMind AI — Streamlit Frontend Presentation Layer

An unsupervised vibration anomaly detection dashboard for rotating machinery built with Streamlit. Communicates exclusively via HTTP REST API calls with the Flask backend.

---

## 1. Scientific & Operational Boundary

MachineMind AI is an **unsupervised vibration anomaly-detection research/learning prototype** built on the NASA IMS Bearing dataset.

* **Approved Terminology:** `"Normal"`, `"Vibration anomaly detected"`, `"Anomalous snapshot — awaiting persistence confirmation"`, `"Watch"`.
* **Explicit Exclusions:** This system does **not** predict physical failures, calculate Remaining Useful Life (RUL), classify fault types, or offer safety-critical performance guarantees.

---

## 2. Setup & Configuration

### Prerequisites
* Python 3.10+
* Running Flask Backend (defaults to `http://localhost:5000`)

### Installation
From the repository root or `frontend/` directory:

```bash
# Install frontend dependencies
pip install -r frontend/requirements.txt
```

### Environment Variables
Copy `.env.example` to `.env` or set environment variables:

```env
BACKEND_URL=http://localhost:5000
READ_TIMEOUT=10
INFERENCE_TIMEOUT=60
```

---

## 3. Running the Dashboard & Tests

### Execute Unit Test Suite
Run the 11-test frontend validation suite:

```bash
python -m pytest frontend/tests
```

### Launch Streamlit Dashboard
Start the web dashboard (from the `frontend/` directory):

```bash
streamlit run app.py
```

Access the UI at `http://localhost:8501`.

---

## 4. Live Presentation & Demo Walkthrough Script

Use this step-by-step presentation script to demonstrate the complete end-to-end functionality of MachineMind AI:

### Step 1: System Health & Overview Page (`app.py`)
1. Open the dashboard at `http://localhost:8501`.
2. Point out the **System Status Banner** confirming connection to Flask (`/api/health`) and Mongo status.
3. Review the **Scientific Research Disclaimer** banner explaining the unsupervised anomaly detection boundaries.
4. Note the top KPI metric row showing total machines, active watch states, anomaly states, and open alert counts.

### Step 2: Register Machinery (`pages/1_Machines.py`)
1. Navigate to **1_Machines** via the sidebar.
2. Under "Register New Machinery", enter Machine ID `bearing_set2_demo`, Name `NASA IMS Test Rig Set 2`, and click **Register Machine**.
3. Select `bearing_set2_demo` from the dropdown list. Notice the initial state is `no_data`.

### Step 3: Client-Side Validation & Anomaly Analysis (`pages/2_Upload_Analyze.py`)
1. Navigate to **2_Upload_Analyze**.
2. **Error Path Test:** Upload an invalid file (e.g., small CSV or wrong columns). Observe the immediate client-side validation error box (`Expected 20480 rows x 4 columns`).
3. **Sequential Anomaly Persistence Test:**
   * Click **Browse files** and select 3 consecutive snapshot files from `ml-service/data/raw/IMS/2nd_test/2nd_test/`:
     - `2004.02.12.10.32.39` (Snapshot 1)
     - `2004.02.12.10.42.39` (Snapshot 2)
     - `2004.02.12.10.52.39` (Snapshot 3)
   * Click **Upload & Run Anomaly Inference**.
4. **Demonstrate 3-Snapshot Persistence Rule:**
   * Point out the step-by-step processing log:
     - **Snapshot 1 & 2:** Anomaly score exceeds baseline threshold, tagged as **Watch** (`consecutive_flagged_count = 1` then `2`). State is labeled `Anomalous snapshot — awaiting persistence confirmation`.
     - **Snapshot 3:** 3rd consecutive exceedance occurs (`consecutive_flagged_count = 3`). Persistence is confirmed, state transitions to **Anomaly Detected**, and a system alert is triggered.
   * Review the 4 channel cards showing individual scores, learned thresholds, and individual consecutive counts.
   * View the 4-channel raw signal waveform preview.

### Step 4: Historical Trends (`pages/3_History.py`)
1. Navigate to **3_History**.
2. Select `bearing_set2_demo`.
3. View the multi-channel time series score trend chart. Point out how individual channel anomaly scores evolved over the 3 snapshots relative to their constant baseline thresholds.
4. Enable **Sequence Replay Mode** to step through historical snapshots and visualize channel score changes interactively.

### Step 5: Alert Lifecycle Management (`pages/4_Alerts.py`)
1. Navigate to **4_Alerts**.
2. Observe the new open alert for `bearing_set2_demo` triggered by Channel 1 meeting the 3-snapshot persistence criteria.
3. Click **Acknowledge Alert**. Verify status updates from `open` to `acknowledged`.

### Step 6: Model Governance & Info (`pages/5_Model_Info.py`)
1. Navigate to **5_Model_Info**.
2. Review the model architecture overview (Isolation Forest + PCA baseline), feature extraction details (RMS, Peak-to-Peak, Crest Factor, Kurtosis, Band Energy), decision boundaries, and system limitations.

---

## 5. Architectural Integrity Checklist

* [x] **No Direct DB / ML Access:** Zero imports of `pymongo`, `joblib`, `sklearn`, or `ml-service`.
* [x] **API Contract Adherence:** All HTTP traffic routed via `services/api_client.py`.
* [x] **Clean Scientific Boundaries:** No false claims regarding RUL, failure prediction, or latency guarantees.
