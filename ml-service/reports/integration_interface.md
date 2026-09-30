# MachineMind AI — Integration Interface Specification

**Component:** `ml-service`  
**Phase:** Phase 11 — ML Completion and Integration Readiness  
**Target Module:** [`src/inference.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/inference.py) (`AnomalyInferenceEngine`)  
**Document Status:** Technical Interface Specification  
**Python Runtime:** Python 3.14.2  

---

## 1. Purpose & Architectural Context

This document defines the software interface contract for integrating the MachineMind AI machine learning inference engine into downstream software architectures (e.g., a future backend service, event processor, or streaming pipeline).

> **Architectural Boundary Note:** The current implementation in `ml-service/` is a **self-contained Python inference module** (`AnomalyInferenceEngine`), **NOT** an active HTTP REST API, gRPC service, or microservice. This document describes how an external application can import, instantiate, and consume the Python inference engine, and provides a proposed blueprint for future API wrapper endpoints.

---

## 2. Inference Engine Specification

The machine learning inference interface is implemented by the `AnomalyInferenceEngine` class located in [`src/inference.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/inference.py).

### 2.1 Class Signature & Initialization

```python
class AnomalyInferenceEngine:
    def __init__(
        self,
        artifacts_dir: Union[str, Path] = "models",
        model_type: str = "iforest",
        stateful: bool = True
    ) -> None:
```

#### Parameters
- **`artifacts_dir`** (`str` or `Path`, default `"models"`): Path to the directory containing packaged joblib artifacts (`iforest_pipeline_v1.joblib` or `pca_pipeline_v1.joblib`) and metadata (`model_metadata.json`).
- **`model_type`** (`str`, default `"iforest"`): Model selection flag. Supported values: `"iforest"` (Isolation Forest) or `"pca"` (PCA Reconstruction Error). Case-insensitive and trimmed.
- **`stateful`** (`bool`, default `True`):
  - `True`: Maintains in-memory temporal persistence counters (`consecutive_exceedances`) across sequential snapshot evaluation calls.
  - `False`: Operates statelessly. Computes raw anomaly scores and raw threshold flags, but sustained flags and system alerts remain `False`.

#### Initialization Mechanics
Upon instantiation, the engine automatically:
1. Loads [`models/model_metadata.json`](file:///d:/Projects/MachineMind%20AI/ml-service/models/model_metadata.json) and verifies schema.
2. Executes runtime vs. metadata version compatibility checks via `_check_environment_compatibility()`.
3. Loads the designated joblib artifact file (`iforest_pipeline_v1.joblib` or `pca_pipeline_v1.joblib`).
4. Validates that all 4 channels (`ch1`, `ch2`, `ch3`, `ch4`) are present in the artifact dictionary.
5. Extracts feature ordering (28 ordered names), channel P99 thresholds, and persistence parameter $k=3$.
6. Initializes in-memory channel persistence counters `{ch: 0 for ch in ["ch1", "ch2", "ch3", "ch4"]}`.

---

### 2.2 Core Methods

#### 1. Snapshot Evaluation: `predict_snapshot()`
```python
def predict_snapshot(
    self,
    raw_snapshot: Union[np.ndarray, pd.DataFrame],
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
```
- **Input**: A single 1-second raw vibration snapshot of shape `(20480, 4)` as a 2D `np.ndarray` or `pd.DataFrame`, plus an optional timestamp string.
- **Output**: A structured Python dictionary containing native types (floats, bools, ints, strings).

#### 2. State Reset: `reset_state()`
```python
def reset_state(self) -> None:
```
- Resets all in-memory channel exceedance counters (`consecutive_exceedances`) to 0. Should be invoked when starting evaluation on a new machinery stream or after maintenance events.

#### 3. Raw Input Validation: `validate_raw_snapshot()`
```python
def validate_raw_snapshot(
    self,
    raw_snapshot: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
```
- Validates data type, dimensionality, shape `(20480, 4)`, and absence of NaN or Infinite values. Returns a validated 2D `float64` NumPy array.

---

### 2.3 Input Contract & Validation Behavior

| Attribute | Required Specification | Validation Rule / Exception |
|---|---|---|
| **Data Container** | `np.ndarray` or `pd.DataFrame` | Raises `TypeError` if other type supplied. |
| **Data Type** | Numeric (`float64`, `float32`, `int64`) | Raises `TypeError` if string or non-numeric type present. |
| **Array Rank** | 2D Array (`ndim == 2`) | Raises `ValueError` if 1D array or 3D tensor supplied. |
| **Snapshot Dimensions** | Shape `(20480, 4)` | Raises `ValueError` if row count $\neq 20480$ or channel count $\neq 4$. |
| **Numerical Integrity** | Finite values only | Raises `ValueError` if array contains `NaN`, `+Inf`, or `-Inf`. |
| **Empty Check** | Non-empty array (`size > 0`) | Raises `ValueError` if array is empty. |

*Input Shape & Types:* `(20480, 4)` shape; accepts `np.ndarray` or `pd.DataFrame` containing numeric data. The validated result is converted to a 2D `float64` NumPy array.

---

### 2.4 Internal Processing Stages

When `predict_snapshot(raw_snapshot)` is called, the engine executes 6 internal stages without calling model `.fit()`:

1. **Validation**: Enforces shape `(20480, 4)` and numerical sanity via `validate_raw_snapshot()`.
2. **Preprocessing**: Converts validated array to DataFrame with column headers `["Channel_1", "Channel_2", "Channel_3", "Channel_4"]`.
3. **Feature Extraction**: Calls `src.feature_extraction.extract_time_features()`. Removes DC offset (`vals - mean`) internally and computes 28 time-domain features (7 per channel: `mean`, `std`, `rms`, `p2p`, `skewness`, `kurtosis`, `crest_factor`).
4. **Per-Channel Scaling & Scoring**: For each channel `ch`:
   - Formulates 7-element feature vector `x_ch` aligned with metadata feature ordering.
   - Evaluates `pipeline.compute_anomaly_scores(x_ch)` using the pre-fitted `RobustScaler` and Isolation Forest / PCA model pipeline loaded from artifact.
   - Evaluates raw threshold condition: `raw_flag = bool(score_val > thresh_val)`.
5. **Persistence Filtering**:
   - If `stateful=True`: Increments `consecutive_exceedances[ch]` by 1 if `raw_flag` is `True`; resets counter to `0` if `False`. Sustained flag is `True` if counter $\ge k$ ($k=3$).
   - If `stateful=False`: Counters remain 0; sustained flags remain `False`.
6. **System Alert Aggregation**:
   - Evaluates Logical OR policy across channels: `system_alert = bool(any(channel_sustained_flags.values()))`.

---

## 3. Output Data Contract

`predict_snapshot()` returns a dictionary containing native Python data types guaranteed to be JSON-serializable:

```json
{
  "timestamp": "2004-02-12 10:32:39",
  "model_type": "iforest",
  "channel_scores": {
    "ch1": 0.38421052631578945,
    "ch2": 0.36105263157894735,
    "ch3": 0.4126315789473684,
    "ch4": 0.37894736842105264
  },
  "channel_thresholds": {
    "ch1": 0.5131679214172415,
    "ch2": 0.4725104927977178,
    "ch3": 0.6225579020689226,
    "ch4": 0.5066280904227561
  },
  "channel_raw_flags": {
    "ch1": false,
    "ch2": false,
    "ch3": false,
    "ch4": false
  },
  "channel_sustained_flags": {
    "ch1": false,
    "ch2": false,
    "ch3": false,
    "ch4": false
  },
  "system_alert": false,
  "state_consecutive_counts": {
    "ch1": 0,
    "ch2": 0,
    "ch3": 0,
    "ch4": 0
  }
}
```

### Field Definitions

- **`timestamp`** (`str` or `None`): Echoes the optional input timestamp string for record tracking.
- **`model_type`** (`str`): Model architecture evaluated (`"iforest"` or `"pca"`).
- **`channel_scores`** (`dict[str, float]`): Continuous anomaly score per channel. Guaranteed to be non-negative floats where **higher score = more anomalous**.
- **`channel_thresholds`** (`dict[str, float]`): Frozen non-parametric $P_{99}$ validation thresholds retrieved from metadata.
- **`channel_raw_flags`** (`dict[str, bool]`): Instantaneous single-snapshot threshold exceedance status (`score > threshold`).
- **`channel_sustained_flags`** (`dict[str, bool]`): Persistence-filtered alert status (`True` only when counter $\ge 3$).
- **`system_alert`** (`bool`): System-level alert decision (`True` if ANY channel sustained flag is `True`).
- **`state_consecutive_counts`** (`dict[str, int]`): Current consecutive exceedance count per channel.

---

## 4. Stateful vs. Stateless Behavior

### 4.1 In-Memory Sequence Tracking ($k=3$)

```
Snapshot N   : Raw Flag = False ──► Counter = 0 ──► Sustained Flag = False ──► System Alert = False
Snapshot N+1 : Raw Flag = True  ──► Counter = 1 ──► Sustained Flag = False ──► System Alert = False
Snapshot N+2 : Raw Flag = True  ──► Counter = 2 ──► Sustained Flag = False ──► System Alert = False
Snapshot N+3 : Raw Flag = True  ──► Counter = 3 ──► Sustained Flag = True  ──► System Alert = True
Snapshot N+4 : Raw Flag = False ──► Counter = 0 ──► Sustained Flag = False ──► System Alert = False
```

- **Reset Rule**: Any single snapshot where raw flag is `False` resets the consecutive counter for that channel to `0` immediately.
- **Process Memory Bound**: Counters are maintained strictly in Python instance memory (`self.consecutive_exceedances`). **State does NOT persist across process restarts or container recycling.**
- **Stateless Mode (`stateful=False`)**: Useful for batch scoring or parallel evaluation where temporal sequence tracking is handled externally.

---

## 5. Model Artifacts & Metadata Loading

The engine relies on packaged artifacts located in `artifacts_dir` (default `"models"`):

1. **[`models/model_metadata.json`](file:///d:/Projects/MachineMind%20AI/ml-service/models/model_metadata.json)**: Human-readable JSON metadata file specifying dataset provenance, 28 ordered feature names, pipeline configurations, environment package versions, channel thresholds, and persistence $k=3$.
2. **[`models/iforest_pipeline_v1.joblib`](file:///d:/Projects/MachineMind%20AI/ml-service/models/iforest_pipeline_v1.joblib)** (3.66 MB): Joblib serialized dictionary containing fitted per-channel `RobustScaler` and `IsolationForest` pipeline objects.
3. **[`models/pca_pipeline_v1.joblib`](file:///d:/Projects/MachineMind%20AI/ml-service/models/pca_pipeline_v1.joblib)** (5.8 KB): Joblib serialized dictionary containing fitted per-channel `RobustScaler` and `PCA` pipeline objects.

> **Security Warning:** Model artifacts use Joblib (`pickle`) serialization. Deserializing untrusted `.joblib` files can execute arbitrary python code. Artifacts must only be loaded from verified local storage.

---

## 6. Error & Exception Handling

The implementation enforces strict defensive error checks:

| Exception Type | Condition / Trigger | Error Message Pattern |
|---|---|---|
| `ValueError` | `model_type` not `"iforest"` or `"pca"` | `Unsupported model_type '...'. Choose 'iforest' or 'pca'.` |
| `FileNotFoundError` | Metadata or joblib artifact missing | `Model metadata JSON not found at: ...` |
| `KeyError` | Artifact missing channel key | `Artifact at ... is missing channel 'chX'.` |
| `TypeError` | Non-ndarray/DataFrame input | `Expected raw_snapshot to be np.ndarray or pd.DataFrame...` |
| `ValueError` | Empty input array | `Input raw_snapshot is empty.` |
| `TypeError` | Non-numeric data types in input | `Input raw_snapshot contains non-numeric data type...` |
| `ValueError` | Wrong array dimensions ($\neq 2\text{D}$) | `Expected 2D raw_snapshot of shape (20480, 4)...` |
| `ValueError` | Wrong array shape ($\neq 20480 \times 4$) | `Expected raw_snapshot shape (20480, 4), got ...` |
| `ValueError` | Input contains `NaN` | `Input raw_snapshot contains NaN values.` |
| `ValueError` | Input contains `Inf` | `Input raw_snapshot contains Infinite values.` |
| `UserWarning` | Environment version mismatch | `Environment version mismatch for 'package_name'...` |

---

## 7. Conceptual Integration Flow

Below is the conceptual flow for integrating `AnomalyInferenceEngine` into an external application:

```
┌────────────────────────────────────────────────────────┐
│               External Backend / Ingestion Service     │
│   (Receives 20,480 x 4 sensor snapshot from stream)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼  Passes raw array (20480, 4)
┌────────────────────────────────────────────────────────┐
│            src.inference.AnomalyInferenceEngine        │
│                                                        │
│  1. validate_raw_snapshot() ──► Checks shape & NaN/Inf │
│  2. extract_time_features() ──► DC offset & 28 feats   │
│  3. RobustScaler & Model   ──► Evaluates scores        │
│  4. Thresholding & k=3     ──► Updates counter state   │
│  5. Logical OR Policy      ──► Computes system alert   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼  Returns native result dict
┌────────────────────────────────────────────────────────┐
│               External Backend / Ingestion Service     │
│   (Publishes alert, logs metrics, triggers downstream)  │
└───────────────────────────┴────────────────────────────┘
```

---

## 8. Future Backend API Blueprint (Proposed / Not Implemented)

To wrap `AnomalyInferenceEngine` inside an HTTP web service (e.g., FastAPI), the following blueprint is proposed:

### Proposed Endpoint: `POST /api/v1/predict`

#### Proposed Request Payload (JSON)
```json
{
  "timestamp": "2004-02-12T10:32:39Z",
  "model_type": "iforest",
  "sensor_data": [
    [0.012, -0.004, 0.015, -0.002],
    "... 20,480 rows total x 4 channels ..."
  ]
}
```

#### Proposed Response Payload (JSON - HTTP 200 OK)
Returns the native output dictionary structure described in Section 3.

#### Proposed HTTP Status Mapping
- `200 OK`: Successful inference.
- `400 Bad Request`: Input validation failure (`ValueError`, wrong shape, `NaN`/`Inf`).
- `422 Unprocessable Entity`: Data type mismatch (`TypeError`).
- `500 Internal Server Error`: Missing model artifacts or unexpected internal exception.

*Implementation Note:* The FastAPI web server wrapper is a future design concept and is **not** currently implemented in `ml-service/`.

---

## 9. Operational Considerations

1. **Sequential Timestamp Ordering**: In stateful mode (`stateful=True`), snapshots must be ingested in chronological order for $k=3$ persistence counter logic to function correctly.
2. **Process Lifetime & State Persistence**: The engine maintains state in process memory. If deployed across multi-worker web servers (e.g., `uvicorn` with 4 workers), state will be fragmented across worker processes unless handled by an external state coordinator (e.g., Redis).
3. **Model Artifact Size**: The packaged Joblib artifacts are approximately 3.66 MB (`iforest_pipeline_v1.joblib`) and 5.8 KB (`pca_pipeline_v1.joblib`) on disk. Runtime memory usage has not been benchmarked.
4. **Inference Latency**: Formal inference-latency benchmarking and concurrency/load testing have not been conducted.

---

## 10. System Limitations

1. **Single Trajectory ($n=1$)**: Evaluated exclusively on NASA IMS Set 2 Bearing 1 outer-race failure trajectory.
2. **No Ground-Truth Health Annotations**: Dataset contains no binary per-snapshot healthy/faulty labels. Anomaly alerts measure statistical novelty relative to baseline operation, **not** verified physical defect onset.
3. **No RUL or Failure Prediction**: Anomaly scores quantify statistical distance from baseline operation; they do **not** estimate remaining useful life (RUL) or exact failure timestamps.
4. **Single Rig Operating Condition**: Constant speed (~2000 RPM) and radial load (6000 lbs) in a laboratory setting. Performance under variable speeds, transient loads, or harsh industrial noise is unverified.
5. **Cross-Channel Vibration Coupling**: The bearings share a common shaft, so cross-channel vibration coupling may contribute to correlated anomaly scores.
6. **Uncalibrated Sensor Units**: Accelerometer values are recorded in uncalibrated raw vibration units.
7. **Evaluation Data Reuse**: Phase 9 model refinements were evaluated on dataset periods examined in earlier phases.
8. **Not Production Certified**: Not certified for industrial machinery protection or safety-critical deployment.

---

## 11. Traceability & Source References

Every specification in this document is traceable to verified codebase files:

- [`src/inference.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/inference.py) — Core `AnomalyInferenceEngine` implementation.
- [`src/test_inference.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/test_inference.py) — Unit test suite verifying input validation, persistence, and output contracts.
- [`models/model_metadata.json`](file:///d:/Projects/MachineMind%20AI/ml-service/models/model_metadata.json) — Metadata JSON specification.
- [`reports/model_card.md`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/model_card.md) — Model card documentation.
- [`README.md`](file:///d:/Projects/MachineMind%20AI/ml-service/README.md) — Main repository documentation.
- [`reports/final_ml_summary.md`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/final_ml_summary.md) — Final technical ML summary report.
