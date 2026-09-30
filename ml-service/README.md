# MachineMind AI

**Unsupervised Vibration Anomaly Detection Prototype for Rotating Machinery**

MachineMind AI is an unsupervised vibration anomaly detection and condition monitoring research prototype for rotating machinery using machine learning. The system learns normal vibration dynamics from an assumed healthy baseline period and detects statistical deviations in sequential accelerometer snapshots.

> **Project Boundary & Scope Note:** MachineMind AI is a research and learning prototype built on laboratory test data. It provides statistical early-warning indicators; it does **NOT** predict remaining useful life (RUL), exact failure timestamps, or specific physical defect diagnoses, and it is **NOT** certified for industrial safety-critical deployment.

---

## Overview

Rotating machinery components, such as rolling-element bearings, experience structural degradation over time under continuous mechanical stress. In condition-monitoring applications, early detection of abnormal vibration patterns allows operators to identify potential machinery distress before catastrophic breakdown.

MachineMind AI addresses this challenge using an **unsupervised machine learning pipeline**:
- **Baseline Learning:** Fits feature scaling and novelty detection models strictly on an assumed healthy initial operating period.
- **Sequential Novelty Scoring:** Computes non-negative anomaly scores for incoming 1-second vibration snapshots.
- **Persistence-Filtered Alerting:** Applies a temporal persistence rule ($k=3$ consecutive exceedances) to suppress transient operational noise while flagging sustained anomalous behavior.

---

## Dataset

The prototype is developed and evaluated on **Set 2** of the **NASA IMS Bearing Dataset** (University of Cincinnati Center for Intelligent Maintenance Systems, hosted on the NASA Prognostics Data Repository):

- **Snapshot Matrix:** 984 sequential snapshot files recorded every 10 minutes (600 seconds) over ~7 days (2004-02-12 10:32:39 to 2004-02-19 06:22:39).
- **Channels & Sensors:** 4 vibration channels recorded from the bearing test rig.
- **Snapshot Size & Rate:** 20,480 samples per channel per snapshot; reported sampling rate 20 kHz.
- **Experimental Setup:** Fixed rotational speed (~2000 RPM) and constant radial load (6000 lbs) on a laboratory test rig.
- **Outcome:** Outer-race defect evolution on Bearing 1 (Channel 1).
- **Data Limitations:** $n=1$ single run-to-failure trajectory, no per-snapshot ground-truth failure-onset labels, uncalibrated raw vibration units, and single operating condition.

*Dataset Citation:* J. Lee et al. (2007). IMS, University of Cincinnati. "Bearing Data Set", NASA Prognostics Data Repository, NASA Ames Research Center.

---

## ML Pipeline

The end-to-end inference and evaluation pipeline operates sequentially:

```
Raw Vibration Snapshot Array (20480 x 4)
       │
       ▼
DC Offset Removal (vals - mean)
       │
       ▼
28 Time-Domain Features (7 per channel)
       │
       ▼
RobustScaler (fit on snapshots 0..159)
       │
       ▼
Per-Channel Anomaly Models (Isolation Forest / PCA)
       │
       ▼
Non-Parametric P99 Calibration Cutoffs (snapshots 160..179)
       │
       ▼
Persistence Filtering (k=3 consecutive exceedances)
       │
       ▼
Logical OR System Alert (Active if ANY channel sustains alert)
```

---

## Feature Engineering

The finalized model pipeline extracts **28 time-domain statistical features** (7 features per channel):

1. **Mean**: DC baseline offset.
2. **Standard Deviation**: AC signal dispersion ($ddof=1$).
3. **Root Mean Square (RMS)**: Total waveform energy $\sqrt{\text{mean}(x^2)}$.
4. **Peak-to-Peak (P2P)**: Dynamic range $\max(x) - \min(x)$.
5. **Skewness**: Third standardized moment (distribution asymmetry).
6. **Kurtosis**: Fisher excess kurtosis ($\text{Gaussian} = 0.0$).
7. **Crest Factor**: Peak-to-RMS spikiness ratio $\frac{\max(|x|)}{\text{RMS} + \epsilon}$.

*Note on Pipeline Evolution:* Initial exploratory feature extraction (Phase 5) generated 36 features including spectral energy and centroid. Controlled feature ablation (Phase 9 EXP-04) selected the 28 time-domain feature set to eliminate high-frequency spectral windowing noise and improve feature parsimony.

---

## Models & Final Configuration

The system packages per-channel Isolation Forest pipelines as the primary model, alongside per-channel Principal Component Analysis (PCA) Reconstruction Error as a secondary reference model.

### Primary Model: Isolation Forest
- **Architecture**: Independent per-channel estimators (4 models total).
- **Hyperparameters**: `n_estimators=100`, `max_samples="auto"`, `contamination="auto"`, `random_state=42`.
- **Normalization**: `RobustScaler` (median/IQR scaling), fit strictly on healthy baseline `snapshots 0..159`.

### Secondary Model: PCA Reconstruction Error
- **Architecture**: Independent per-channel estimators (4 models total).
- **Hyperparameters**: `n_components=3` components, `random_state=42`.

### Calibration & Decision Logic
- **Baseline Fit Window**: `snapshots 0..159` (~26.7 hours, assumed healthy).
- **Calibration Window**: `snapshots 160..179` (~3.3 hours). Non-parametric $P_{99}$ score percentiles establish channel anomaly cutoffs.
- **Persistence Filter**: Requires $k=3$ consecutive raw threshold exceedances (20-minute confirmation delay) to trigger a sustained channel alert.
- **System Aggregation**: Logical OR policy across channels (system alert is active if ANY bearing channel triggers a sustained alert).

---

## Inference Engine Usage

The [`src/inference.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/inference.py) module provides the `AnomalyInferenceEngine` class for snapshot evaluation:

```python
from src.inference import AnomalyInferenceEngine
import numpy as np

# 1. Initialize inference engine (loads packaged artifacts & metadata)
engine = AnomalyInferenceEngine(artifacts_dir="models", model_type="iforest", stateful=True)

# 2. Receive raw 4-channel snapshot array shape (20480, 4)
raw_snapshot = np.random.normal(0, 0.05, size=(20480, 4))

# 3. Predict anomaly scores and alert status
result = engine.predict_snapshot(raw_snapshot, timestamp="2004-02-12 10:32:39")

print("System Alert:", result["system_alert"])
print("Channel Scores:", result["channel_scores"])
print("Sustained Flags:", result["channel_sustained_flags"])
```

### Engine Capabilities
- **Input Validation**: Validates raw input shape `(20480, 4)` and rejects non-numeric, empty, NaN, or Inf inputs.
- **Environment Compatibility Check**: Compares runtime dependency versions against `model_metadata.json` records and issues `UserWarning` if mismatched.
- **Stateful vs Stateless Modes**: Supports stateful sequential counter tracking ($k=3$) or stateless single-shot scoring. Includes `reset_state()` for sequence resets.

---

## Repository Structure

```
ml-service/
├── data/
│   ├── raw/                  # Raw dataset archives (IMS.zip, git-ignored)
│   └── processed/            # Processed manifests & feature matrices (git-ignored)
├── models/
│   ├── iforest_pipeline_v1.joblib  # Packaged Isolation Forest model artifact (git-ignored)
│   ├── pca_pipeline_v1.joblib      # Packaged PCA model artifact (git-ignored)
│   └── model_metadata.json         # Human-readable metadata & thresholds (tracked)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb# Signal Preprocessing & Windowing
│   ├── 04_statistical_baseline.ipynb # Z-Score & Modified Z-Score Baseline
│   ├── 05_model_development.ipynb    # iForest & PCA Model Development
│   ├── 06_evaluation.ipynb           # Chronological Evaluation Protocol
│   ├── 07_model_improvement.ipynb   # Controlled Experiments EXP-04..EXP-12
│   └── 08_inference_check.ipynb     # Inference Engine Verification
├── reports/
│   ├── figures/              # Trend & evaluation visualization PNGs
│   ├── evaluation_protocol.md# Frozen evaluation protocol specification
│   ├── model_card.md         # Detailed Model Card
│   └── ...                   # Phase documentation & experiment logs
├── src/                      # Reusable python modules & unit test suites
│   ├── preprocessing.py      # DC offset removal & windowing
│   ├── feature_extraction.py # 28 time-domain feature extraction
│   ├── baseline.py           # Statistical baseline calculations
│   ├── models.py             # iForest & PCA pipeline classes
│   ├── evaluation.py         # Persistence filtering & evaluation logic
│   ├── export_models.py      # Artifact packaging & metadata builder
│   ├── inference.py          # AnomalyInferenceEngine class
│   └── test_*.py             # Unit test suite (56 tests)
├── requirements.txt          # Pinned Python dependencies
├── PROJECT_CONTEXT.md        # Permanent project guidelines & context
├── ML_PROGRESS.md           # Live phase progress tracker
└── README.md                 # Project README
```

*Git Exclusion Note:* Raw dataset archives (`IMS.zip`), extracted snapshot files, generated CSV matrices, model binaries (`*.joblib`), and virtual environments (`venv/`) are strictly excluded from version control via `.gitignore`.

---

## Reproducibility Status

This project has been verified in Python **3.14.2** with pinned dependencies in [`requirements.txt`](file:///d:/Projects/MachineMind%20AI/ml-service/requirements.txt):

```text
numpy==2.5.3
pandas==3.0.6
scipy==1.18.1
scikit-learn==1.9.1
joblib==1.6.0
matplotlib==3.11.2
seaborn==0.13.2
jupyter==1.1.1
```

### Reproducibility Classification
- **LEVEL A (Same Code + Same Data + Same Environment $\rightarrow$ Identical Results)**: **SUPPORTED & VERIFIED**. Executing `unittest discover -s src` passes 56/56 unit tests. Model scores match batch references with exact floating-point precision (iForest max diff `0.00e+00`, PCA max diff `1.77e-14`).
- **LEVEL B (Same Code + Same Data + Different Environment)**: **NOT DEMONSTRATED**. A different compatible environment was not explicitly tested.
- **LEVEL C (Scientific Reproducibility of Methodology/Protocol)**: **PARTIALLY SUPPORTED**. The computational protocol is fully documented and top-to-bottom executable, but this does NOT establish multi-machine scientific generalizability or independent physical replication.

---

## Evaluation Summary

Model performance was evaluated chronologically on NASA IMS Set 2:
- **Baseline Fit Window**: `snapshots 0..159` (~26.7 hours).
- **Baseline Reference Window**: `snapshots 160..199` (~6.7 hours, 0 sustained false alarms under $k=3$).
- **Sequential Evaluation Window**: `snapshots 200..983` (~130.7 hours).
- **Detection Confirmation**: On Channel 1 (Bearing 1 outer-race defect), the persistence-filtered ($k=3$) Isolation Forest issued a sustained anomaly alert at snapshot index 534 (`2004-02-15 20:42:39`). Approximately 74h 50m elapsed between this alert timestamp and the end of the available recording at snapshot index 983 (`2004-02-19 06:22:39`), spanning 450 snapshots inclusive of both endpoints.

*Methodological Disclosure:* The Phase 8 evaluation period (`snapshots 200..983`) was examined during Phase 8 prior to Phase 9 model improvements. Consequently, Phase 9 results represent a retrospective controlled refinement rather than a double-blind holdout test.

---

## Limitations

1. **Single Failure Trajectory ($n=1$):** IMS Set 2 documents a single outer-race defect trajectory on Bearing 1. Results describe a single-case study and cannot prove general industrial reliability.
2. **No Ground-Truth Health Annotations:** Dataset contains no binary per-snapshot healthy/faulty labels. Anomaly alerts measure statistical novelty relative to baseline operation, **not** verified physical fault onset.
3. **No RUL or Failure-Time Prediction:** Anomaly scores quantify statistical distance from baseline operation; they do **not** estimate remaining useful life (RUL) or exact failure timestamps.
4. **Single Rig Operating Condition:** Test conducted under constant speed (~2000 RPM) and radial load (6000 lbs) in a laboratory setting. Performance under variable speeds, transient loads, or harsh industrial noise is unverified.
5. **Cross-Channel Vibration Coupling:** The bearings share a common shaft, so cross-channel vibration coupling may contribute to correlated anomaly scores.
6. **Evaluation Data Reuse:** Phase 9 model refinements were evaluated on dataset periods examined in earlier phases.

---

## Security & Model Artifact Warning

Model artifacts (`.joblib` files in `models/`) rely on `joblib` / `pickle` serialization. **Security Warning:** Never load joblib or pickle model artifacts from untrusted or unverified external sources, as deserialization can execute arbitrary code.

---

## Status

**Completed Research / Learning Prototype.** All ML development phases (Phases 1 through 10) and reproducibility audits (Phase 11) are complete and verified.
