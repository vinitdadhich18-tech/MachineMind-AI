# MachineMind AI — Final ML Technical Summary

**Author:** Machine Learning Engineering Team  
**Project:** MachineMind AI (`ml-service/`)  
**Phase:** Phase 11 — ML Completion and Integration Readiness  
**Document Status:** Complete Final Technical Summary  
**Python Environment:** 3.14.2 (Virtual Environment `venv`)  
**Target Dataset:** NASA IMS Bearing Dataset (Set 2)

---

## 1. Executive Summary

MachineMind AI is an **unsupervised vibration anomaly detection and condition monitoring research prototype** designed for rotating machinery. The system learns normal baseline vibration dynamics from an initial assumed healthy operating window and flags statistical novelties in sequential 1-second accelerometer snapshots.

- **ML Task**: Unsupervised Anomaly Detection (Novelty Detection).
- **Target Dataset**: NASA IMS Bearing Dataset (Set 2: 984 snapshot files, 4 vibration channels).
- **Primary Output**: Continuous per-channel anomaly scores, persistence-filtered ($k=3$) sustained alert flags, and a Logical OR system alert decision.
- **Current Project Status**: **Completed Research / Learning Prototype** (Phases 1 through 11 fully executed and verified).
- **Explicit Boundary & Non-Goals**: This system is a research prototype built on laboratory test data. It does **NOT** predict remaining useful life (RUL), exact failure timestamps, or specific physical defect diagnoses, and it is **NOT** certified for industrial safety-critical deployment.

---

## 2. Problem Definition

Rotating machinery components, such as rolling-element bearings, suffer structural fatigue under continuous mechanical load. In industrial condition monitoring, detecting subtle shifts in vibration signatures before major catastrophic breakdown enables proactive maintenance scheduling.

### Anomaly Detection vs. Physical Fault Diagnosis
MachineMind AI explicitly separates **statistical anomaly detection** from **physical fault diagnosis**:
- **Statistical Anomaly Detection (Implemented)**: Quantifies whether a new vibration snapshot statistically deviates from the baseline reference distribution. An alert signifies *"this measurement differs significantly from normal baseline operation."*
- **Physical Fault Diagnosis (Out of Scope)**: Classifies specific defect mechanisms (e.g., inner race, outer race, ball, or cage fault) or estimates structural degradation time-to-failure. The prototype does **NOT** perform fault classification or failure-time prediction.

---

## 3. Dataset

The prototype was developed and evaluated on **Set 2** of the **NASA IMS Bearing Dataset** (University of Cincinnati Center for Intelligent Maintenance Systems, hosted on the NASA Prognostics Data Repository):

- **Snapshot Count**: 984 sequential snapshot files recorded every 10 minutes (600 seconds) over approximately 7 days (2004-02-12 10:32:39 to 2004-02-19 06:22:39).
- **Vibration Channels**: 4 vibration channels recorded from the bearing test rig.
- **Snapshot Size & Rate**: 20,480 samples per channel per snapshot; reported sampling rate 20 kHz (1.024-second duration per snapshot).
- **Operating Conditions**: Fixed rotational speed (~2000 RPM) and constant radial load (6000 lbs) on a laboratory shaft rig.
- **Degradation Outcome**: Natural outer-race defect evolution on Bearing 1 (Channel 1).

### Verified Dataset Limitations
1. **Single Trajectory ($n=1$)**: The dataset documents a single run-to-failure experiment. Results represent a single-case study under constant laboratory conditions.
2. **No Per-Snapshot Ground Truth**: The dataset contains no binary healthy/faulty annotations per snapshot. The healthy period is an assumed window.
3. **Uncalibrated Sensor Units**: Vibration amplitudes are recorded in uncalibrated raw vibration units without explicit acceleration calibration multipliers.
4. **Shared Shaft Coupling**: The bearings share a common shaft, so cross-channel vibration coupling may contribute to correlated anomaly scores.

---

## 4. Development Pipeline (Phases 1–5)

The initial ML development phases established data loading, signal cleaning, and feature extraction:

1. **Acquisition & Verification (Phase 1–2)**: Primary archive `IMS.zip` (SHA-256 `6CB42C263B0281C725ABF99F4B9FCF49915C949F31DBD2333877DC2E06CE9EC2`) was verified and structured under `data/raw/`.
2. **Exploratory Data Analysis (Phase 3)**: Scanned all 984 snapshot files (100% valid structure, 0 NaN/Inf values). Documented Fisher excess kurtosis conventions ($\text{Gaussian}=0.0$).
3. **Signal Preprocessing (Phase 4)**: Implemented mean-centering DC offset removal (`vals - mean`) and non-overlapping snapshot windowing. Generated `data/processed/manifest_set2.csv`.
4. **Feature Engineering (Phase 5)**: Extracted an initial set of 36 features (7 time-domain + 2 frequency-domain per channel across 4 channels) into `data/processed/features_set2.csv`.

---

## 5. Statistical Baseline (Phase 6)

Phase 6 developed non-ML statistical baseline scoring to establish reference behavior:
- **Partitioning**: Fit window (`snapshots 0..159`, ~26.7 hours), Reference Evaluation window (`snapshots 160..199`, ~6.7 hours), Sequential Evaluation window (`snapshots 200..983`, ~130.7 hours).
- **Metrics**: Standard $Z$-score and Modified $Z$-score (MAD-based) calculated per feature with zero-std numerical guards ($\epsilon = 1e-12$).
- **Role**: Mathematical reference indicators (lines drawn at 3.0 and 3.5) established baseline score behavior, confirming stable reference values ($\sim 1.0$) during healthy operation and sharp score elevation late in the run. The baseline was not used as a supervised predictive validator.

---

## 6. Initial ML Models (Phase 7)

Phase 7 introduced unsupervised machine learning models evaluated on the initial 36-feature matrix:
- **Models Implemented**: Per-channel Isolation Forest (`n_estimators=100`, `contamination="auto"`) and per-channel Principal Component Analysis (PCA) Reconstruction Error ($k=2$ components).
- **Scaler**: `RobustScaler` (median/IQR scaling), fit strictly on healthy baseline `snapshots 0..159`.
- **Thresholding**: Derived non-parametric $P_{99}$ score cutoffs from validation snapshots (`160..199`).
- **Score Normalization**: Standardized outputs so that non-negative scores guarantee `HIGHER SCORE = MORE ANOMALOUS`.

*Evolution Note:* This initial 36-feature, PCA $k=2$ configuration served as the baseline model. Phase 9 subsequently refined the pipeline via systematic ablation and grid searches.

---

## 7. Evaluation Methodology & Protocol (Phase 8)

Phase 8 established a frozen evaluation protocol ([`reports/evaluation_protocol.md`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/evaluation_protocol.md)) to test model behavior across chronological partitions:

- **Chronological Partitions**: Fit (`0..159`), Validation (`160..199`), Sequential Evaluation (`200..983`).
- **Persistence Filtering**: Evaluated temporal persistence rules ($k=1, 3, 5$). A sustained alert requires $k$ consecutive raw threshold exceedances. Persistence $k=3$ (20-minute confirmation delay) eliminated false alarms on the evaluated reference validation period (`160..199`).
- **System Aggregation**: Logical OR policy across the 4 bearing channels.

### Methodological Disclosure: Evaluation-Data Reuse
The sequential evaluation period (`snapshots 200..983`) was examined during Phase 8 prior to Phase 9 model improvement experiments. Consequently, Phase 9 experiment results represent a **retrospective controlled refinement** on previously observed data rather than an untouched double-blind holdout test.

---

## 8. Model Improvement Experiments (Phase 9)

Phase 9 executed controlled experiments (EXP-04 through EXP-12) to refine feature sets, hyperparameter choices, and alert aggregation rules:

| Experiment | Question | Empirical Result | Approved Decision |
|---|---|---|---|
| **EXP-04** | Full 36 features vs. 28 time-domain features? | Tied at 0 sustained FAs ($k=3$). 28-feature set reduced raw single-snapshot false alarms by 25.0% (9 vs 12). | **Select 28 Time-Domain Feature Set** (parsimony, lower raw noise). |
| **EXP-05** | `RobustScaler` vs `StandardScaler`? | Identical anomaly scores ($\max \|\Delta s\| = 0.0$). Isolation Forest splits are rank-invariant. | **Retain `RobustScaler`** as baseline control. |
| **EXP-06** | PCA rank $k \in \{1, 2, 3\}$ components? | All achieved 0 sustained FAs. $k=3$ reduced diagnostic raw FAs to 4/80 and lowered calibration MSE to $0.1525$. | **Select PCA $k=3$** as improved component rank. |
| **EXP-07** | iForest $n_{\text{est}} \in \{50,100,200\} \times \text{max\_samp} \in \{0.5,1.0,\text{"auto"}\}$? | All 9 grid configurations achieved 0 sustained false alarms ($k=3$). | **Retain `n_estimators=100`, `max_samples="auto"`** as baseline control. |
| **EXP-08** | Parametric vs non-parametric $P_{99}$ threshold? | $P_{99}$ and $P_{99.5}$ yielded identical cutoffs; parametric bounds were overly conservative. | **Retain Non-Parametric $P_{99}$ Cutoff** on calibration data. |
| **EXP-09** | Persistence filter $k \in \{1, 3, 5\}$ exceedances? | $k=3$ and $k=5$ eliminated diagnostic false alarms. $k=3$ requires 20-min delay vs 40-min for $k=5$. | **Retain $k=3$ Persistence Filter**; it provided the same sustained-false-alarm suppression as $k=5$ with a shorter confirmation delay. |
| **EXP-10** | Baseline fit window $N_{\text{fit}}=100$ vs $160$? | Both achieved 0 sustained FAs. $N_{\text{fit}}=160$ provided lower raw noise (9 vs 12). Calibration shift confound noted. | **Retain $N_{\text{fit}}=160$** as baseline control window. |
| **EXP-11** | Per-Channel (4 models) vs Pooled (1 joint model)? | Both configurations produced zero sustained false alarms on the diagnostic comparison, while per-channel modeling preserved channel-level localization. | **Retain Per-Channel Modeling** with Logical OR Aggregation. |
| **EXP-12** | Master Synthesis & Alert Policy? | Synthesized EXP-04..EXP-11 outcomes into finalized system spec. | **Final System Pipeline Approved**. |

---

## 9. Final Model Configuration

The finalized MachineMind AI model configuration consists of:

- **Raw Input**: 4-channel vibration snapshot array of shape `(20480, 4)`.
- **Preprocessing**: Mean-centering DC offset removal (`vals - mean`).
- **Feature Set**: 28 time-domain features (7 per channel: `mean`, `std`, `rms`, `p2p`, `skewness`, `kurtosis`, `crest_factor`).
- **Scaler**: `RobustScaler` (median/IQR scaling), fit strictly on healthy baseline `snapshots 0..159`.
- **Primary Model**: Per-channel Isolation Forest (`n_estimators=100`, `max_samples="auto"`, `contamination="auto"`, `random_state=42`).
- **Secondary Model**: Per-channel PCA Reconstruction Error ($k=3$ components, `random_state=42`).
- **Calibration Window**: `snapshots 160..179` (~3.3 hours). Non-parametric $P_{99}$ score percentiles define channel thresholds.
- **Persistence Filter**: $k=3$ consecutive exceedances required for sustained channel alert.
- **System Alert Policy**: Logical OR across the 4 bearing channels.

---

## 10. Model Packaging & Inference Engine (Phase 10)

Phase 10 packaged trained pipelines and implemented the inference engine:

- **Artifacts**: Packaged into [`models/iforest_pipeline_v1.joblib`](file:///d:/Projects/MachineMind%20AI/ml-service/models/iforest_pipeline_v1.joblib), [`models/pca_pipeline_v1.joblib`](file:///d:/Projects/MachineMind%20AI/ml-service/models/pca_pipeline_v1.joblib), and human-readable [`models/model_metadata.json`](file:///d:/Projects/MachineMind%20AI/ml-service/models/model_metadata.json).
- **Inference Engine**: [`src/inference.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/inference.py) implements `AnomalyInferenceEngine`, providing input validation, DC removal, 28-feature extraction, scaler application, score computation, threshold checking, $k=3$ counter state management, and Logical OR alert decision making.
- **Test Suite Verification**: Executed `unittest discover -s src` (**56 of 56 unit tests passed**).
- **Numerical Reference Check**: Fresh-process inference scores matched batch reference calculations with exact floating-point precision (Isolation Forest max diff `0.00e+00`, PCA max diff `1.77e-14`).
- **Deployment Status**: The inference engine is a self-contained python inference module, **not** a certified production microservice or API server.

---

## 11. Reproducibility Audit (Phase 11)

Project reproducibility was audited in Python **3.14.2** with pinned dependencies in [`requirements.txt`](file:///d:/Projects/MachineMind%20AI/ml-service/requirements.txt):

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
- **LEVEL A (Same Code + Same Data + Same Environment $\rightarrow$ Identical Results)**: **SUPPORTED & VERIFIED**. All 56 unit tests pass, and model anomaly scores match reference outputs within `0.00e+00` (iForest) and `1.77e-14` (PCA).
- **LEVEL B (Same Code + Same Data + Different Environment)**: **NOT DEMONSTRATED**. A different compatible environment was not explicitly tested.
- **LEVEL C (Scientific Reproducibility of Methodology/Protocol)**: **PARTIALLY SUPPORTED**. The computational protocol is fully documented and top-to-bottom executable, but this does NOT establish multi-machine scientific generalizability or independent physical replication.

---

## 12. Data & Repository Integrity Audit

Read-only verification of raw data and Git repository safety confirmed:

- **Raw Archive Integrity**: [`data/raw/IMS.zip`](file:///d:/Projects/MachineMind%20AI/ml-service/data/raw/IMS.zip) size `1,061,902,801` bytes, SHA-256 hash `6CB42C263B0281C725ABF99F4B9FCF49915C949F31DBD2333877DC2E06CE9EC2` (**100% Match**).
- **Extracted Dataset Audit**: 984 files in `data/raw/IMS/2nd_test/2nd_test/` (544,618,480 total bytes). Exhaustive scan of all 80,609,280 scalar numeric values confirmed **0 malformed files, 0 NaN values, 0 +Inf values, and 0 -Inf values**.
- **Git Exclusion Verification**: Verified via `git check-ignore -v` that raw dataset archives (`IMS.zip`), extracted raw files, processed CSVs (`features_set2.csv`, `model_scores_set2.csv`), model joblib binaries (`*.joblib`), and virtual environments (`venv/`) are strictly ignored by `.gitignore`.

---

## 13. Key Empirical Results

1. **False-Alarm Observations**:
   - **Diagnostic Slice**: The selected 28-feature configuration produced 9 raw false-alarm snapshots on `snapshots 180..199`, with 0 sustained false alarms under $k=3$.
   - **Reference Period**: Under the established reference evaluation on `snapshots 160..199`, $k=3$ produced 0 sustained false alarms.
2. **Detection Confirmation Timing**: On Channel 1 (Bearing 1 outer-race defect), the persistence-filtered ($k=3$) Isolation Forest issued a sustained anomaly alert at snapshot index 534 (`2004-02-15 20:42:39`).
3. **Elapsed Time to End of Recording**: Approximately **74 hours and 50 minutes** elapsed between the alert confirmation timestamp at snapshot 534 (`2004-02-15 20:42:39`) and the end of the available recording at snapshot index 983 (`2004-02-19 06:22:39`), spanning 450 snapshots inclusive of both endpoints. *(Note: This elapsed time is measured against the end of recording as an experimental proxy, NOT a verified physical failure timestamp).*
4. **Feature Parsimony**: Ablating from 36 features to 28 time-domain features reduced raw single-snapshot noise sensitivity by 25.0% while maintaining zero sustained false alarms on evaluated diagnostic windows.

---

## 14. Comprehensive Limitations

1. **Single Trajectory ($n=1$)**: Evaluated on a single run-to-failure trajectory on Bearing 1. Results describe a single-case study and cannot prove general industrial reliability.
2. **No Ground-Truth Health Annotations**: Dataset contains no binary per-snapshot healthy/faulty labels. Anomaly alerts measure statistical novelty relative to baseline operation, **not** verified physical fault onset.
3. **No RUL or Failure Prediction**: Anomaly scores quantify statistical distance from baseline operation; they do **not** estimate remaining useful life (RUL) or exact failure timestamps.
4. **Fixed Operating Condition**: Test conducted under constant speed (~2000 RPM) and radial load (6000 lbs) in a laboratory setting. Performance under variable speeds, transient loads, or harsh industrial noise is unverified.
5. **Cross-Channel Vibration Coupling**: The bearings share a common shaft, so cross-channel vibration coupling may contribute to correlated anomaly scores.
6. **Evaluation Data Reuse**: Phase 9 model refinements were evaluated on dataset periods examined in earlier phases.
7. **Uncalibrated Sensor Units**: Accelerometer values are recorded in uncalibrated raw vibration units.
8. **Unverified Environment Generalization**: Level B cross-environment reproducibility was not tested.

---

## 15. What the Project Demonstrates

The empirical evidence directly supports that the project successfully demonstrates:

- **Reproducible Data Pipeline**: End-to-end automated signal verification, DC offset removal, and 28 time-domain feature extraction.
- **Unsupervised Anomaly Modeling**: Modular per-channel Isolation Forest and PCA pipelines calibrated strictly on baseline data.
- **Persistence Filtering**: Effective suppression of transient single-snapshot exceedances on evaluated reference windows using temporal persistence ($k=3$).
- **Deterministic Inference Packaging**: Versioned joblib packaging and an `AnomalyInferenceEngine` achieving exact numerical score reproducibility.
- **Methodological Transparency**: Fully documented evaluation protocol, experiment logs, model cards, and limitation disclosures.

---

## 16. What the Project Does Not Demonstrate

The project explicitly does **NOT** demonstrate or establish:

- Exact failure-time prediction or remaining useful life (RUL) estimation.
- Supervised defect classification (inner race vs outer race vs ball vs cage).
- Generalization across different machines, bearing geometries, or operating loads.
- Industrial reliability or production safety certification.
- Double-blind holdout evaluation (due to Phase 8 evaluation data re-inspection).

---

## 17. Final Project Status

**Completed Research / Learning Prototype.**

All ML development phases (Phases 1 through 10) and completion audits (Phase 11 Steps 1 through 5) have been fully executed, verified, and documented.
