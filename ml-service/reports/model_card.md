# MachineMind AI — Model Card

**Project:** MachineMind AI  
**Task:** Unsupervised Vibration Anomaly Detection  
**Model Version:** 1.0.0  
**Creation Date:** September 30, 2026  
**Author / Maintainer:** Computer Science Student Learning ML & Project Mentor  
**Target Environment:** Python 3.14.2 (`ml-service/`)  

---

## 1. Model Overview

MachineMind AI is a vibration-based condition-monitoring research prototype for rotating machinery (rolling-element bearings). It employs **unsupervised anomaly detection** to learn baseline vibration behavior from an assumed healthy operating period and flag statistical deviations in subsequent measurements.

- **Model Architectures:**
  1. **Isolation Forest Anomaly Pipeline:** 4 independent per-channel Isolation Forests (`ch1`..`ch4`), `n_estimators=100`, `max_samples="auto"`, `random_state=42`.
  2. **PCA Reconstruction Error Pipeline:** 4 independent per-channel PCA models (`ch1`..`ch4`), `n_components=3`, `random_state=42`.
- **Channel Strategy:** Per-channel independent modeling (4 score streams) to preserve spatial bearing defect localization across the four sensors.
- **Packaged Artifacts:**
  - `models/iforest_pipeline_v1.joblib`
  - `models/pca_pipeline_v1.joblib`
  - `models/model_metadata.json`
- **Inference Engine Interface:** [`src/inference.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/inference.py) (`AnomalyInferenceEngine`).

---

## 2. Intended Use & Disclaimers

### Intended Use
- **Primary Purpose:** Research and educational demonstration of an end-to-end unsupervised condition-monitoring ML pipeline built on laboratory vibration data.
- **Target Users:** Computer science students, ML practitioners, and researchers studying anomaly detection methods.
- **Input:** 1-second raw vibration snapshot of shape `(20480, 4)` recorded at 20 kHz sampling frequency across 4 bearing channels.
- **Output:** Continuous anomaly scores per channel, raw threshold exceedance flags, persistence-filtered sustained flags ($k=3$), and a Logical OR system alert indicator.

### Explicit Disclaimers & Prohibited Claims
- **NOT an Industrial Safety System:** This prototype is NOT certified for industrial machinery protection, plant operation, or safety-critical monitoring.
- **NO Failure-Time Prediction:** The model does NOT predict the exact time of machine failure.
- **NO Remaining Useful Life (RUL) Prediction:** The model does NOT estimate RUL or time-to-maintenance.
- **Anomaly $\neq$ Fault Diagnosis:** An alert indicates statistical deviation from baseline vibration patterns, NOT proof of a specific physical failure mechanism (e.g., inner race, outer race, ball, or cage defect).

---

## 3. Dataset and Provenance

- **Dataset:** NASA IMS Bearing Dataset (University of Cincinnati Center for Intelligent Maintenance Systems, hosted in the NASA Prognostics Data Repository).
- **Citation:** J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. "Bearing Data Set", NASA Prognostics Data Repository, NASA Ames Research Center.
- **Subset:** **Set 2** (984 snapshot files, 4 channels per snapshot, 20,480 samples/file at 20 kHz, recorded every 10 minutes from Feb 12, 2004 to Feb 19, 2004).
- **Outcome:** Documented outer-race failure in Bearing 1 at the conclusion of the 7-day run-to-failure experiment.
- **Test Rig Setting:** Laboratory test rig with 4 bearings mounted on a shared shaft under constant rotational speed (~2000 RPM) and radial load (6000 lbs).

---

## 4. Training, Calibration, and Evaluation Protocol

The dataset is partitioned chronologically to enforce strict data-leakage prevention:

1. **Baseline Fit Partition (Snapshots `0..159`, $N=160$, ~26.7 hours):** Assumed healthy initial period. `RobustScaler` statistics ($\text{median}$, $\text{IQR}$) and model estimators (`IsolationForest`, `PCA`) are fitted **strictly** on this partition.
2. **Calibration Partition (Snapshots `160..179`, $N=20$, ~3.3 hours):** Used strictly to compute non-parametric $P_{99}$ percentile threshold lines ($T_{c, \text{cal\_P99}}$). Zero fitting or scaling occurs on this data.
3. **Evaluation Period (Snapshots `200..983`, $N=784$, ~5.4 days):** Used to evaluate sequential detection timing and false-alarm performance.
4. **Confound Disclosure:** The evaluation partition (`200..983`) was inspected during Phase 8 prior to Phase 9 model improvement experiments (`EXP-04`..`EXP-12`). Therefore, snapshots 200..983 cannot be claimed as an untouched blind test set.

---

## 5. Feature Engineering

The model consumes a finalized **28-feature time-domain set** (7 statistical features per channel across 4 channels):

- **Per-Channel Time-Domain Features (7 stats/channel):**
  1. `mean`: Arithmetic mean (DC baseline offset)
  2. `std`: Sample standard deviation ($ddof=1$, AC dispersion)
  3. `rms`: Root Mean Square amplitude (total signal energy)
  4. `p2p`: Peak-to-peak amplitude ($\max - \min$)
  5. `skewness`: Third standardized moment (distribution asymmetry)
  6. `kurtosis`: Fisher excess kurtosis (4th moment minus 3.0)
  7. `crest_factor`: Peak-to-RMS ratio ($\max(|x|) / (\text{rms} + \epsilon)$)
- **Deterministic Feature Ordering (28 Features):**
  `ch1_mean`, `ch1_std`, `ch1_rms`, `ch1_p2p`, `ch1_skewness`, `ch1_kurtosis`, `ch1_crest_factor`,  
  `ch2_mean`, `ch2_std`, `ch2_rms`, `ch2_p2p`, `ch2_skewness`, `ch2_kurtosis`, `ch2_crest_factor`,  
  `ch3_mean`, `ch3_std`, `ch3_rms`, `ch3_p2p`, `ch3_skewness`, `ch3_kurtosis`, `ch3_crest_factor`,  
  `ch4_mean`, `ch4_std`, `ch4_rms`, `ch4_p2p`, `ch4_skewness`, `ch4_kurtosis`, `ch4_crest_factor`.

---

## 6. Preprocessing

- **DC-Offset Removal:** Raw signals undergo mean-centering per snapshot using `src/preprocessing.py` logic.
- **Feature Scaling:** `RobustScaler` (scikit-learn) is fitted strictly on baseline fit snapshots `0..159` to protect against micro-outliers without assuming Gaussian distributions.

---

## 7. Model Specifications

### A. Isolation Forest Pipeline
- **Architecture:** 4 per-channel Isolation Forests (`ch1`, `ch2`, `ch3`, `ch4`).
- **Hyperparameters:** `n_estimators=100`, `max_samples="auto"`, `contamination="auto"`, `random_state=42`.
- **Score Function:** Standardized decision score `score = 0.5 - decision_function(X_scaled)` (higher = more anomalous).

### B. PCA Reconstruction Error Pipeline
- **Architecture:** 4 per-channel PCA models (`ch1`, `ch2`, `ch3`, `ch4`).
- **Hyperparameters:** `n_components=3`, `random_state=42`.
- **Score Function:** Mean Squared Reconstruction Error `score = mean((X_scaled - X_reconstructed)^2)` (higher = more anomalous).

---

## 8. Thresholding & Score Conventions

Thresholds are non-parametric $P_{99}$ percentile cutoffs derived on calibration snapshots `160..179` ($N=20$) and read directly from `models/model_metadata.json` at runtime:

### Authoritative $P_{99}$ Calibration Thresholds

| Channel | Isolation Forest $P_{99}$ Threshold | PCA Reconstruction Error $P_{99}$ Threshold |
|---|---|---|
| **`ch1`** | `0.513168` | `0.460780` |
| **`ch2`** | `0.472510` | `0.396495` |
| **`ch3`** | `0.622558` | `0.194598` |
| **`ch4`** | `0.506628` | `0.503693` |

- **Comparison Operator:** `raw_flag = score > threshold`.

---

## 9. Decision & Alert Logic

1. **Raw Threshold Exceedance:** Evaluates whether instantaneous score $S_{c, t} > T_{c, \text{cal\_P99}}$.
2. **Stateful Persistence Filter ($k=3$):** Requires $k=3$ consecutive raw threshold exceedances before activating a channel's `sustained_flag`.
   - Under the dataset's 10-minute recording interval, $k=3$ introduces a **20-minute online confirmation delay**.
   - If any snapshot falls below threshold, the counter immediately resets to 0.
3. **Logical OR System Aggregation Policy:**
   $$\text{System Alert}_t = \bigvee_{c=1}^4 \text{Sustained Flag}_{c, t}$$
   A system alert is active if **ANY** channel achieves sustained exceedance ($k=3$), maximizing sensitivity to single-bearing localized defects.

---

## 10. Verification Evidence & Quality Audits

The packaged models and inference engine have passed complete empirical verification:

- **Unit Test Suite:** **56/56 unit tests passed** (0 failures, 0 errors) in [`src/test_inference.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/test_inference.py).
- **Numerical Reference Consistency:**
  - Isolation Forest max absolute score difference vs reference CSV: **`0.00e+00`** (Exact match).
  - PCA Reconstruction Error max absolute score difference vs reference CSV: **`1.77e-14`**.
  - Target tolerance ($\le 10^{-5}$): **Passed cleanly**.
- **Fresh-Process Artifact Smoke Test:** Verified artifact loading and scoring in a fresh Python process.
- **Top-to-Bottom Notebook Execution:** [`notebooks/08_inference_check.ipynb`](file:///d:/Projects/MachineMind%20AI/ml-service/notebooks/08_inference_check.ipynb) executed clean top-to-bottom with 0 errors.

---

## 11. Scientific & Engineering Limitations

1. **Single Failure Trajectory ($n=1$):** IMS Set 2 documents a single outer-race failure on Bearing 1. Results represent a single-case study and cannot prove general industrial reliability.
2. **Laboratory Environment:** Fixed speed (~2000 RPM) and constant load (6000 lbs). Performance under variable speed, transient loads, or harsh industrial noise is unverified.
3. **Assumed Baseline Health:** The healthy baseline (`0..159`) is an assumption. No binary healthy/faulty annotations exist in the raw dataset.
4. **Bearings Shared Shaft Coupling:** The 4 bearings share a common shaft, causing mechanical vibration coupling across channels during late-stage failure.
5. **Uncalibrated Sensor Units:** Sensor acceleration units and physical calibration constants are not specified in NASA documentation.
6. **Selection Bias Risk:** Hyperparameters and feature sets were selected in Phase 9 after inspecting Phase 8 evaluation outputs.
7. **Serialization Dependencies:** `.joblib` files rely on binary structures. Environment mismatch between Python or scikit-learn versions can cause loading issues.

---

## 12. Security Notice

> **IMPORTANT SECURITY WARNING:**  
> Joblib and pickle files execute arbitrary Python bytecode during deserialization. **Never load `.joblib` model artifacts from untrusted, unverified, or external third-party sources.** Only load model artifacts generated by your trusted build pipeline.

---

## 13. Packaged Technical Environment

| Dependency | Packaged Metadata Version |
|---|---|
| **Python** | `3.14.2` |
| **NumPy** | `2.5.3` |
| **pandas** | `3.0.6` |
| **scikit-learn** | `1.9.1` |
| **joblib** | `1.6.0` |

---

*This Model Card was generated automatically during Phase 10 Model Packaging for MachineMind AI and reflects verified empirical results.*
