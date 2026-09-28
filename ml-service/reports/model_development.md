# Phase 7 Report: Machine Learning Model Development

**Project:** MachineMind AI  
**Dataset:** NASA IMS Bearing Dataset (Subset: Set 2, 984 snapshots)  
**Task:** Unsupervised Anomaly Detection  
**Phase:** 7 of 11 — ML Model Development  
**Author:** MachineMind AI Team  
**Date:** September 29, 2026  

---

## 1. Executive Summary

In Phase 7, we developed and evaluated two justified, reproducible, unsupervised machine learning candidate model architectures: **Isolation Forest** (tree-partitioning based) and **PCA Reconstruction Error** (linear subspace / distance-based). Models were trained using a **Per-Channel Strategy** across four vibration channels (`ch1`, `ch2`, `ch3`, `ch4`) monitoring four bearings mounted on a single shared rotating shaft.

### Key Accomplishments
1. **Leakage-Safe Baseline Scoping**: Scalers (`RobustScaler`) and estimator models were bundled into unified pipelines and fit strictly on baseline-fit snapshots (`0..159`, $N=160$, ~26.7 hours).
2. **Score Normalization & Positive Unification**: All model outputs were standardized to guarantee `HIGHER SCORE = MORE ANOMALOUS` with non-negative score distributions.
3. **Leakage-Free Threshold Calibration**: Exploratory reference thresholds were computed exclusively on the validation period (`160..199`, $N=40$, ~6.7 hours) using parametric ($\mu + 3\sigma$) and non-parametric percentile ($P_{99}$) rules.
4. **Reproducibility & Unit Testing**: Fixed random seeds (`random_state=42`) were recorded, and all 4 automated unit tests in `src/test_models.py` passed cleanly (verifying baseline scoping isolation, seed reproducibility, zero NaNs, non-negativity, and synthetic anomaly injection detection).

---

## 2. Problem Framing & Algorithm Justification

### 2.1 Problem Framing: Novelty Detection
- **Outlier Detection**: Assumes training data contains a mixture of normal and anomalous samples.
- **Novelty Detection**: Assumes training data consists **strictly of assumed normal operating data**. The model learns the boundary of baseline behavior during fitting. Any new sample deviating from this boundary is flagged as a statistical novelty (anomaly).
- **Why Novelty Detection**: In condition monitoring, we train strictly on an initial period assumed to be healthy (snapshots `0..159`). Per-snapshot health labels do not exist during fitting, nor do we want degraded operational points contaminating our baseline reference.

### 2.2 Selected Model Candidates

| Model Architecture | Algorithm Family | Intuition & Mechanism | Key Hyperparameters |
|---|---|---|---|
| **Isolation Forest (iForest)** | Tree-based / Partitioning | Isolates observations by randomly splitting features. Outliers require fewer splits (shorter tree path length) because they lie in low-density feature regions. | `n_estimators=100`, `max_samples='auto'`, `contamination='auto'`, `random_state=42` |
| **PCA Reconstruction Error** | Linear Subspace / Distance | Projects normal feature vectors onto top $k=2$ principal components and reconstructs back to original space. Deviating signals produce high Mean Squared Reconstruction Error $\|X - \hat{X}\|_2^2$. | `n_components=2`, `whiten=False`, `random_state=42` |

### 2.3 Excluded Architectures (Scope Boundaries)
- **Deep Learning (Autoencoders, LSTMs, VAEs)**: Excluded due to small training set ($N=160$), risk of overfitting, lack of feature interpretability, extra framework overhead (TensorFlow/PyTorch), and higher computational complexity.
- **Outlier Detection Algorithms (e.g. LOF in fit-predict mode)**: Excluded in favor of explicit novelty-mode pipelines capable of scoring new unseen streaming snapshots.

---

## 3. Data Preprocessing & Pipeline Scoping

### 3.1 Feature Representation & Sensor Arrangement
The test rig consists of four bearings mounted on a single shared rotating shaft, monitored by four vibration accelerometer channels (`ch1` to `ch4`). Each channel contains 9 numerical features extracted in Phase 5:
- **Time-Domain (7)**: `mean`, `std`, `rms`, `p2p`, `skewness`, `kurtosis`, `crest_factor`
- **Spectral-Domain (2)**: `spectral_energy`, `spectral_centroid`

### 3.2 Robust Scaling & Pipeline Bundling
Distance and subspace methods (PCA) are sensitive to feature scales. `RobustScaler` scales features using median and Interquartile Range ($IQR = Q_3 - Q_1$):
$$z_{\text{robust}} = \frac{x - \text{median}}{\text{IQR}}$$

**Leakage Safeguard**: Scaler parameters are calculated **strictly from the baseline fit period (snapshots 0..159)**. Bundling `RobustScaler` and the estimator inside `AnomalyDetectionPipeline` guarantees that scale parameters remain frozen during inference on validation and evaluation snapshots.

---

## 4. Score Standardization & Threshold Calibration

### 4.1 Score Standardization (`Higher = More Anomalous`)
- **Isolation Forest**: scikit-learn `decision_function` returns positive values for normal inliers ($\sim +0.05 \text{ to } +0.4$) and negative values for outliers ($< 0$). We map raw outputs via:
  $$S_{\text{iForest}} = 0.5 - \text{decision\_function}(X)$$
  This guarantees normal inliers score $\sim 0.1 \text{ to } 0.45$ (low) while anomalous points score $> 0.5$ (high).
- **PCA Reconstruction Error**: Naturally computes non-negative Mean Squared Reconstruction Error across feature dimensions:
  $$S_{\text{PCA}} = \frac{1}{d} \sum_{j=1}^d (x_j - \hat{x}_j)^2$$

### 4.2 Validation Threshold Calibration (Snapshots 160..199)
Exploratory reference thresholds are computed exclusively on validation period scores ($N=40$ snapshots, `160..199`):

| Vibration Channel | Model Architecture | Validation Mean ($\mu_{\text{val}}$) | Validation Std ($\sigma_{\text{val}}$) | $\mu + 3\sigma$ Threshold | $P_{99}$ Threshold | Validation Max |
|---|---|---|---|---|---|---|
| **Channel 1 (`ch1`)** | Isolation Forest | 0.4382 | 0.0447 | 0.5724 | **0.5310** | 0.5339 |
| **Channel 1 (`ch1`)** | PCA Reconstruction Error | 0.2866 | 0.2733 | 1.1063 | **1.0721** | 1.1195 |
| **Channel 2 (`ch2`)** | Isolation Forest | 0.4171 | 0.0362 | 0.5258 | **0.5184** | 0.5381 |
| **Channel 2 (`ch2`)** | PCA Reconstruction Error | 0.2629 | 0.1934 | 0.8431 | **0.8657** | 0.9148 |
| **Channel 3 (`ch3`)** | Isolation Forest | 0.4825 | 0.0612 | 0.6660 | **0.5990** | 0.6069 |
| **Channel 3 (`ch3`)** | PCA Reconstruction Error | 0.2246 | 0.1625 | 0.7122 | **0.6793** | 0.7005 |
| **Channel 4 (`ch4`)** | Isolation Forest | 0.4337 | 0.0669 | 0.6344 | **0.6247** | 0.6498 |
| **Channel 4 (`ch4`)** | PCA Reconstruction Error | 0.4089 | 0.3270 | 1.3899 | **1.3281** | 1.4608 |

### 4.3 Statistical Limitations of Validation Thresholds
- **Sample Size Constraint**: The validation partition contains only $N=40$ snapshots. Percentile estimates ($P_{99}$) derived from 40 samples rely on linear interpolation between the top ranked validation values and have inherent sampling variance.
- **Exploratory Reference Indicators**: These thresholds serve as preliminary, exploratory reference indicators for quantitative comparison in Phase 8. They are **not operationally validated industrial alarm thresholds**, and no health state ground-truth is implied by a threshold exceedance.

---

## 5. Visual Comparative Analysis

### 5.1 Channel 1 Score Timelines & Initial Transient
![Bearing 1 Model Scores Comparison](figures/models/model_scores_ch1_comparison.png)

1. **Phase 6 Baseline RMS Z-Score**: Remains stable near $\sim 0.92$ during fit and validation, exhibiting sustained elevation starting around snapshot ~531-585, reaching a peak Z-score $> 12.0$ near recording end.
2. **Isolation Forest (`ch1`)**:
   - **Initial Snapshot Transient**: At snapshot index 0, `score_iforest_ch1` records a transient peak value of **0.7587**, which exceeds the validation $P_{99}$ reference threshold (**0.5310**). Consequently, scores during the `0..199` fit/validation period are **not consistently below threshold**.
   - **Baseline & Validation Behavior**: Across the remainder of the baseline fit (`0..159`) and validation (`160..199`) periods, scores average $\sim 0.4382$.
   - **Sequential Evaluation**: Sustained score elevation (5 consecutive snapshots exceeding validation $P_{99}$) begins at **snapshot index 532**, elevating to $> 0.60$ as recording progresses.
3. **PCA Reconstruction Error (`ch1`)**:
   - Averages $\sim 0.2866$ during fit and validation. Exhibits sustained elevation exceeding validation $P_{99}$ (**1.0721**) starting at **snapshot index 532**, reaching MSE $> 100.0$ near recording end.

### 5.2 Multi-Channel Sensor Observations across Shared Shaft
![Isolation Forest All Channels](figures/models/model_scores_all_channels_iforest.png)
![PCA Error All Channels](figures/models/model_scores_all_channels_pca.png)

#### Observed Score Behaviors
- **Channel 1 (`ch1`)**: Exhibits early, sustained anomaly score progression with 460 iForest exceedances and 453 PCA exceedances out of 784 sequential evaluation snapshots. First sustained exceedance starts at snapshot **532**.
- **Channel 2 (`ch2`)**: Exhibits sustained Isolation Forest threshold exceedance starting at snapshot **347** (426 total evaluation exceedances). PCA Reconstruction Error for `ch2` exhibits sustained exceedance starting at snapshot **477** (381 total evaluation exceedances).
- **Channels 3 & 4 (`ch3`, `ch4`)**: PCA Reconstruction Error exhibits sustained exceedances starting at snapshot **438** (`ch3`, 324 exceedances) and snapshot **659** (`ch4`, 332 exceedances). Isolation Forest scores for `ch3` and `ch4` remain below threshold throughout most of the experiment, exhibiting late sustained exceedances at snapshot **891** (`ch3`) and **902** (`ch4`).

#### Interpretation & Scientific Caveats
- **Cross-Channel Vibration Transfer Hypothesis**: The four sensor channels monitor four bearings mounted on a single shared rotating shaft. The observed score elevations in channels 2, 3, and 4 in the second half of the recording represent plausible mechanical vibration transfer across the shared shaft structure.
- **Strict Limitation**: Cross-channel vibration transfer is a physical hypothesis consistent with the rig architecture, **not a causally established fact**. No per-file physical fault diagnosis or defect localization is claimed.

---

## 6. Empirical Verification & Unit Test Suite

Execution of `src/test_models.py` verified the following requirements:
1. `test_fit_scoping_isolation`: `PASSED` — Confirmed `RobustScaler` and models saw strictly $N=160$ baseline-fit samples.
2. `test_seed_reproducibility`: `PASSED` — Confirmed identical output vectors across multiple seeded runs (`random_state=42`).
3. `test_scores_validity_and_sign_convention`: `PASSED` — Confirmed 0 NaNs/Infs, valid array shapes ($984 \times 12$), and strictly non-negative scores.
4. `test_synthetic_anomaly_detection`: `PASSED` — Confirmed models reliably detect synthetic amplitude/mean shift anomalies injected into baseline data.

---

## 7. Project Limitations & Scientific Scope

1. **Assumed Healthy Baseline**: The initial partition (snapshots `0..159`) is assumed to represent normal operation based on early recording status. No verified healthy-state ground-truth labels exist.
2. **Single Degradation Trajectory**: IMS Set 2 documents a single run-to-failure experiment. Model results represent a single-case study on laboratory data and cannot prove industrial reliability or generalize to other operating conditions.
3. **No Failure-Time or RUL Prediction**: Model scores indicate statistical deviation from the baseline fit period. The models do **not** predict remaining useful life (RUL) or exact failure timestamps.
4. **PCA Subspace Rank**: PCA reconstruction error sensitivity depends on component rank ($k=2$). Formal hyperparameter tuning will be evaluated in Phase 9.

---

## 8. Deliverables Table

| File Path | Description / Purpose |
|---|---|
| [src/models.py](file:///d:/Projects/MachineMind%20AI/ml-service/src/models.py) | Reusable `AnomalyDetectionPipeline`, `PCAReconstructionModel`, thresholding logic, and per-channel model runner. |
| [src/test_models.py](file:///d:/Projects/MachineMind%20AI/ml-service/src/test_models.py) | Unit test suite verifying scoping, reproducibility, score non-negativity, and synthetic anomaly detection. |
| [notebooks/05_model_development.ipynb](file:///d:/Projects/MachineMind%20AI/ml-service/notebooks/05_model_development.ipynb) | Interactive notebook performing model training, thresholding, and visual evaluation. |
| [data/processed/model_scores_set2.csv](file:///d:/Projects/MachineMind%20AI/ml-service/data/processed/model_scores_set2.csv) | Saved dataset containing computed anomaly scores for all channels across models. |
| [reports/figures/models/](file:///d:/Projects/MachineMind%20AI/ml-service/reports/figures/models/) | Directory containing comparative visualization plots. |
