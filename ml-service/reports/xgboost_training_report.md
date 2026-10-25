# XGBoost RUL Supervised Model Training & Evaluation Report

**Phase 5 Deliverable**  
**Date:** 2026-10-01 06:51 UTC  
**Model Name:** `MachineMind_XGBoost_RUL_Regressor` (v1.0.0)  
**Git Commit:** `8759bb2a2af50453b02789566f1d9aec0ae561e8`  
**Data Fingerprint:** `86babe8e29a8ab58...`

---

## 1. Dataset & Chronological Split Overview

- **Source:** Historical NASA IMS Bearing Set 2 replayed vibration telemetry.
- **Snapshot Count:** 984 snapshots ($20,480 	imes 4$ vibration samples per snapshot).
- **Feature Vector:** 36 canonical features (9 time/frequency features $	imes$ 4 channels).
- **Target:** `capped_rul_400` ($\min(983 - t, 400)$ snapshots).

### Split Configuration

| Split | Index Range | Snapshot Count | Percentage |
|---|---|---|---|
| **Training** | 0 – 600 | 601 | 61.1% |
| **Validation** | 601 – 750 | 150 | 15.2% |
| **Test (Frozen)** | 751 – 983 | 233 | 23.7% |

---

## 2. Temporal Leakage Audit Results

An automated leakage audit was executed prior to training:

- **Chronological Split Contiguity & Ordering:** PASSED
- **Feature Leakage Check (No target/sequence in features):** PASSED
- **Preprocessor Isolation (Fit strictly on train set):** PASSED
- **Causality Verification (Snapshot $t$ independent of $t > t$):** PASSED

---

## 3. Measured Model Evaluation & Baselines

All metrics are measured from actual execution. Test set was evaluated **ONCE** after configuration freeze.

### Metric Comparison Table (Target Unit: Snapshots)

| Model | Train RMSE | Train MAE | Val RMSE | Val MAE | Test RMSE | Test MAE | Test $R^2$ |
|---|---|---|---|---|---|---|---|
| **Mean Predictor Baseline** | 1.8603 | 1.198 | 101.2283 | 91.5 | 290.8831 | 283.0 | -17.7031 |
| **Linear Regression Baseline** | 1.439 | 0.6175 | 87.4788 | 79.762 | 254.3357 | 248.2499 | -13.2986 |
| **XGBoost Regressor (Selected)** | **0.085** | **0.0143** | **95.3683** | **85.0198** | **284.5865** | **276.4418** | **-16.9022** |

### Approx Time Error Conversion (Test Set)
- **XGBoost Test MAE:** 276.4418 snapshots $pprox$ 46.07 hours.
- **XGBoost Test RMSE:** 284.5865 snapshots $pprox$ 47.43 hours.

---

## 4. Top Feature Importance (Gain Metric)

> **Disclaimer:** Feature importance reflects decision-tree splitting gain and does **not** constitute physical/causal proof of bearing mechanics.

| Feature | Gain Score |
|---|---|
| `ch4_rms` | 66.1584 |
| `ch4_spectral_energy` | 57.4711 |
| `ch1_rms` | 46.4978 |
| `ch4_mean` | 40.5950 |
| `ch2_kurtosis` | 38.0147 |
| `ch3_rms` | 36.9810 |
| `ch3_p2p` | 31.5742 |
| `ch2_std` | 30.7433 |
| `ch3_std` | 25.6512 |
| `ch3_skewness` | 24.1735 |

---

## 5. Artifact Bundle & Reproducibility Command

The following artifacts have been persisted:
1. `ml-service/models/xgboost_rul_model.json` (Native XGBoost format)
2. `ml-service/models/xgboost_scaler.joblib` (Fitted RobustScaler)
3. `ml-service/models/xgboost_model_metadata.json` (Versioned metadata)

### Regeneration Command
To reproduce the feature matrix, leakage audit, training, and report generation from raw data:
```bash
python -m src.train_xgboost
```
