# MachineMind AI — Phase 4: Canonical Signal Processing & Feature Engineering Pipeline Documentation

**Module**: `ml-service/src/canonical_feature_pipeline.py`  
**Test Suite**: `ml-service/src/test_phase4_canonical_features.py`  
**Canonical Feature Dimension**: **36 features** (9 features per channel $\times$ 4 channels)  
**Date**: October 1, 2026  

---

## 1. Executive Summary & Objective

Phase 4 establishes **ONE single, authoritative canonical feature-engineering pipeline** (`CanonicalFeaturePipeline`). 

This pipeline processes raw vibration snapshots into deterministic 36-dimensional feature vectors. It is used identically for:
1. **Offline Dataset Processing**: Processing raw NASA IMS Set 2 files for future XGBoost training.
2. **Live Streaming Inference**: Processing real-time `TelemetryRecord` objects received over HiveMQ Cloud TLS.
3. **Time-Series Persistence**: Feeding summary feature points to InfluxDB (`vibration_features`).

```
OFFLINE PATH:
Raw NASA IMS Snapshot File ──> CanonicalFeaturePipeline ──> CanonicalFeatureVector (36,) ──> [Future XGBoost Training]

ONLINE PATH:
HiveMQ Cloud (TLS) ──> TelemetryIngestionConsumer ──> TelemetryRecord ──> CanonicalFeaturePipeline ──> CanonicalFeatureVector (36,)
                                                                                                        │
                                                                                                        ├──> [Future XGBoost Inference]
                                                                                                        └──> InfluxDBWriter ('vibration_features')
```

---

## 2. Canonical Feature Schema Specification

For each of the 4 vibration channels (`ch1`, `ch2`, `ch3`, `ch4`), 9 domain-proven vibration features are calculated in deterministic order:

| Index | Feature Name Suffix | Domain | Formula / Definition | Physical & Statistical Significance |
|---|---|---|---|---|
| 1 | `mean` | Time | $\frac{1}{N} \sum x_i$ | Arithmetic baseline / DC offset. |
| 2 | `std` | Time | $\sqrt{\frac{1}{N-1} \sum (x_i - \bar{x})^2}$ | AC signal dispersion around baseline. |
| 3 | `rms` | Time | $\sqrt{\frac{1}{N} \sum x_i^2}$ | Total signal energy / RMS amplitude. |
| 4 | `p2p` | Time | $\max(x) - \min(x)$ | Peak-to-peak dynamic range. |
| 5 | `skewness` | Time | $\frac{\frac{1}{N}\sum (x_i - \bar{x})^3}{s^3}$ | Asymmetry of vibration probability density. |
| 6 | `kurtosis` | Time | $\frac{\frac{1}{N}\sum (x_i - \bar{x})^4}{s^4} - 3$ | Fisher excess kurtosis; detects early spikiness/bearing impact faulting ($\text{Gaussian}=0$). |
| 7 | `crest_factor` | Time | $\frac{\max(\lvert x \rvert)}{\text{RMS} + \epsilon}$ | Ratio of extreme peak spikes relative to overall RMS energy. |
| 8 | `spectral_energy` | Frequency | $\sum \lvert X(f_k) \rvert^2$ | Hann-windowed FFT positive frequency energy sum. |
| 9 | `spectral_centroid` | Frequency | $\frac{\sum f_k \cdot \lvert X(f_k) \rvert}{\sum \lvert X(f_k) \rvert}$ | Center of mass frequency (Hz); tracks high-frequency energy migration as bearing degrades. |

**Total Dimension**: $4 \text{ channels} \times 9 \text{ features} = \mathbf{36 \text{ features}}$.  
**Column Ordering**: Deterministic sequence (`ch1_mean`, `ch1_std`, ..., `ch4_spectral_centroid`).

---

## 3. Offline / Online Parity Design

- **Shared Core Implementation**: `process_snapshot_dataframe()` and `process_telemetry_record()` both invoke the exact same underlying mathematical utilities in `extract_time_features()` and `extract_frequency_features()`.
- **Parity Test Evidence**: `test_offline_online_parity()` verifies that extracting features from an offline DataFrame vs an online `TelemetryRecord` for the exact same raw array produces float vectors matching to relative tolerance $10^{-12}$ (`rtol=1e-12`).

---

## 4. Separation of Learned Preprocessing (Scaler Adapter)

- **`FeatureScalerAdapter`**: Wraps `RobustScaler` / `StandardScaler`.
- **Fit Rule**: `fit(X_train)` is executed strictly on the training feature matrix $(N_{\text{train}}, 36)$.
- **Transform Rule**: `transform(X)` applies fitted location/scale statistics to validation or streaming vectors without updating model parameters.
- **Leakage Prevention**: Asserts `RuntimeError` if `transform()` is called prior to `fit()`.

---

## 5. Non-Causal Data Leakage Controls

- Feature extraction for snapshot $t$ uses **strictly sample points within snapshot $t$** ($t' \le t$).
- Zero look-ahead windows, zero centered rolling filters, and zero target-derived normalized index features are used.
- Verified by `test_no_future_data_leakage()`.

---

## 6. Viva Defense Summary

> **How does MachineMind prevent feature drift between offline model training and online streaming inference?**  
> "We implemented `CanonicalFeaturePipeline` as a single authoritative module. Whether processing a raw historical file for training or a real-time MQTT `TelemetryRecord` over HiveMQ Cloud, the exact same mathematical code path is executed. Our automated parity test proves that offline and online feature vectors match with 12 decimal places of precision ($10^{-12}$), eliminating offline/online feature drift entirely."

---

## 7. Verification Evidence

- **Phase 4 Unit Tests**: `6/6 passed in 14.33s` (`ml-service/src/test_phase4_canonical_features.py`)
- **Full Workspace Test Suite**: **`51/51 passed in 18.25s`** (Backend: 16, Frontend: 11, Phase 1: 5, Phase 2: 9, Phase 3: 4, Phase 4: 6).
- **Live HiveMQ Cloud Streaming Integration Test**: 2 real NASA IMS snapshots published over HiveMQ Cloud TLS $\rightarrow$ ingested into `TelemetryRecord` $\rightarrow$ transformed into valid 36-dimensional finite feature vectors.
