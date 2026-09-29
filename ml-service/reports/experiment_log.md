# Phase 9: Model Improvement — Controlled Experiment Log

**Project:** MachineMind AI  
**Dataset:** NASA IMS Bearing Dataset (Set 2, 984 snapshots)  
**Task:** Unsupervised Vibration Anomaly Detection  
**Document Type:** Master Experiment Log & Audit Record  
**Author:** MachineMind AI Team & Mentor  
**Date:** September 30, 2026  

---

## 1. Executive Overview & Experiment Governance

This log documents controlled, one-factor-at-a-time (OFAT) experiments conducted during **Phase 9: Model Improvement**. 

### Strict Methodological Boundaries
1. **Chronological Partitions:**
   - **Fit Partition:** Snapshots `0..159` ($N=160$, ~26.7 hrs). Model fitting, parameter estimation ($\mu, \sigma$), and scaler transformation (`RobustScaler`).
   - **Calibration Partition:** Snapshots `160..179` ($N=20$, ~3.3 hrs). Derivation of threshold reference lines ($P_{99}$, $T_{\text{cal}}$) and calibration dispersion ($\sigma_{\text{cal}}$).
   - **Diagnostic Comparison Window:** Snapshots `180..199` ($N=20$, ~3.3 hrs). Model selection and comparative metric assessment.
   - **Evaluation Partition:** Snapshots `200..983` ($N=784$, ~130.7 hrs). Held out completely from hyperparameter tuning and model selection.
2. **Selection Bias Safeguard:** Snapshots `180..199` represent a small diagnostic window ($N=20$) subject to repeated evaluation. Repeated comparisons introduce selection bias; results are treated as comparative diagnostic evidence rather than definitive statistical validation.

---

## 2. Historical Baseline Log (Phase 6 – Phase 8)

| ID | Phase | Experiment Description | Configuration (Features, Scaler, Estimator, Seed) | Partitions (Fit / Val / Eval) | Key Result Summary | Decision / Status |
|---|---|---|---|---|---|---|
| **EXP-01** | Phase 6 | Non-ML Statistical Baseline Scoring | 36 features, Composite RMS Z-Score (`score_rms_z`) | Fit: `0..159`<br>Ref: `160..199`<br>Seq: `200..983` | Baseline score mean=0.92 (fit), 0.97 (ref eval). Sustained elevation above $T=3.0$ starts at snapshot 578. | Reference Baseline Control |
| **EXP-02** | Phase 7 | Per-Channel iForest & PCA Development | 36 features (9/ch), `RobustScaler`, `iForest` ($n_{\text{est}}=100$) / `PCA` ($k=2$), seed=42 | Fit: `0..159`<br>Val: `160..199`<br>Eval: `200..983` | Both models track degradation trajectory on Bearing 1 (`ch1`) starting at snapshot 532; `ch2` shows early elevation due to shaft coupling. | ML Model Benchmark |
| **EXP-03** | Phase 8 | Persistence Filtering & Evaluation Protocol | 36 features, `iForest`/`PCA`, validation $P_{99}$ thresholds, $k=1..5$ | Fit: `0..159`<br>Val: `160..199`<br>Eval: `200..983` | Persistence $k=3$ eliminates 100% of reference false alarms ($R_{\text{FA}}=0.0\%$). Confirms `ch1` alert at snapshot 534 (3.1 days lead time). | Frozen Evaluation Protocol |

---

## 3. Phase 9 Experiment Execution Log

### EXP-04: Feature Set Ablation

- **Date:** September 30, 2026
- **Status:** **Completed & Audited**
- **Single Factor Under Test:** Feature Set Composition (Full 36 Features vs. Time-Domain 28 Features).

#### 1. Hypothesis
Restricting feature extraction to time-domain statistical indicators (28 features across 4 channels) removes high-frequency FFT windowing noise and spectral variance while maintaining equal or superior false-alarm suppression on the diagnostic comparison window compared to the full 36-feature set.

#### 2. Experimental Configurations
- **Control Configuration (Full Feature Set - 36 Features):**  
  9 features per channel $\times$ 4 channels: `mean`, `std`, `rms`, `p2p`, `skewness`, `kurtosis`, `crest_factor`, `spectral_energy`, `spectral_centroid`.
- **Experimental Configuration (Time-Domain Subset - 28 Features):**  
  7 time-domain features per channel $\times$ 4 channels: `mean`, `std`, `rms`, `p2p`, `skewness`, `kurtosis`, `crest_factor`. Excludes frequency-domain features `spectral_energy` and `spectral_centroid`.
- **Fixed Parameters Across Both Runs:**
  - Scaler: `RobustScaler()` fit strictly on snapshots `0..159`.
  - Estimator: `IsolationForest(n_estimators=100, max_samples="auto", contamination="auto", random_state=42)` per channel.
  - Threshold Calibration Rule: Validation $P_{99}$ derived strictly on calibration snapshots `160..179` ($N=20$).
  - Persistence Filter: $k=3$ consecutive exceedances.
  - Partitions: Fit `0..159` ($N=160$), Calibration `160..179` ($N=20$), Diagnostic Comparison `180..199` ($N=20$).

#### 3. Metric Definitions
- **Primary Decision Metric:** Sustained diagnostic false-alarm snapshot count ($N_{\text{FA, sustained\_snapshots\_diag}}$) under $k=3$ on snapshots `180..199` ($N=20$ per channel, total $N=80$ evaluations across 4 channels).
- **Secondary Decision Metric:** Raw diagnostic false-alarm snapshot count ($N_{\text{FA, raw\_snapshots\_diag}}$) under $k=1$ on snapshots `180..199` ($N=20$ per channel, total $N=80$).
- **Descriptive Statistics:** Calibration percentile threshold ($T_{\text{cal\_P99}}$) and calibration standard deviation ($\sigma_{\text{cal}}$) on snapshots `160..179`.

#### 4. Detailed Empirical Results

| Channel / Configuration | Features ($N_{\text{feat}}$) | Calibration Std Dev ($\sigma_{\text{cal}}$) | Calib $P_{99}$ Threshold ($T_{\text{cal}}$) | Diagnostic Raw FA ($N_{\text{FA, raw\_diag}}$) | Diagnostic Sustained FA ($N_{\text{FA, sust\_diag}}$, $k=3$) |
|---|---|---|---|---|---|
| **Control: Full Feature Set (36)** | | | | | |
| - Channel 1 (`ch1`) | 9 | 0.038366 | 0.492242 | 4 / 20 | **0 / 20** |
| - Channel 2 (`ch2`) | 9 | 0.026355 | 0.451808 | 5 / 20 | **0 / 20** |
| - Channel 3 (`ch3`) | 9 | 0.059908 | 0.603034 | 0 / 20 | **0 / 20** |
| - Channel 4 (`ch4`) | 9 | 0.056255 | 0.529307 | 3 / 20 | **0 / 20** |
| **Control Aggregate Total** | **36** | — | — | **12 / 80 (15.0%)** | **0 / 80 (0.0%)** |
| | | | | | |
| **Experimental: Time-Domain Subset (28)** | | | | | |
| - Channel 1 (`ch1`) | 7 | 0.042077 | 0.513168 | 3 / 20 | **0 / 20** |
| - Channel 2 (`ch2`) | 7 | 0.034818 | 0.472510 | 2 / 20 | **0 / 20** |
| - Channel 3 (`ch3`) | 7 | 0.062475 | 0.622558 | 0 / 20 | **0 / 20** |
| - Channel 4 (`ch4`) | 7 | 0.054877 | 0.506628 | 4 / 20 | **0 / 20** |
| **Experimental Aggregate Total** | **28** | — | — | **9 / 80 (11.25%)** | **0 / 80 (0.0%)** |

#### 5. Decision Rule Analysis & Conclusion
1. **Primary Metric Comparison:** Both configurations achieved **0 sustained false alarms** ($k=3$) on the diagnostic window `180..199` ($0 / 80 = 0.0\%$).
2. **Secondary Metric Comparison:** The 28-feature time-domain subset reduced raw snapshot false alarms from **12 down to 9** across the 4 channels (a 25.0% reduction in raw noise sensitivity).
3. **Decision & Protocol Alignment:** Following the approved decision rule, because the primary metric is tied at 0, the secondary metric favors the **28-feature time-domain subset** for higher parsimony and lower raw noise sensitivity. Both configurations are logged, and the 36-feature set remains the primary baseline control.

#### 6. Verification & Audit Sign-Off
- **Reproducibility:** Confirmed reproducibility under `random_state=42`. All 36 unit tests in `src/` passed cleanly (0 failures, 0 errors).
- **Data Integrity:** Verified 0 NaNs, 0 Infs in feature matrix and score outputs.
- **Data Partitions:** Verified strict separation (Fit: 0–159, Cal: 160–179, Diag: 180–199, Eval: 200–983).

---

### EXP-05: Feature Scaling Method Comparison

- **Date:** September 30, 2026
- **Status:** **Completed & Audited**
- **Single Factor Under Test:** Feature Scaling Strategy (`RobustScaler` vs. `StandardScaler`).

#### 1. Hypothesis
Comparing `RobustScaler` (median/IQR scaling) against `StandardScaler` (mean/std scaling) tests whether standardizing variance rather than robust quantiles reduces false alarms on the diagnostic comparison window. Because Isolation Forest splits depend on relative feature rank order rather than scale, affine monotonic feature transformations fitted on identical baseline data are expected to yield identical anomaly scores and diagnostic false alarm metrics.

#### 2. Experimental Configurations
- **Control Configuration A (`RobustScaler`):**  
  `RobustScaler()` fit strictly on baseline fit snapshots `0..159`.
- **Candidate Configuration B (`StandardScaler`):**  
  `StandardScaler()` fit strictly on baseline fit snapshots `0..159`.
- **Fixed Parameters Across Both Runs:**
  - Feature Set: Primary 36-feature set (and reference 28-feature time-domain set).
  - Estimator: `IsolationForest(n_estimators=100, max_samples="auto", contamination="auto", random_state=42)` per channel.
  - Threshold Calibration Rule: Validation $P_{99}$ derived strictly on calibration snapshots `160..179` ($N=20$).
  - Persistence Filter: $k=3$ consecutive exceedances.
  - Partitions: Fit `0..159` ($N=160$), Calibration `160..179` ($N=20$), Diagnostic Comparison `180..199` ($N=20$).

#### 3. Metric Definitions
- **Primary Decision Metric:** Sustained diagnostic false-alarm snapshot count ($N_{\text{FA, sustained\_snapshots\_diag}}$) under $k=3$ on snapshots `180..199` ($N=20$ per channel, total $N=80$ evaluations across 4 channels).
- **Secondary Decision Metric:** Raw diagnostic false-alarm snapshot count ($N_{\text{FA, raw\_snapshots\_diag}}$) under $k=1$ on snapshots `180..199` ($N=20$ per channel, total $N=80$).
- **Descriptive Statistics:** Calibration mean ($Mean_{\text{cal}}$), calibration standard deviation ($\sigma_{\text{cal}}$), and $P_{99}$ threshold ($T_{\text{cal\_P99}}$) on snapshots `160..179`.

#### 4. Detailed Empirical Results

| Feature Set / Scaling Strategy | Channel | Calib Mean ($Mean_{\text{cal}}$) | Calib Std Dev ($\sigma_{\text{cal}}$) | Calib $P_{99}$ Threshold ($T_{\text{cal}}$) | Diagnostic Raw FA ($N_{\text{FA, raw\_diag}}$) | Diagnostic Sustained FA ($N_{\text{FA, sust\_diag}}$, $k=3$) |
|---|---|---|---|---|---|---|
| **36-Feature Baseline** | | | | | | |
| **RobustScaler (Control)** | `ch1` | 0.434546 | 0.038366 | 0.492242 | 4 / 20 | **0 / 20** |
| | `ch2` | 0.409019 | 0.026355 | 0.451808 | 5 / 20 | **0 / 20** |
| | `ch3` | 0.487942 | 0.059908 | 0.603034 | 0 / 20 | **0 / 20** |
| | `ch4` | 0.416732 | 0.056255 | 0.529307 | 3 / 20 | **0 / 20** |
| **RobustScaler 36-Feat Total** | **All 4** | — | — | — | **12 / 80 (15.0%)** | **0 / 80 (0.0%)** |
| | | | | | | |
| **StandardScaler (Candidate)** | `ch1` | 0.434546 | 0.038366 | 0.492242 | 4 / 20 | **0 / 20** |
| | `ch2` | 0.409019 | 0.026355 | 0.451808 | 5 / 20 | **0 / 20** |
| | `ch3` | 0.487942 | 0.059908 | 0.603034 | 0 / 20 | **0 / 20** |
| | `ch4` | 0.416732 | 0.056255 | 0.529307 | 3 / 20 | **0 / 20** |
| **StandardScaler 36-Feat Total** | **All 4** | — | — | — | **12 / 80 (15.0%)** | **0 / 80 (0.0%)** |
| | | | | | | |
| **28-Feature Reference** | | | | | | |
| **RobustScaler (Control)** | **All 4** | — | — | — | **9 / 80 (11.25%)** | **0 / 80 (0.0%)** |
| **StandardScaler (Candidate)** | **All 4** | — | — | — | **9 / 80 (11.25%)** | **0 / 80 (0.0%)** |

#### 5. Decision Rule Application & Conclusion
1. **Decision Rule Criteria:** The approved Phase 9 decision rule specifies that `RobustScaler` remains the baseline control. `StandardScaler` is accepted ONLY if it reduces diagnostic false-alarm metrics on snapshots `180..199`. Lower calibration standard deviation ($\sigma_{\text{cal}}$) is descriptive only and cannot be used alone as proof of better model quality.
2. **Empirical Outcome:** `StandardScaler` and `RobustScaler` produce **identical diagnostic metrics** across all channels ($N_{\text{FA, sustained}} = 0$, $N_{\text{FA, raw}} = 12$ on 36 features; $N_{\text{FA, sustained}} = 0$, $N_{\text{FA, raw}} = 9$ on 28 features).
3. **Decision Outcome:** Because `StandardScaler` does not reduce diagnostic false alarms, **RobustScaler is retained as the baseline control** for consistency and parsimony.

#### 6. Verification & Audit Sign-Off
- **Reproducibility:** Confirmed reproducibility under `random_state=42`. All 36 unit tests passed cleanly.
- **Data Integrity:** 0 NaNs, 0 Infs detected.
- **Data Partitions:** Verified strict partition bounds (Fit: 0–159, Cal: 160–179, Diag: 180–199, Eval: 200–983).

---

### EXP-06: PCA Component Count Comparison

- **Date:** September 30, 2026
- **Status:** **Completed & Audited**
- **Single Factor Under Test:** PCA Reconstruction Component Count ($k=1$, $k=2$, $k=3$).

#### 1. Hypothesis
Testing PCA component counts ($k \in \{1, 2, 3\}$) evaluates the trade-off between baseline reconstruction fidelity and noise sensitivity. Increasing component count $k$ retains more variance in the baseline subspace, reducing baseline reconstruction error ($\text{MSE}_{\text{cal}}$) and diagnostic false alarms, provided the additional components represent normal system dynamics rather than high-frequency noise.

#### 2. Experimental Configurations
- **Control Configuration A ($k=2$):**  
  `PCA(n_components=2, random_state=42)` bundled with `RobustScaler()` fit strictly on snapshots `0..159`.
- **Candidate Configuration B ($k=1$):**  
  `PCA(n_components=1, random_state=42)` bundled with `RobustScaler()` fit strictly on snapshots `0..159`.
- **Candidate Configuration C ($k=3$):**  
  `PCA(n_components=3, random_state=42)` bundled with `RobustScaler()` fit strictly on snapshots `0..159`.
- **Fixed Parameters Across All Runs:**
  - Feature Set: Primary 36-feature set (and reference 28-feature time-domain set).
  - Preprocessing: `RobustScaler()` fit strictly on `0..159`.
  - Threshold Calibration Rule: Validation $P_{99}$ derived strictly on calibration snapshots `160..179` ($N=20$).
  - Persistence Filter: $k=3$ consecutive exceedances.
  - Partitions: Fit `0..159` ($N=160$), Calibration `160..179` ($N=20$), Diagnostic Comparison `180..199` ($N=20$).

#### 3. Metric Definitions
- **Primary Decision Metric:** Sustained diagnostic false-alarm snapshot count ($N_{\text{FA, sustained\_snapshots\_diag}}$) under $k=3$ on snapshots `180..199` ($N=20$ per channel, total $N=80$ evaluations across 4 channels).
- **Secondary Decision Metric:** Raw diagnostic false-alarm snapshot count ($N_{\text{FA, raw\_snapshots\_diag}}$) under $k=1$ on snapshots `180..199` ($N=20$ per channel, total $N=80$).
- **Tertiary Tie-Breaker Metric:** Average Calibration Reconstruction MSE ($\text{MSE}_{\text{cal}}$) on snapshots `160..179` ($N=20$).

#### 4. Detailed Empirical Results

| Feature Set / PCA Component Count | Channel | Calib Mean MSE ($\text{MSE}_{\text{cal}}$) | Calib Std Dev ($\sigma_{\text{cal}}$) | Calib $P_{99}$ Threshold ($T_{\text{cal}}$) | Diagnostic Raw FA ($N_{\text{FA, raw\_diag}}$) | Diagnostic Sustained FA ($N_{\text{FA, sust\_diag}}$, $k=3$) |
|---|---|---|---|---|---|---|
| **36-Feature Baseline** | | | | | | |
| **PCA $k=1$** | `ch1` | 0.537694 | 0.359057 | 1.318714 | 2 / 20 | **0 / 20** |
| | `ch2` | 0.432950 | 0.226950 | 0.980925 | 2 / 20 | **0 / 20** |
| | `ch3` | 0.565686 | 0.320444 | 1.464737 | 1 / 20 | **0 / 20** |
| | `ch4` | 0.403622 | 0.330745 | 1.118542 | 1 / 20 | **0 / 20** |
| **PCA $k=1$ 36-Feat Total** | **All 4** | **0.484988** | — | — | **6 / 80 (7.50%)** | **0 / 80 (0.0%)** |
| | | | | | | |
| **PCA $k=2$ (Control)** | `ch1` | 0.231314 | 0.151276 | 0.589649 | 4 / 20 | **0 / 20** |
| | `ch2` | 0.239878 | 0.210733 | 0.850215 | 0 / 20 | **0 / 20** |
| | `ch3` | 0.227658 | 0.141821 | 0.651673 | 0 / 20 | **0 / 20** |
| | `ch4` | 0.333426 | 0.320328 | 1.107877 | 1 / 20 | **0 / 20** |
| **PCA $k=2$ 36-Feat Total** | **All 4** | **0.258069** | — | — | **5 / 80 (6.25%)** | **0 / 80 (0.0%)** |
| | | | | | | |
| **PCA $k=3$ (Candidate)** | `ch1` | 0.161570 | 0.120567 | 0.424097 | 2 / 20 | **0 / 20** |
| | `ch2` | 0.178972 | 0.161410 | 0.588735 | 0 / 20 | **0 / 20** |
| | `ch3` | 0.100413 | 0.083682 | 0.357422 | 0 / 20 | **0 / 20** |
| | `ch4` | 0.169107 | 0.150860 | 0.569700 | 1 / 20 | **0 / 20** |
| **PCA $k=3$ 36-Feat Total** | **All 4** | **0.152516** | — | — | **3 / 80 (3.75%)** | **0 / 80 (0.0%)** |
| | | | | | | |
| **28-Feature Reference** | | | | | | |
| **PCA $k=1$** | **All 4** | **0.472132** | — | — | **5 / 80 (6.25%)** | **0 / 80 (0.0%)** |
| **PCA $k=2$ (Control)** | **All 4** | **0.234346** | — | — | **5 / 80 (6.25%)** | **0 / 80 (0.0%)** |
| **PCA $k=3$ (Candidate)** | **All 4** | **0.104339** | — | — | **4 / 80 (5.00%)** | **0 / 80 (0.0%)** |

#### 5. Decision Rule Application & Conclusion
1. **Primary Metric Comparison ($N_{\text{FA, sustained\_snapshots\_diag}}$, $k=3$):** All configurations ($k=1, 2, 3$) achieved **0 sustained false alarms** on snapshots `180..199` ($0 / 80 = 0.0\%$).
2. **Secondary Metric Comparison ($N_{\text{FA, raw\_snapshots\_diag}}$, $k=1$):** $k=3$ reduced raw false-alarm snapshots from 5 ($k=2$) down to **3** on the 36-feature set (and from 5 to 4 on the 28-feature set).
3. **Tertiary Metric Comparison ($\text{MSE}_{\text{cal}}$):** $k=3$ achieved the lowest average reconstruction error during calibration ($\text{MSE}_{\text{cal}} = 0.152516$ vs $0.258069$ for $k=2$).
4. **Decision Outcome:** Following the approved decision rule, **$k=3$ is accepted as the improved PCA component count**, while $k=2$ remains documented as the Phase 8 baseline control.

#### 6. Verification & Audit Sign-Off
- **Reproducibility:** Confirmed reproducibility under `random_state=42`. All 36 unit tests passed cleanly.
- **Data Integrity:** 0 NaNs, 0 Infs detected.
- **Data Partitions:** Verified strict partition bounds (Fit: 0–159, Cal: 160–179, Diag: 180–199, Eval: 200–983).

---

### EXP-07: Isolation Forest Hyperparameter Comparison

- **Date:** September 30, 2026
- **Status:** **Completed & Audited**
- **Single Factor Under Test:** Isolation Forest Hyperparameters (`n_estimators` $\in \{50, 100, 200\}$, `max_samples` $\in \{0.5, 1.0, \text{"auto"}\}$).

#### 1. Hypothesis
Grid searching `n_estimators` ($50, 100, 200$) and `max_samples` ($0.5, 1.0, \text{"auto"}$) evaluates whether tuning tree ensemble depth and sub-sampling size reduces diagnostic false alarms on snapshots `180..199` while keeping scaling and partitioning fixed.

#### 2. Experimental Configurations
- **Baseline Control Configuration:** `n_estimators=100`, `max_samples="auto"`.
- **Hyperparameter Grid (9 combinations per feature set):**
  - `n_estimators`: $50, 100, 200$
  - `max_samples`: $0.5, 1.0, \text{"auto"}$
- **Fixed Parameters Across All Runs:**
  - Feature Sets: Primary 36-feature set (and reference 28-feature time-domain set).
  - Preprocessing: `RobustScaler()` fit strictly on `0..159`.
  - Threshold Calibration Rule: Validation $P_{99}$ derived strictly on calibration snapshots `160..179` ($N=20$).
  - Persistence Filter: $k=3$ consecutive exceedances.
  - Partitions: Fit `0..159` ($N=160$), Calibration `160..179` ($N=20$), Diagnostic Comparison `180..199` ($N=20$).

#### 3. Metric Definitions
- **Primary Decision Metric:** Sustained diagnostic false-alarm snapshot count ($N_{\text{FA, sustained\_snapshots\_diag}}$) under $k=3$ on snapshots `180..199` ($N=20$ per channel, total $N=80$ evaluations across 4 channels).
- **Secondary Decision Metric:** Raw diagnostic false-alarm snapshot count ($N_{\text{FA, raw\_snapshots\_diag}}$) under $k=1$ on snapshots `180..199` ($N=20$ per channel, total $N=80$).
- **Computational Metric:** Pipeline fit & score runtime (milliseconds across 4 channels).

#### 4. Detailed Empirical Results

| Feature Set | `n_estimators` | `max_samples` | Calib Mean Score ($Mean_{\text{cal}}$) | Calib $P_{99}$ Threshold ($T_{\text{cal}}$) | Diag Raw FA ($N_{\text{FA, raw\_diag}}$) | Diag Sustained FA ($N_{\text{FA, sust\_diag}}$, $k=3$) | Runtime (ms) |
|---|---|---|---|---|---|---|---|
| **36-Feature Baseline** | 50 | 0.5 | 0.4641 | 0.5541 | 8 / 80 | **0 / 80** | 432.2 ms |
| | 50 | 1.0 | 0.4401 | 0.5264 | 9 / 80 | **0 / 80** | 387.8 ms |
| | 50 | "auto" | 0.4401 | 0.5264 | 9 / 80 | **0 / 80** | 387.5 ms |
| | 100 | 0.5 | 0.4584 | 0.5425 | 10 / 80 | **0 / 80** | 760.9 ms |
| | 100 | 1.0 | 0.4243 | 0.5191 | 12 / 80 | **0 / 80** | 722.3 ms |
| | **100 (Control)** | **"auto"** | **0.4243** | **0.5191** | **12 / 80** | **0 / 80** | **644.1 ms** |
| | 200 | 0.5 | 0.4578 | 0.5413 | 9 / 80 | **0 / 80** | 1248.2 ms |
| | 200 | 1.0 | 0.4256 | 0.5166 | 12 / 80 | **0 / 80** | 1268.5 ms |
| | 200 | "auto" | 0.4256 | 0.5166 | 12 / 80 | **0 / 80** | 1263.5 ms |
| | | | | | | | |
| **28-Feature Reference** | 50 | 0.5 | 0.4655 | 0.5581 | 9 / 80 | **0 / 80** | 335.8 ms |
| | 50 | 1.0 | 0.4387 | 0.5315 | 8 / 80 | **0 / 80** | 320.6 ms |
| | 50 | "auto" | 0.4387 | 0.5315 | 8 / 80 | **0 / 80** | 332.8 ms |
| | 100 | 0.5 | 0.4630 | 0.5539 | 8 / 80 | **0 / 80** | 656.0 ms |
| | 100 | 1.0 | 0.4155 | 0.5287 | 9 / 80 | **0 / 80** | 618.9 ms |
| | **100 (Control)** | **"auto"** | **0.4155** | **0.5287** | **9 / 80** | **0 / 80** | **662.6 ms** |
| | 200 | 0.5 | 0.4636 | 0.5473 | 9 / 80 | **0 / 80** | 1285.1 ms |
| | 200 | 1.0 | 0.4187 | 0.5232 | 9 / 80 | **0 / 80** | 1251.4 ms |
| | 200 | "auto" | 0.4187 | 0.5232 | 9 / 80 | **0 / 80** | 1296.5 ms |

#### 5. Decision Rule Application & Conclusion
1. **Primary Metric Comparison ($N_{\text{FA, sustained\_snapshots\_diag}}$, $k=3$):** All 9 grid configurations achieved **0 sustained false alarms** on snapshots `180..199` ($0 / 80 = 0.0\%$).
2. **Secondary Metric & Stability Analysis:** Smaller subsampling (`max_samples=0.5`) slightly reduces raw snapshot exceedances but increases estimator variance across random seeds due to reduced sample size ($N=80$ sub-sampled from $160$).
3. **Decision Outcome:** Because all grid points maintain 0 sustained false alarms ($k=3$), **the baseline control configuration (`n_estimators=100`, `max_samples="auto"`) is retained** for stability, ensemble variance reduction, and standard scikit-learn defaults.

#### 6. Verification & Audit Sign-Off
- **Reproducibility:** Confirmed reproducibility under `random_state=42`. All 36 unit tests passed cleanly.
- **Data Integrity:** 0 NaNs, 0 Infs detected.
- **Data Partitions:** Verified strict partition bounds (Fit: 0–159, Cal: 160–179, Diag: 180–199, Eval: 200–983).

---

### EXP-08: Threshold Derivation Strategy Comparison

- **Date:** September 30, 2026
- **Status:** **Completed & Audited**
- **Single Factor Under Test:** Anomaly Threshold Strategy (Fit $\mu + 3\sigma$, Fit $\mu + 4\sigma$, Calibration $P_{99}$, Calibration $P_{99.5}$, Calibration $\mu + 3\sigma$).

#### 1. Hypothesis
Evaluating candidate threshold derivation methods tests whether calibrating cutoffs on validation snapshots `160..179` ($P_{99}$, $P_{99.5}$) vs fit snapshots `0..159` ($\mu + 3\sigma$, $\mu + 4\sigma$) balances false-alarm suppression and anomaly sensitivity. Fit-derived thresholds are explicitly designated as exploratory references because they are derived on training data.

#### 2. Experimental Configurations
- **Baseline Control Threshold:** Calibration $P_{99}$ derived on snapshots `160..179` ($N=20$).
- **Candidate Threshold Strategies:**
  - Fit $\mu + 3\sigma$ (Exploratory reference derived on snapshots `0..159`)
  - Fit $\mu + 4\sigma$ (Exploratory reference derived on snapshots `0..159`)
  - Calibration $P_{99.5}$ (Candidate derived on snapshots `160..179`)
  - Calibration $\mu + 3\sigma$ (Candidate derived on snapshots `160..179`)
- **Fixed Parameters Across All Runs:**
  - Estimator: `IsolationForest(n_estimators=100, max_samples="auto", random_state=42)` per channel.
  - Feature Sets: Primary 36-feature set (and reference 28-feature time-domain set).
  - Preprocessing: `RobustScaler()` fit strictly on `0..159`.
  - Persistence Filter: $k=3$ consecutive exceedances.
  - Partitions: Fit `0..159` ($N=160$), Calibration `160..179` ($N=20$), Diagnostic Comparison `180..199` ($N=20$).

#### 3. Metric Definitions
- **Primary Decision Metric:** Sustained diagnostic false-alarm snapshot count ($N_{\text{FA, sustained\_snapshots\_diag}}$) under $k=3$ on snapshots `180..199` ($N=20$ per channel, total $N=80$ evaluations across 4 channels).
- **Secondary Decision Metric:** Raw diagnostic false-alarm snapshot count ($N_{\text{FA, raw\_snapshots\_diag}}$) under $k=1$ on snapshots `180..199` ($N=20$ per channel, total $N=80$).
- **Threshold Value Metric:** Exact scalar cutoff values derived per channel.

#### 4. Detailed Empirical Results

| Feature Set / Threshold Strategy | `ch1` Threshold | `ch2` Threshold | `ch3` Threshold | `ch4` Threshold | Diagnostic Raw FA ($N_{\text{FA, raw\_diag}}$) | Diagnostic Sustained FA ($N_{\text{FA, sust\_diag}}$, $k=3$) |
|---|---|---|---|---|---|---|
| **36-Feature Baseline** | | | | | | |
| **Fit $\mu + 3\sigma$ (Exploratory)** | 0.605243 | 0.612682 | 0.604011 | 0.594609 | 1 / 80 (1.25%) | **0 / 80 (0.0%)** |
| **Fit $\mu + 4\sigma$ (Exploratory)** | 0.663013 | 0.675343 | 0.659840 | 0.653209 | 0 / 80 (0.0%) | **0 / 80 (0.0%)** |
| **Calib $P_{99}$ (Control)** | 0.492242 | 0.451808 | 0.603034 | 0.529307 | **12 / 80 (15.0%)** | **0 / 80 (0.0%)** |
| **Calib $P_{99.5}$ (Candidate)** | 0.492536 | 0.451972 | 0.604952 | 0.532066 | **12 / 80 (15.0%)** | **0 / 80 (0.0%)** |
| **Calib $\mu + 3\sigma$** | 0.549643 | 0.488083 | 0.667665 | 0.585497 | 2 / 80 (2.5%) | **0 / 80 (0.0%)** |
| | | | | | | |
| **28-Feature Reference** | | | | | | |
| **Fit $\mu + 3\sigma$ (Exploratory)** | 0.617437 | 0.619790 | 0.616402 | 0.607225 | 1 / 80 (1.25%) | **0 / 80 (0.0%)** |
| **Fit $\mu + 4\sigma$ (Exploratory)** | 0.677123 | 0.684226 | 0.673727 | 0.668912 | 0 / 80 (0.0%) | **0 / 80 (0.0%)** |
| **Calib $P_{99}$ (Control)** | 0.513168 | 0.472510 | 0.622558 | 0.506628 | **9 / 80 (11.25%)** | **0 / 80 (0.0%)** |
| **Calib $P_{99.5}$ (Candidate)** | 0.514622 | 0.472582 | 0.627375 | 0.506806 | **9 / 80 (11.25%)** | **0 / 80 (0.0%)** |
| **Calib $\mu + 3\sigma$** | 0.569730 | 0.508265 | 0.684026 | 0.582713 | 2 / 80 (2.5%) | **0 / 80 (0.0%)** |

#### 5. Decision Rule Application & Conclusion
1. **Primary Metric Comparison ($N_{\text{FA, sustained\_snapshots\_diag}}$, $k=3$):** All 5 threshold strategies achieved **0 sustained false alarms** on snapshots `180..199` ($0 / 80 = 0.0\%$).
2. **Methodological Distinction & Decision:**
   - Fit-derived thresholds ($\mu + 3\sigma, \mu + 4\sigma$) set cutoffs high ($\sim 0.60 - 0.68$) and remain exploratory references because fitting data is used for threshold calculation.
   - Calibration $P_{99.5}$ produces cutoffs nearly identical to Calibration $P_{99}$ due to calibration window sample size ($N=20$), yielding identical diagnostic false-alarm counts (12 raw, 0 sustained on 36 features; 9 raw, 0 sustained on 28 features).
3. **Decision Outcome:** Because Calibration $P_{99.5}$ provides no reduction in diagnostic false alarms compared to $P_{99}$, **Calibration $P_{99}$ is retained as the baseline control threshold**.

#### 6. Verification & Audit Sign-Off
- **Reproducibility:** Confirmed reproducibility under `random_state=42`. All 36 unit tests passed cleanly.
- **Data Integrity:** 0 NaNs, 0 Infs detected.
- **Data Partitions:** Verified strict partition bounds (Fit: 0–159, Cal: 160–179, Diag: 180–199, Eval: 200–983).

---

### EXP-09: Persistence Filter Comparison

- **Date:** September 30, 2026
- **Status:** **Completed & Audited**
- **Single Factor Under Test:** Persistence Filter Requirement ($k=1$, $k=3$, $k=5$ consecutive exceedances).

#### 1. Hypothesis
Evaluating persistence filtering requirements ($k \in \{1, 3, 5\}$) quantifies the operational trade-off between instant alert reactivity ($k=1$), noise-spike suppression ($k=3$), and delayed online alert confirmation ($k=5$). Increasing $k$ suppresses isolated transient spikes at the cost of $10 \times (k-1)$ minutes of online confirmation latency.

#### 2. Experimental Configurations
- **Baseline Control Persistence Filter ($k=3$):** 3 consecutive exceedances required (20-minute confirmation delay).
- **Candidate Persistence Filters:**
  - $k=1$: Instantaneous exceedance (0-minute delay; no persistence filter).
  - $k=5$: Robust persistence filter (40-minute confirmation delay).
- **Fixed Parameters Across All Runs:**
  - Model: `IsolationForest(n_estimators=100, max_samples="auto", random_state=42)` per channel.
  - Preprocessing: `RobustScaler()` fit strictly on `0..159`.
  - Feature Sets: Primary 36-feature set (and reference 28-feature time-domain set).
  - Threshold Calibration Rule: Validation $P_{99}$ derived strictly on calibration snapshots `160..179` ($N=20$).
  - Partitions: Fit `0..159` ($N=160$), Calibration `160..179` ($N=20$), Diagnostic Comparison `180..199` ($N=20$).

#### 3. Metric Definitions
- **Raw Threshold Exceedances:** Snapshot count in `180..199` with score $S_t > T_{\text{cal}}$ ($k=1$).
- **Sustained Alert Events ($N_{\text{events}}$):** Count of contiguous alert runs of length $L \ge k$ starting in `180..199`.
- **Sustained Anomalous Snapshots ($N_{\text{sust\_snapshots}}$):** Individual snapshots in `180..199` belonging to sustained alert runs.
- **Online Confirmation Delay:** Time elapsed before a causal online monitor confirms $k$ consecutive exceedances: $\Delta t = (k - 1) \times 10 \text{ mins}$.

#### 4. Detailed Empirical Results

| Feature Set / Persistence Filter ($k$) | Confirmation Delay | Diag Raw Exceedances ($k=1$) | Diag Sustained Events | Diag Sustained Snapshots | Diagnostic FA Rate ($R_{\text{FA, sust\_diag}}$) |
|---|---|---|---|---|---|
| **36-Feature Baseline** | | | | | |
| **$k=1$ (Instantaneous)** | 0 mins (0 snaps) | 12 / 80 | 9 events | 12 / 80 | 15.0% |
| **$k=3$ (Baseline Control)** | **20 mins (2 snaps)** | **12 / 80** | **0 events** | **0 / 80** | **0.0%** |
| **$k=5$ (Robust)** | 40 mins (4 snaps) | 12 / 80 | 0 events | 0 / 80 | 0.0% |
| | | | | | |
| **28-Feature Reference** | | | | | |
| **$k=1$ (Instantaneous)** | 0 mins (0 snaps) | 9 / 80 | 8 events | 9 / 80 | 11.25% |
| **$k=3$ (Baseline Control)** | **20 mins (2 snaps)** | **9 / 80** | **0 events** | **0 / 80** | **0.0%** |
| **$k=5$ (Robust)** | 40 mins (4 snaps) | 9 / 80 | 0 events | 0 / 80 | 0.0% |

#### 5. Decision Rule Application & Conclusion
1. **False-Alarm Suppression:** Unfiltered $k=1$ produces 9 spurious alert events (12 flagged snapshots) on snapshots `180..199`. Both $k=3$ and $k=5$ eliminate **100% of diagnostic false alarms** ($R_{\text{FA, sustained}} = 0.0\%$).
2. **Confirmation Latency Trade-off:** $k=3$ achieves complete false-alarm suppression with only a 20-minute online confirmation delay, whereas $k=5$ introduces double the confirmation latency (40 minutes) without any incremental reduction in diagnostic false alarms.
3. **Decision Outcome:** **$k=3$ is retained as the baseline control persistence filter**, offering optimal false-alarm suppression while minimizing online alert confirmation latency.

#### 6. Verification & Audit Sign-Off
- **Reproducibility:** Confirmed reproducibility under `random_state=42`. All 36 unit tests passed cleanly.
- **Data Integrity:** 0 NaNs, 0 Infs detected.
- **Data Partitions:** Verified strict partition bounds (Fit: 0–159, Cal: 160–179, Diag: 180–199, Eval: 200–983).

---

### EXP-10: Training Window Sensitivity (Exploratory)

- **Date:** September 30, 2026
- **Status:** **Completed & Audited**
- **Single Factor Under Test:** Historical Fit Window Length ($N_{\text{fit}}=100$ vs. $N_{\text{fit}}=160$).

#### 1. Hypothesis
Evaluating model sensitivity to training window size ($N_{\text{fit}}=100$ vs. $N_{\text{fit}}=160$) tests whether a shorter baseline fitting period (~16.7 hours) provides sufficient variance representation to maintain false-alarm suppression on the common diagnostic comparison window (`180..199`) compared to the standard 160-snapshot window (~26.7 hours).

#### 2. Experimental Configurations
- **Control Configuration B ($N_{\text{fit}}=160$):**  
  Fit partition: snapshots `0..159` ($N=160$). Calibration partition: snapshots `160..179` ($N=20$).
- **Candidate Configuration A ($N_{\text{fit}}=100$):**  
  Fit partition: snapshots `0..99` ($N=100$). Calibration partition: snapshots `100..119` ($N=20$).
- **Fixed Parameters Across Both Runs:**
  - Common Assessment Window: Diagnostic snapshots `180..199` ($N=20$).
  - Model: `IsolationForest(n_estimators=100, max_samples="auto", random_state=42)` per channel.
  - Preprocessing: `RobustScaler()` fit strictly on the designated fit partition.
  - Threshold Calibration Rule: Validation $P_{99}$ derived strictly on each configuration's calibration window.
  - Persistence Filter: $k=3$ consecutive exceedances.

#### 3. Metric Definitions
- **Primary Decision Metric:** Sustained diagnostic false-alarm snapshot count ($N_{\text{FA, sustained\_snapshots\_diag}}$) under $k=3$ on common snapshots `180..199` ($N=20$ per channel, total $N=80$ evaluations across 4 channels).
- **Secondary Decision Metric:** Raw diagnostic false-alarm snapshot count ($N_{\text{FA, raw\_snapshots\_diag}}$) under $k=1$ on common snapshots `180..199` ($N=20$ per channel, total $N=80$).
- **Calibration Threshold Metrics:** Exact scalar $P_{99}$ cutoffs derived per channel.

#### 4. Detailed Empirical Results

| Feature Set / Fit Window Config | Fit Window | Calib Window | Diagnostic Raw FA ($N_{\text{FA, raw\_diag}}$) | Diagnostic Sustained FA ($N_{\text{FA, sust\_diag}}$, $k=3$) | Diagnostic FA Rate ($R_{\text{FA, sust\_diag}}$) |
|---|---|---|---|---|---|
| **36-Feature Baseline** | | | | | |
| **Config A ($N_{\text{fit}}=100$)** | `0..99` ($N=100$) | `100..119` ($N=20$) | 21 / 80 (26.25%) | **0 / 80** | **0.0%** |
| **Config B ($N_{\text{fit}}=160$, Control)** | `0..159` ($N=160$) | `160..179` ($N=20$) | **12 / 80 (15.0%)** | **0 / 80** | **0.0%** |
| | | | | | |
| **28-Feature Reference** | | | | | |
| **Config A ($N_{\text{fit}}=100$)** | `0..99` ($N=100$) | `100..119` ($N=20$) | 12 / 80 (15.0%) | **0 / 80** | **0.0%** |
| **Config B ($N_{\text{fit}}=160$, Control)** | `0..159` ($N=160$) | `160..179` ($N=20$) | **9 / 80 (11.25%)** | **0 / 80** | **0.0%** |

#### 5. Decision Rule Application & Conclusion
1. **Primary Metric Comparison ($N_{\text{FA, sustained\_snapshots\_diag}}$, $k=3$):** Both configurations ($N_{\text{fit}}=100$ and $N_{\text{fit}}=160$) achieved **0 sustained false alarms** on the common diagnostic window `180..199` ($0 / 80 = 0.0\%$).
2. **Secondary Metric Comparison ($N_{\text{FA, raw\_snapshots\_diag}}$, $k=1$):** Config B ($N_{\text{fit}}=160$) achieved significantly lower raw snapshot exceedances than Config A ($N_{\text{fit}}=100$) (12 vs 21 on 36 features; 9 vs 12 on 28 features). Fitting on only 100 snapshots provides a narrower representation of baseline variance, leading to lower calibration thresholds on `ch3` ($0.5168$ vs $0.6030$) and higher raw noise sensitivity.
3. **Methodological Limitation Disclosure:** Config A and Config B use different calibration sub-periods (`100..119` vs `160..179`). Evaluating both on the common window `180..199` enables comparative assessment, but does not make the experiment fully controlled due to the confounding calibration shift.
4. **Decision Outcome:** **Config B ($N_{\text{fit}}=160$) is retained as the baseline control fit window size**, offering superior noise suppression and more robust baseline estimation.

#### 6. Verification & Audit Sign-Off
- **Reproducibility:** Confirmed reproducibility under `random_state=42`. All 36 unit tests passed cleanly.
- **Data Integrity:** 0 NaNs, 0 Infs detected.
- **Data Partitions:** Verified strict partition bounds (Fit A: 0–99, Cal A: 100–119; Fit B: 0–159, Cal B: 160–179; Diag: 180–199, Eval: 200–983).

---

### EXP-11: Per-Channel vs. Pooled Modeling Architecture (Exploratory)

- **Date:** September 30, 2026
- **Status:** **Completed & Audited**
- **Single Factor Under Test:** System Architecture (Per-Channel 4-Model Pipeline vs. Pooled Joint Model).

#### 1. Hypothesis
Comparing Per-Channel modeling (4 separate estimators trained on single-channel features) with Pooled modeling (1 joint estimator trained on 36 concatenated multi-channel features) tests whether cross-channel feature pooling improves overall false-alarm suppression or dilutes spatial anomaly resolution.

#### 2. Experimental Configurations
- **Control Configuration A (Per-Channel Modeling):**  
  4 separate `IsolationForest` estimators (1 per channel). Each estimator fits strictly on its channel's features (`0..159`). Thresholds ($P_{99}$) derived per channel (`160..179`). Evaluated per channel on `180..199` ($N=80$ total channel-snapshot evaluations).
- **Candidate Configuration B (Pooled Joint Modeling):**  
  1 joint `IsolationForest` estimator fit on 36 features concatenated across all 4 channels (`0..159`). Single system threshold ($P_{99}$) derived on pooled scores (`160..179`). Evaluated on pooled scores on `180..199` ($N=20$ snapshot evaluations).
- **Fixed Parameters Across Both Runs:**
  - Estimator: `IsolationForest(n_estimators=100, max_samples="auto", random_state=42)`.
  - Feature Scaler: `RobustScaler()` fit strictly on `0..159`.
  - Feature Sets: Primary 36-feature set (and reference 28-feature time-domain set).
  - Persistence Filter: $k=3$ consecutive exceedances.
  - Partitions: Fit `0..159` ($N=160$), Calibration `160..179` ($N=20$), Diagnostic Comparison `180..199` ($N=20$).

#### 3. Metric Definitions
- **Primary Decision Metric:** Sustained diagnostic false-alarm snapshot count ($N_{\text{FA, sustained\_snapshots\_diag}}$) under $k=3$ on snapshots `180..199`.
- **Secondary Decision Metric:** Raw diagnostic false-alarm snapshot count ($N_{\text{FA, raw\_snapshots\_diag}}$) under $k=1$ on snapshots `180..199`.
- **Spatial Resolution Metric:** Ability to localize anomalies to specific bearing channels (`ch1`..`ch4`).

#### 4. Detailed Empirical Results

| Feature Set / Architecture Config | Model Count | Calib $P_{99}$ Threshold ($T_{\text{cal}}$) | Diagnostic Raw FA ($N_{\text{FA, raw\_diag}}$) | Diagnostic Sustained FA ($N_{\text{FA, sust\_diag}}$, $k=3$) | Diagnostic FA Rate ($R_{\text{FA, sust\_diag}}$) |
|---|---|---|---|---|---|
| **36-Feature Baseline** | | | | | |
| **Config A (Per-Channel Control)** | 4 models | `ch1`: 0.4922, `ch2`: 0.4518<br>`ch3`: 0.6030, `ch4`: 0.5293 | **12 / 80 (15.0%)** | **0 / 80** | **0.0%** |
| **Config B (Pooled Candidate)** | 1 model | Pooled System: 0.4881 | **5 / 20 (25.0%)** | **0 / 20** | **0.0%** |
| | | | | | |
| **28-Feature Reference** | | | | | |
| **Config A (Per-Channel Control)** | 4 models | `ch1`: 0.5132, `ch2`: 0.4725<br>`ch3`: 0.6226, `ch4`: 0.5066 | **9 / 80 (11.25%)** | **0 / 80** | **0.0%** |
| **Config B (Pooled Candidate)** | 1 model | Pooled System: 0.4643 | **3 / 20 (15.0%)** | **0 / 20** | **0.0%** |

#### 5. Decision Rule Application & Conclusion
1. **Primary Metric Comparison ($N_{\text{FA, sustained\_snapshots\_diag}}$, $k=3$):** Both architectures achieved **0 sustained false alarms** on diagnostic snapshots `180..199` ($0.0\%$).
2. **Secondary Metric & Spatial Localization Analysis:**
   - Raw single-snapshot false alarm rate: Pooled modeling registered 5 flagged snapshots out of 20 (25.0%), whereas per-channel modeling registered 12 flagged channel-snapshots out of 80 (15.0%).
   - Spatial localization: Per-channel modeling preserves explicit spatial location resolution (identifying specifically which bearing channel `ch1`..`ch4` deviates), whereas pooled joint modeling concatenates features into a single global score, obscuring localized bearing defect origins.
3. **Decision Outcome:** **Per-Channel Modeling is retained as the primary baseline architecture**, maintaining channel-specific spatial diagnostic resolution while achieving zero sustained false alarms.

#### 6. Verification & Audit Sign-Off
- **Reproducibility:** Confirmed reproducibility under `random_state=42`. All 36 unit tests passed cleanly.
- **Data Integrity:** 0 NaNs, 0 Infs detected.
- **Data Partitions:** Verified strict partition bounds (Fit: 0–159, Cal: 160–179, Diag: 180–199, Eval: 200–983).

---

### EXP-12: Final Model Improvement Synthesis & Architectural Audit

- **Date:** September 30, 2026
- **Status:** **Completed & Reconciled Audit (Finalized)**
- **Single Factor Under Test:** Full Multi-Factor Synthesis & Pipeline Selection (`EXP-04` through `EXP-11`).

#### 1. Hypothesis
Synthesizing empirical findings across feature selection (`EXP-04`), feature scaling (`EXP-05`), PCA component count (`EXP-06`), Isolation Forest hyperparameters (`EXP-07`), threshold derivation (`EXP-08`), persistence filtering (`EXP-09`), fit window size (`EXP-10`), and modeling architecture (`EXP-11`) establishes a defensible, leakage-free anomaly detection pipeline optimized for false-alarm suppression and spatial defect localization.

#### 2. Master Synthesis of Completed Experiments (Reconciled & Approved)

| Experiment ID | Tested Factor | Control vs. Candidates | Selected Final Setting | Selection Rationale & Decision Rule Outcome |
|---|---|---|---|---|
| **EXP-04** | Feature Set | Full 36 vs. Time-Domain 28 | **28-Feature Time-Domain Set** | 28-feature set reduced raw false alarms (9 vs 12 snapshots on 180..199); primary metric $N_{\text{FA, sust}}$ tied at 0. Selected per EXP-04 decision rule and user approval. |
| **EXP-05** | Feature Scaler | RobustScaler vs. StandardScaler | **RobustScaler (Baseline Control)** | Identical anomaly scores due to rank-order invariance of Isolation Forest split decisions. RobustScaler retained for consistency. |
| **EXP-06** | PCA Rank | $k=1$ vs. $k=2$ vs. $k=3$ components | **$k=3$ Components (Accepted Improvement)** | $k=3$ reduced raw false alarms to 3/4 snapshots and achieved lowest calibration MSE ($\text{MSE}_{\text{cal}} = 0.1525$ vs $0.2581$). Fully traceable and accepted. |
| **EXP-07** | iForest Hyperparams | Grid $n_{\text{est}} \in \{50, 100, 200\} \times max\_samp \in \{0.5, 1.0, \text{"auto"}\}$ | **$n_{\text{est}}=100$, $max\_samp=\text{"auto"}$ (Control)** | All 9 grid configurations achieved 0 sustained false alarms ($k=3$). Control retained per decision rule to avoid parameter churn. Traceable. |
| **EXP-08** | Threshold Strategy | Fit $\mu+3\sigma, \mu+4\sigma$ vs. Cal $P_{99}, P_{99.5}, \mu+3\sigma$ | **Cal $P_{99}$ Threshold (Baseline Control)** | Cal $P_{99.5}$ yielded cutoffs identical to $P_{99}$ due to calibration sample size ($N=20$). Cal $P_{99}$ retained as baseline control. Fit cutoffs kept as exploratory. Traceable. |
| **EXP-09** | Persistence Filter | $k=1$ vs. $k=3$ vs. $k=5$ consecutive exceedances | **$k=3$ Filter (Baseline Control)** | $k=3$ eliminated 100% of diagnostic false alarms ($R_{\text{FA, sust}} = 0.0\%$) with a modest 20-min delay (vs 40-min delay for $k=5$). Retained. |
| **EXP-10** | Fit Window Length | $N_{\text{fit}}=100$ vs. $N_{\text{fit}}=160$ | **$N_{\text{fit}}=160$ Window (Baseline Control)** | $N_{\text{fit}}=160$ provided lower raw noise sensitivity (12 vs 21 on 36-feat; 9 vs 12 on 28-feat). Note: Calibration windows differed (`100..119` vs `160..179`), introducing a calibration shift. |
| **EXP-11** | Modeling Architecture | Per-Channel (4 models) vs. Pooled (1 joint model) | **Per-Channel + Logical OR Aggregation** | Both achieved 0 sustained false alarms. Per-channel architecture retained to preserve spatial defect localization (`ch1`..`ch4`). System alert uses Logical OR aggregation across channels ($k=3$). |

#### 3. Approved Pipeline Decisions & Reconciliation Findings
1. **EXP-04 Feature Set Selection:** The 28-feature time-domain set achieved lower raw false alarms (9 vs 12) while tying at 0 sustained false alarms ($k=3$). Evaluated under the EXP-04 decision rule and approved by the user for final Phase 9 pipeline packaging.
2. **EXP-10 Calibration Shift Confound:** Raw false alarms on 36 features were **21** for $N_{\text{fit}}=100$ (Cal `100..119`) and **12** for $N_{\text{fit}}=160$ (Cal `160..179`). On 28 features, raw false alarms were **12** for $N_{\text{fit}}=100$ and **9** for $N_{\text{fit}}=160$. Narrative claims that calibration was held constant are unverified and rejected due to differing calibration windows.
3. **EXP-11 System Alert Aggregation:** Per-channel modeling is combined with Logical OR system aggregation: a global system alert is active if ANY channel meets the $k=3$ sustained threshold condition. Aligns with Phase 8 evaluation protocol Section J.
4. **EXP-06, 07, 08 Traceability:** EXP-06 ($k=3$ selection), EXP-07 (control retention), and EXP-08 ($P_{99}$ control retention) are fully verified and traceable to empirical logs and approved decision rules.

#### 4. Methodological Limitations & Disclosures
1. **Small Diagnostic Sample Size ($N=20$):** The diagnostic window contains only 20 snapshots per channel. A single snapshot change alters the raw false alarm rate by 5.0%.
2. **Selection Bias Risk:** Repeated evaluation of candidate feature subsets on snapshots `180..199` introduces potential selection bias.
3. **Assumed Healthy Baseline:** Snapshots `0..199` are assumed healthy based on statistical signal stability. False alarms are defined strictly relative to this operational assumption.
4. **Single-Case Study ($n=1$):** Results reflect a single run-to-failure trajectory under laboratory conditions (~2000 RPM, 6000 lb load) and cannot prove general industrial reliability.
5. **No Ground-Truth Health Annotations:** Anomaly alerts measure statistical novelty relative to baseline operation, **not** verified physical defect onset, exact failure time, or remaining useful life (RUL).
