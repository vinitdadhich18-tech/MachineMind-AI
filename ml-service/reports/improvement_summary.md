# Phase 9 Summary: Model Improvement & Final Architecture Selection

**Project:** MachineMind AI  
**Dataset:** NASA IMS Bearing Dataset (Subset: Set 2, 984 snapshots)  
**Task:** Unsupervised Vibration Anomaly Detection  
**Phase:** Phase 9 of 11 — Model Improvement  
**Document Type:** Final Model Improvement Report & Architectural Synthesis  
**Author:** MachineMind AI Team & Mentor  
**Date:** September 30, 2026  

---

## 1. Executive Summary

Phase 9 conducts controlled, one-factor-at-a-time (OFAT) improvement experiments across eight methodological dimensions (`EXP-04` through `EXP-11`) and synthesizes the findings in `EXP-12` to establish the final, verified unsupervised anomaly detection pipeline for MachineMind AI.

### Approved Final Pipeline & Architectural Decisions
1. **Feature Set Selection (`EXP-04`):** **28-Feature Time-Domain Set** (7 time-domain statistical features per channel: `mean`, `std`, `rms`, `p2p`, `skewness`, `kurtosis`, `crest_factor`).
   - *Rationale:* Evaluated under the approved EXP-04 decision rule. While both feature sets tied on the primary metric ($0$ sustained false alarms under $k=3$), the 28-feature set achieved a 25.0% reduction in raw snapshot noise (9 vs 12 raw false alarms on `180..199`) and improved feature parsimony by excluding high-frequency spectral windowing noise. User approved this selection.
2. **Feature Scaling Method (`EXP-05`):** **`RobustScaler`** (fitted strictly on baseline fit snapshots `0..159`).
   - *Rationale:* Isolation Forest decision tree split logic is invariant to monotonic feature scaling ($\max |\Delta s| = 0.0$). `RobustScaler` is retained as baseline control to protect against baseline outliers.
3. **PCA Component Rank (`EXP-06`):** **$k=3$ Principal Components**.
   - *Rationale:* Increasing PCA rank from $k=2$ to $k=3$ reduced raw diagnostic false alarms from 5 down to 3 snapshots (36 features) / 4 snapshots (28 features), while reducing calibration reconstruction error ($\text{MSE}_{\text{cal}} = 0.1525$ vs $0.2581$).
4. **Isolation Forest Hyperparameters (`EXP-07`):** **`n_estimators=100`, `max_samples="auto"`, `random_state=42`**.
   - *Rationale:* Grid search across 9 parameter combinations yielded identical primary performance ($0$ sustained false alarms for all grid points). Baseline control configuration is retained per decision rule to avoid parameter churn.
5. **Threshold Derivation Strategy (`EXP-08`):** **Calibration $P_{99}$ Percentile Threshold ($T_{\text{cal\_P99}}$)**.
   - *Rationale:* Derived strictly on calibration snapshots `160..179` ($N=20$). Fit-derived thresholds ($\mu+3\sigma, \mu+4\sigma$) are kept as exploratory reference cutoffs.
6. **Persistence Filter (`EXP-09`):** **$k=3$ Consecutive Exceedances** (20-minute online confirmation delay).
   - *Rationale:* Eliminates 100% of diagnostic false alarms ($R_{\text{FA, sustained}} = 0.0\%$) with a 20-minute confirmation delay (compared to 40 minutes for $k=5$).
7. **Fit Window Sensitivity & Confound Disclosure (`EXP-10`):** **Standard Fit Window $N_{\text{fit}}=160$ (~26.7 hours)**.
   - *Confound Explanation:* $N_{\text{fit}}=100$ used calibration window `100..119`, while $N_{\text{fit}}=160$ used `160..179`. Because calibration windows differed, the raw false alarm difference (21 vs 12 on 36-feat; 12 vs 9 on 28-feat) includes a calibration shift confound. $N_{\text{fit}}=160$ is retained as baseline control.
8. **Modeling Architecture & System Alert Aggregation (`EXP-11` & `EXP-12`):** **Per-Channel Modeling with Logical OR System Aggregation**.
   - *Rationale:* 4 separate per-channel models (`ch1`..`ch4`) preserve spatial bearing defect localization. System-level alerts use **Logical OR Aggregation**: a system alert is triggered if **ANY channel** triggers a sustained alert ($k=3$), aligning with the frozen Phase 8 protocol and maximizing single-bearing defect sensitivity. User approved this policy.

---

## 2. Reconciled Synthesis Table (EXP-04 to EXP-11)

| Experiment ID | Tested Dimension / Factor | Compared Configurations | Partitions (Fit / Cal / Diag) | Primary Metric ($N_{\text{FA, sust\_diag}}$, $k=3$) | Secondary Metric ($N_{\text{FA, raw\_diag}}$) | Observed Outcome | Decision & Selection Rule | Verification Status |
|---|---|---|---|---|---|---|---|---|
| **EXP-04** | Feature Set Ablation | Full 36 Features vs. Time-Domain 28 Features | Fit: `0..159`<br>Cal: `160..179`<br>Diag: `180..199` | **0 vs. 0** ($0.0\%$) | **12 vs. 9** snapshots | Time-domain set reduced raw false alarms by 25%. Primary metric tied at 0. | **Select 28-Feature Set** per decision rule (lower raw noise, parsimony). | **Verified & User Approved** |
| **EXP-05** | Feature Scaling Method | `RobustScaler` vs. `StandardScaler` | Fit: `0..159`<br>Cal: `160..179`<br>Diag: `180..199` | **0 vs. 0** ($0.0\%$) | **12 vs. 12** (36-feat)<br>**9 vs. 9** (28-feat) | Identical score trajectories due to rank-order invariance of tree splits. | Retain `RobustScaler` as baseline control for consistency. | **Verified** |
| **EXP-06** | PCA Component Count | $k=1$ vs. $k=2$ vs. $k=3$ components | Fit: `0..159`<br>Cal: `160..179`<br>Diag: `180..199` | **0 vs. 0 vs. 0** ($0.0\%$) | **6 vs. 5 vs. 3** (36-feat)<br>**5 vs. 5 vs. 4** (28-feat) | $k=3$ reduced raw false alarms to 3/4 snapshots and lowered calibration MSE to $0.1525$. | **Accept $k=3$** as improved PCA component rank ($k=2$ retained as baseline). | **Verified** |
| **EXP-07** | iForest Hyperparameters | $n_{\text{est}} \in \{50, 100, 200\} \times max\_samp \in \{0.5, 1.0, \text{"auto"}\}$ | Fit: `0..159`<br>Cal: `160..179`<br>Diag: `180..199` | **0 across all 9 grid points** | **8 to 12** (36-feat)<br>**8 to 9** (28-feat) | All grid points achieved 0 sustained false alarms under $k=3$. | Retain `n_estimators=100`, `max_samples="auto"` as baseline control per decision rule. | **Verified** |
| **EXP-08** | Threshold Strategy | Fit $\mu+3\sigma, \mu+4\sigma$ vs. Cal $P_{99}, P_{99.5}, \mu+3\sigma$ | Fit: `0..159`<br>Cal: `160..179`<br>Diag: `180..199` | **0 across all strategies** | **0 to 12** (36-feat)<br>**0 to 9** (28-feat) | Cal $P_{99}$ and $P_{99.5}$ yielded identical cutoffs due to $N=20$ calibration sample size. | Retain Calibration $P_{99}$ as baseline control. Fit cutoffs kept as exploratory. | **Verified** |
| **EXP-09** | Persistence Filter | $k=1$ vs. $k=3$ vs. $k=5$ consecutive exceedances | Fit: `0..159`<br>Cal: `160..179`<br>Diag: `180..199` | **12 vs. 0 vs. 0** (36-feat)<br>**9 vs. 0 vs. 0** (28-feat) | **12 vs. 12 vs. 12** (36-feat)<br>**9 vs. 9 vs. 9** (28-feat) | $k=3$ and $k=5$ eliminated 100% of diagnostic false alarms. $k=3$ requires 20 min delay vs 40 min for $k=5$. | Retain $k=3$ as baseline control filter. | **Verified** |
| **EXP-10** | Fit Window Sensitivity | $N_{\text{fit}}=100$ vs. $N_{\text{fit}}=160$ | Fit A: `0..99`<br>Fit B: `0..159`<br>Diag: `180..199` | **0 vs. 0** ($0.0\%$) | **21 vs. 12** (36-feat)<br>**12 vs. 9** (28-feat) | $N_{\text{fit}}=160$ provided lower raw noise sensitivity. Calibration windows differed (confounding shift). | Retain $N_{\text{fit}}=160$ as baseline control fit window length. | **Verified** (Calibration shift noted) |
| **EXP-11** | System Architecture | Per-Channel (4 models) vs. Pooled (1 joint model) | Fit: `0..159`<br>Cal: `160..179`<br>Diag: `180..199` | **0 / 80 vs. 0 / 20** snapshots | **12 / 80 vs. 5 / 20** (36-feat)<br>**9 / 80 vs. 3 / 20** (28-feat) | Both achieved 0 sustained false alarms. Per-channel preserves spatial bearing localization. | Retain Per-Channel Modeling with Logical OR Aggregation. | **Verified** (Denominator mismatch noted) |

---

## 3. Final Approved Model Pipeline Specifications

The final Phase 9 pipeline specifications for Phase 10 packaging are established:

### A. Isolation Forest Anomaly Detection Pipeline
- **Architecture:** 4 Independent Per-Channel Isolation Forests (`ch1`, `ch2`, `ch3`, `ch4`).
- **Feature Set:** 28-feature time-domain set (7 features/channel: `mean`, `std`, `rms`, `p2p`, `skewness`, `kurtosis`, `crest_factor`).
- **Feature Preprocessing:** `RobustScaler()` fit strictly on baseline fit snapshots `0..159`.
- **Estimator Hyperparameters:** `IsolationForest(n_estimators=100, max_samples="auto", contamination="auto", random_state=42)` per channel.
- **Fit Partition:** Snapshots `0..159` ($N=160$, ~26.7 hours).
- **Calibration Partition:** Snapshots `160..179` ($N=20$, ~3.3 hours).
- **Threshold Rule:** Channel-specific $P_{99}$ percentile line ($T_{c, \text{cal\_P99}}$) derived on `160..179`.
- **Persistence Filter:** $k=3$ consecutive exceedances (20-minute online confirmation delay).
- **Channel Output:** Binary flag $\mathbb{I}(S_{c, t} > T_{c, \text{cal\_P99}})$ + normalized score margin $(S_{c, t} - T_c) / T_c$.
- **System Alert Policy:** **Logical OR Aggregation**:
  $$\text{Alert}_{\text{system}, t} = \bigvee_{c=1}^4 \text{Alert}_{c, t} \quad \text{where } \text{Alert}_{c, t} = \prod_{\tau=t-2}^t \mathbb{I}(S_{c, \tau} > T_{c, \text{cal\_P99}})$$

### B. PCA Reconstruction Error Pipeline
- **Architecture:** 4 Independent Per-Channel PCA Models (`ch1`, `ch2`, `ch3`, `ch4`).
- **Feature Set:** 28-feature time-domain set (and 36-feature set compatibility).
- **Feature Preprocessing:** `RobustScaler()` fit strictly on baseline fit snapshots `0..159`.
- **Estimator Hyperparameters:** `PCA(n_components=3, random_state=42)` per channel.
- **Fit Partition:** Snapshots `0..159` ($N=160$, ~26.7 hours).
- **Calibration Partition:** Snapshots `160..179` ($N=20$, ~3.3 hours).
- **Threshold Rule:** Channel-specific $P_{99}$ percentile line ($T_{c, \text{cal\_P99}}$) derived on `160..179`.
- **Persistence Filter:** $k=3$ consecutive exceedances (20-minute online confirmation delay).
- **System Alert Policy:** Logical OR Aggregation across channels under $k=3$ persistence.

---

## 4. Methodological Limitations & Data Leakage Prevention Disclosures

1. **Prior Inspection of Evaluation Data (`200..983`):** The evaluation partition was scored and inspected during Phase 8. Therefore, snapshots 200–983 cannot be treated as a strict untouched blind test set.
2. **Zero Evaluation Partition Tuning:** No hyperparameters, scalers, features, or thresholds were selected or tuned using snapshots 200–983 during Phase 9.
3. **Small Diagnostic Sample Size ($N=20$):** Diagnostic comparison was conducted strictly on snapshots `180..199` ($N=20$). A single snapshot change alters the raw false alarm rate by 5.0%.
4. **Selection Bias Risk:** Repeatedly testing multiple candidate configurations against the same 20 diagnostic snapshots introduces potential selection bias.
5. **EXP-11 Denominator Mismatch Disclosure:** Per-channel modeling produces 4 score streams ($N=80$ channel-snapshot evaluations), whereas pooled modeling produces 1 score stream ($N=20$ system-snapshot evaluations). Direct raw false alarm percentage comparisons between the two must account for this denominator difference.
6. **Single Run-to-Failure Trajectory ($n=1$):** IMS Set 2 documents a single outer-race defect on Bearing 1 under constant load (~2000 RPM, 6000 lb). Results describe a single-case study and cannot prove general industrial reliability.
7. **No Ground-Truth Health Annotations:** Dataset contains no binary healthy/faulty labels per snapshot. Anomaly alerts measure statistical novelty relative to baseline operation, **not** verified physical defect onset, exact failure time, or remaining useful life (RUL).

---

## 5. Deliverables & Audit Sign-Off

- [`reports/improvement_summary.md`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/improvement_summary.md): Final Phase 9 Summary Report (Updated & Approved).
- [`reports/experiment_log.md`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/experiment_log.md): Master Experiment Log with reconciled audit entries.
- [`notebooks/07_model_improvement.ipynb`](file:///d:/Projects/MachineMind%20AI/ml-service/notebooks/07_model_improvement.ipynb): Model improvement notebook executed clean.
- Unit Test Suite (`src/`): All 36 unit tests passed cleanly (0 failures, 0 errors).
