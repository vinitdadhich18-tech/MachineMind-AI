# Statistical Baseline Report — NASA IMS Bearing Dataset (Set 2)

**Project:** MachineMind AI  
**Component:** `ml-service`  
**Phase:** Phase 6 — Statistical Baseline  
**Date:** 2026-09-28  
**Document Status:** Complete & Verified Baseline Execution Report  

---

## 1. Executive Summary & Objective

This report documents the statistical baseline evaluation for **Set 2** of the **NASA IMS Bearing Dataset**. 

The objective of Phase 6 is to implement a transparent, non-machine-learning statistical reference model to quantify vibration signal deviations across 984 sequential 1-second snapshots. The baseline parameters are fitted strictly on an initial chronological period and kept frozen while evaluating all subsequent snapshots.

> **Crucial Methodological Caveats & Scope Boundaries:**
> 1. **Provisional Reference Baseline:** The first 160 snapshots form a provisional fitting baseline. *The health status of these files is an unverified assumption and is not independently confirmed by ground-truth labels.*
> 2. **Provisional Reference Evaluation Period:** The 40 snapshots from index 160 to 199 form a reference evaluation period. *This period is also not independently confirmed healthy.*
> 3. **Provisional Reference Boundaries:** Threshold lines drawn at **3.0** and **3.5** are uncalibrated reference indicators (e.g., standard deviation / MAD multiples), *not validated failure thresholds*.
> 4. **Single-Case Study:** IMS Set 2 contains a single run-to-failure trajectory (bearing 1 outer-race defect). Results *do not constitute failure prediction or Remaining Useful Life (RUL) estimation*.

---

## 2. Chronological Dataset Partitions

To prevent temporal data leakage, the 984 sequential snapshots (recorded at 10-minute intervals over ~7 days) are chronologically partitioned as follows:

| Partition Period | Snapshot Index Range | Snapshot Count ($N$) | Duration (approx.) | Purpose & Methodological Status |
|---|---|---|---|---|
| **Baseline Fitting Period** | `0` to `159` | 160 | ~26.7 hours | Used strictly to compute baseline mean, std, median, MAD, IQR, and P2P. *Provisional reference baseline, unverified health status.* |
| **Reference Evaluation Period** | `160` to `199` | 40 | ~6.7 hours | Evaluated using frozen baseline parameters to check stability on unseen reference data. *Unverified health status.* |
| **Sequential Evaluation Period** | `200` to `983` | 784 | ~130.7 hours | Evaluated sequentially using frozen baseline parameters to track degradation trajectory to test completion. |

---

## 3. Statistical Baseline Methodology & Scoring Definitions

### 3.1 Parameter Fitting & Scale-Aware Guarding

Using the first 160 rows of the feature matrix ($984 \text{ rows} \times 36 \text{ features}$ from [`data/processed/features_set2.csv`](file:///d:/Projects/MachineMind%20AI/ml-service/data/processed/features_set2.csv)), baseline statistics are computed per feature column:

- **Mean ($\mu$) & Sample Standard Deviation ($\sigma$):** Standard parametric measures of location and dispersion ($ddof=1$).
- **Median ($\text{Med}$) & Median Absolute Deviation ($\text{MAD}$):** Robust non-parametric measures of location and dispersion ($\text{MAD} = \text{median}(|x - \text{Med}|)$).
- **Interquartile Range ($\text{IQR}$) & Peak-to-Peak Range ($\text{P2P}$):** $\text{IQR} = Q_{75} - Q_{25}$ and $\text{P2P} = \max(x) - \min(x)$.

#### Scale-Aware Guarding & Feature Eligibility Policy:
To prevent zero-division or numerical explosion when computing Z-scores on near-constant signals, baseline parameters enforce scale-aware lower cutoffs:
$$\text{Cutoff}_{\text{std}} = \max(10^{-6}, 10^{-4} \cdot \text{P2P}_{\text{fit}}), \quad \sigma_{\text{guarded}} = \max(\sigma_{\text{fit}}, \text{Cutoff}_{\text{std}})$$
$$\text{Cutoff}_{\text{mad}} = \max(10^{-6}, 10^{-4} \cdot \text{IQR}_{\text{fit}}), \quad \text{MAD}_{\text{guarded}} = \max(\text{MAD}_{\text{fit}}, \text{Cutoff}_{\text{mad}})$$

If a feature exhibits zero variation ($\text{P2P}_{\text{fit}} = 0.0$) across all 160 fitting snapshots, it is marked ineligible and excluded from composite scoring.
- **Eligible Z-score Features:** **36 / 36** (0 excluded)
- **Eligible Modified Z-score Features:** **36 / 36** (0 excluded)

---

### 3.2 Four Composite Anomaly Scores

For each snapshot $t \in [0, 983]$, standard Z-scores ($Z_{i,t} = \frac{x_{i,t} - \mu_i}{\sigma_{\text{guarded}, i}}$) and robust modified Z-scores ($M_{i,t} = 0.6745 \cdot \frac{x_{i,t} - \text{Med}_i}{\text{MAD}_{\text{guarded}, i}}$) are computed using the frozen baseline parameters. These per-feature Z-scores are aggregated into four composite anomaly scores:

1. **Max Absolute Z-Score (`score_max_z`):**
   $$\text{score\_max\_z}_t = \max_{i \in \text{Eligible}} |Z_{i,t}|$$
   *Measures the single worst single-feature deviation from baseline.*
2. **RMS Composite Z-Score (`score_rms_z`):**
   $$\text{score\_rms\_z}_t = \sqrt{\frac{1}{K_{\text{eligible}}} \sum_{i \in \text{Eligible}} Z_{i,t}^2}$$
   *Measures total standardized energy deviation across all features ($K_{\text{eligible}}=36$).*
3. **Max Absolute Modified Z-Score (`score_max_m`):**
   $$\text{score\_max\_m}_t = \max_{i \in \text{Eligible}} |M_{i,t}|$$
   *Measures the maximum feature deviation using robust median/MAD scaling.*
4. **RMS Composite Modified Z-Score (`score_rms_m`):**
   $$\text{score\_rms\_m}_t = \sqrt{\frac{1}{K_{\text{eligible}}} \sum_{i \in \text{Eligible}} M_{i,t}^2}$$
   *Measures robust overall system energy deviation across all features.*

---

## 4. Empirical Observations & Results

The composite scores across the 984 snapshots demonstrate clear degradation trends over time:

### 4.1 Summary Statistics per Partition

| Composite Metric | Baseline Fitting (0-159) Mean [Max] | Reference Eval (160-199) Mean [Max] | Sequential Eval (200-983) Mean [Max] | Full Run Maximum Value |
|---|---|---|---|---|
| `score_max_z` | 2.58 [4.49] | 2.65 [4.14] | 10.97 [89.37] | 89.37 (snapshot 705) |
| `score_rms_z` | 0.99 [1.44] | 1.01 [1.32] | 3.51 [25.10] | 25.10 (snapshot 705) |
| `score_max_m` | 2.62 [4.43] | 2.67 [3.94] | 10.87 [86.72] | 86.72 (snapshot 705) |
| `score_rms_m` | 0.99 [1.44] | 1.01 [1.33] | 3.54 [24.97] | 24.97 (snapshot 705) |

### 4.2 Key Trajectory Observations

1. **Baseline Stability (Snapshots 0 to 199):**
   - During the baseline fitting period (0-159), the RMS composite scores (`score_rms_z` and `score_rms_m`) fluctuate closely around **1.0** (mean $\approx 0.99$, standard deviation $\approx 0.16$), confirming standard normalization behavior.
   - During the unseen reference evaluation period (160-199), the scores remain highly stable (mean $\approx 1.01$), staying well within normal baseline variation.

2. **Early Degradation Emergence (~Snapshot 530 - 700):**
   - Around snapshot **530** (~5,300 minutes into the test), `score_rms_z` and `score_rms_m` exhibit a sustained upward trend, crossing the provisional reference line **3.0** for the first time.
   - A dramatic acceleration peak occurs around snapshot **705** (`score_rms_z` reaches **25.10**, `score_max_z` reaches **89.37**), corresponding to severe physical spalling development reported on bearing 1.

3. **Late-Stage Fluctuation & Test End:**
   - Following snapshot 700, scores exhibit intense high-amplitude oscillations, reflecting complex fault wear, transient debris clearance, and extreme signal non-stationarity prior to test termination at snapshot 983.

---

## 5. Artifact & Deliverable Locations

All generated artifacts have been verified and saved to their exact paths:

- **Notebook:** [`notebooks/04_statistical_baseline.ipynb`](file:///d:/Projects/MachineMind%20AI/ml-service/notebooks/04_statistical_baseline.ipynb) (Executed top-to-bottom without errors)
- **Exported Dataset:** [`data/processed/baseline_scores_set2.csv`](file:///d:/Projects/MachineMind%20AI/ml-service/data/processed/baseline_scores_set2.csv) ($984 \text{ rows} \times 8 \text{ columns}$)
- **Time-Series Figures:**
  1. [`reports/figures/baseline/baseline_score_max_z.png`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/figures/baseline/baseline_score_max_z.png)
  2. [`reports/figures/baseline/baseline_score_rms_z.png`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/figures/baseline/baseline_score_rms_z.png)
  3. [`reports/figures/baseline/baseline_score_max_m.png`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/figures/baseline/baseline_score_max_m.png)
  4. [`reports/figures/baseline/baseline_score_rms_m.png`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/figures/baseline/baseline_score_rms_m.png)

---

## 6. Empirical Validation Results

The baseline dataset and pipeline passed all 7 automated empirical verification checks:

1. **Row Count Alignment:** Re-loaded dataset contains exactly **984 rows** matching the original raw snapshot count.
2. **Column Integrity:** Exported dataset contains exact 8 columns (`file_index`, `filename`, `timestamp`, `is_valid`, `score_max_z`, `score_rms_z`, `score_max_m`, `score_rms_m`).
3. **Chronological Index Order:** `file_index` is strictly contiguous from `0` to `983`.
4. **Metadata Preservation:** Reloaded `filename` and `timestamp` match [`data/processed/features_set2.csv`](file:///d:/Projects/MachineMind%20AI/ml-service/data/processed/features_set2.csv) identically.
5. **No Missing Values:** `0` NaN or Null entries across all rows and columns.
6. **Numerical Finiteness:** 100% of composite score values are finite floating-point numbers.
7. **Frozen Parameters:** Evaluation scores on snapshots 0–159 are 100% deterministic and unaffected by subsequent snapshots.

---

## 7. Explicit Limitations & Unresolved Issues

1. **Unverified Health Assumption:** The baseline fitting period (0-159) is assumed healthy based on early amplitude stability; no physical inspection or ground-truth health labels exist for individual snapshots.
2. **Provisional Reference Boundaries:** The reference lines at 3.0 and 3.5 are mathematical reference indicators, not tuned or calibrated alarm thresholds. False alarm rates and detection precision are uncalibrated.
3. **Single Failure Scenario:** IMS Set 2 documents a single failure run (outer-race defect on bearing 1). Performance on other failure modes (inner race, roller ball) or operating conditions is unverified.
4. **No Failure-Time or RUL Prediction:** The baseline anomaly score measures statistical distance from baseline, not time-to-failure or Remaining Useful Life.
