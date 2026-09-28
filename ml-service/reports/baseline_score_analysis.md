# Anomaly Score Analysis & Threshold Sensitivity Report — NASA IMS Bearing Dataset (Set 2)

**Project:** MachineMind AI  
**Component:** `ml-service`  
**Phase:** Phase 6, Step 4 — Anomaly Score Analysis & Threshold Sensitivity Exploration (Audited)  
**Date:** 2026-09-28  
**Document Status:** Verified & Audited Exploratory Analysis Report  

---

## 1. Executive Summary & Objective

This report documents the audited exploratory analysis of four composite statistical anomaly scores (`score_max_z`, `score_rms_z`, `score_max_m`, `score_rms_m`) evaluated across the 984 snapshots of **IMS Bearing Dataset (Set 2)**.

The objective is to analyze score distributions, compare standard parametric scores against robust non-parametric scores, examine threshold exceedances ($T=3.0, 3.5$), and evaluate consecutive exceedance run behaviors.

> **Crucial Methodological Caveats & Scope Boundaries:**
> 1. **Exploratory Analysis Only:** This report presents descriptive empirical observations. It is **not** an operationally validated failure-detection system.
> 2. **Unverified Health Assumptions:** Snapshots 0–159 (baseline fitting) and snapshots 160–199 (reference evaluation) are chronological reference splits. Their physical health status is an unverified hypothesis.
> 3. **Provisional Reference Boundaries:** Thresholds $3.0$ and $3.5$ are uncalibrated exploratory reference lines, **not** validated operational failure thresholds.
> 4. **No Failure Prediction or RUL Claims:** IMS Set 2 documents a single run-to-failure trajectory (bearing 1 outer race defect). Results do not constitute failure prediction, false alarm rate validation, or Remaining Useful Life (RUL) estimation.

---

## 2. Dataset & Chronological Partitions

The source dataset [`data/processed/baseline_scores_set2.csv`](file:///d:/Projects/MachineMind%20AI/ml-service/data/processed/baseline_scores_set2.csv) contains **984 snapshots** and 8 columns (`file_index`, `filename`, `timestamp`, `is_valid`, `score_max_z`, `score_rms_z`, `score_max_m`, `score_rms_m`).

Chronological Partitions:
- **Baseline Fitting Period (Snapshots 0–159, N=160, ~26.7 hrs):** Used strictly to compute frozen baseline parameters (mean, std, median, MAD).
- **Reference Evaluation Period (Snapshots 160–199, N=40, ~6.7 hrs):** Evaluated with frozen parameters to check score stability on unseen baseline data.
- **Sequential Evaluation Period (Snapshots 200–983, N=784, ~130.7 hrs):** Evaluated sequentially to track long-term signal deviation to test completion.

---

## 3. Score Distribution Statistics by Partition

Descriptive statistics calculated across the three chronological partitions:

| Partition | Metric | Count | Mean | Median | Std | Min | Max | P90 | P95 | P99 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline Fitting (0-159) | `score_max_z` | 160 | 2.23 | 2.04 | 1.05 | 0.95 | 11.41 | 2.92 | 3.32 | 6.00 |
| Baseline Fitting (0-159) | `score_rms_z` | 160 | 0.92 | 0.86 | 0.38 | 0.46 | 4.45 | 1.15 | 1.29 | 2.19 |
| Baseline Fitting (0-159) | `score_max_m` | 160 | 2.73 | 2.31 | 2.31 | 1.27 | 27.48 | 3.74 | 4.43 | 9.81 |
| Baseline Fitting (0-159) | `score_rms_m` | 160 | 1.09 | 0.97 | 0.72 | 0.61 | 9.00 | 1.32 | 1.54 | 2.92 |
| Reference Evaluation (160-199) | `score_max_z` | 40 | 2.30 | 2.20 | 0.52 | 1.59 | 4.04 | 2.86 | 2.95 | 3.88 |
| Reference Evaluation (160-199) | `score_rms_z` | 40 | 0.97 | 0.95 | 0.20 | 0.59 | 1.45 | 1.26 | 1.30 | 1.39 |
| Reference Evaluation (160-199) | `score_max_m` | 40 | 2.61 | 2.32 | 0.81 | 1.73 | 6.19 | 3.56 | 3.64 | 5.28 |
| Reference Evaluation (160-199) | `score_rms_m` | 40 | 1.09 | 1.07 | 0.23 | 0.71 | 1.92 | 1.36 | 1.40 | 1.80 |
| Sequential Evaluation (200-983) | `score_max_z` | 784 | 49.17 | 11.61 | 147.01 | 1.36 | 2238.36 | 98.86 | 135.21 | 749.32 |
| Sequential Evaluation (200-983) | `score_rms_z` | 784 | 12.30 | 3.43 | 28.50 | 0.64 | 400.94 | 26.37 | 37.88 | 149.30 |
| Sequential Evaluation (200-983) | `score_max_m` | 784 | 55.69 | 11.52 | 167.90 | 1.36 | 2554.71 | 112.92 | 154.41 | 855.29 |
| Sequential Evaluation (200-983) | `score_rms_m` | 784 | 13.01 | 3.50 | 31.35 | 0.70 | 449.17 | 27.48 | 39.81 | 163.39 |

---

## 4. Complete Threshold Exceedance Analysis ($T=3.0, T=3.5$)

Complete $4 \times 3 \times 2 = 24$ combination table listing snapshot counts and percentages exceeding provisional reference boundaries:

| Partition | Metric | Threshold ($T$) | Count Above | Total Snapshots | Percentage Above (%) |
|---|---|---|---|---|---|
| Baseline Fitting (0-159) | `score_max_z` | 3.0 | 14 | 160 | 8.75% |
| Baseline Fitting (0-159) | `score_max_z` | 3.5 | 7 | 160 | 4.38% |
| Baseline Fitting (0-159) | `score_rms_z` | 3.0 | 2 | 160 | 1.25% |
| Baseline Fitting (0-159) | `score_rms_z` | 3.5 | 1 | 160 | 0.62% |
| Baseline Fitting (0-159) | `score_max_m` | 3.0 | 31 | 160 | 19.38% |
| Baseline Fitting (0-159) | `score_max_m` | 3.5 | 19 | 160 | 11.88% |
| Baseline Fitting (0-159) | `score_rms_m` | 3.0 | 2 | 160 | 1.25% |
| Baseline Fitting (0-159) | `score_rms_m` | 3.5 | 2 | 160 | 1.25% |
| Reference Evaluation (160-199) | `score_max_z` | 3.0 | 2 | 40 | 5.00% |
| Reference Evaluation (160-199) | `score_max_z` | 3.5 | 2 | 40 | 5.00% |
| Reference Evaluation (160-199) | `score_rms_z` | 3.0 | 0 | 40 | 0.00% |
| Reference Evaluation (160-199) | `score_rms_z` | 3.5 | 0 | 40 | 0.00% |
| Reference Evaluation (160-199) | `score_max_m` | 3.0 | 10 | 40 | 25.00% |
| Reference Evaluation (160-199) | `score_max_m` | 3.5 | 5 | 40 | 12.50% |
| Reference Evaluation (160-199) | `score_rms_m` | 3.0 | 0 | 40 | 0.00% |
| Reference Evaluation (160-199) | `score_rms_m` | 3.5 | 0 | 40 | 0.00% |
| Sequential Evaluation (200-983) | `score_max_z` | 3.0 | 532 | 784 | 67.86% |
| Sequential Evaluation (200-983) | `score_max_z` | 3.5 | 482 | 784 | 61.48% |
| Sequential Evaluation (200-983) | `score_rms_z` | 3.0 | 408 | 784 | 52.04% |
| Sequential Evaluation (200-983) | `score_rms_z` | 3.5 | 390 | 784 | 49.74% |
| Sequential Evaluation (200-983) | `score_max_m` | 3.0 | 574 | 784 | 73.21% |
| Sequential Evaluation (200-983) | `score_max_m` | 3.5 | 511 | 784 | 65.18% |
| Sequential Evaluation (200-983) | `score_rms_m` | 3.0 | 410 | 784 | 52.30% |
| Sequential Evaluation (200-983) | `score_rms_m` | 3.5 | 392 | 784 | 50.00% |

---

## 5. Consecutive Exceedance Analysis

A **consecutive exceedance run** is defined as a contiguous sequence of snapshot indices $[i_{\text{start}}, i_{\text{end}}]$ where every snapshot score strictly exceeds the given threshold ($T > 3.0$ or $T > 3.5$).

### Duration Conventions & Definitions
To ensure mathematical precision, two distinct duration conventions are defined and reported:

1. **Snapshot Coverage Duration:**
   $$\text{Coverage Duration} = \text{Run Length (Snapshots)} \times 10 \text{ minutes}$$
   *Measures the total physical sampling window represented by $N$ consecutive 10-minute snapshot intervals.* For a run of 453 snapshots, $453 \times 10 = 4,530 \text{ minutes} = 75.50 \text{ hours}$ (75 hours 30 minutes).

2. **Elapsed Timestamp Span:**
   $$\text{Elapsed Span} = (\text{Run Length (Snapshots)} - 1) \times 10 \text{ minutes} = \text{Timestamp}_{\text{end}} - \text{Timestamp}_{\text{start}}$$
   *Measures the exact time elapsed between the timestamp of the first snapshot in the run and the timestamp of the final snapshot.* For a run of 453 snapshots (indices 531 to 983, `2004-02-16 03:02:39` to `2004-02-19 06:22:39`), $(453 - 1) \times 10 = 4,520 \text{ minutes} = 75.33 \text{ hours}$ (75 hours 20 minutes).

> **Explanatory Note:** The Elapsed Timestamp Span is exactly one 10-minute sampling interval shorter than the Snapshot Coverage Duration ($10 \text{ minutes} = 0.17 \text{ hours}$) because $N$ snapshot points contain $(N - 1)$ intervals between them.

Complete 8-combination consecutive exceedance table with both duration metrics:

| Metric | Threshold ($T$) | Num Runs | Longest Run Length | Start Index | End Index | Snapshot Coverage Duration | Elapsed Timestamp Span |
|---|---|---|---|---|---|---|---|
| `score_max_z` | 3.0 | 68 | 453 | 531 | 983 | 75.50 hrs (4530 min) | 75.33 hrs (4520 min) |
| `score_max_z` | 3.5 | 38 | 453 | 531 | 983 | 75.50 hrs (4530 min) | 75.33 hrs (4520 min) |
| `score_rms_z` | 3.0 | 7 | 399 | 585 | 983 | 66.50 hrs (3990 min) | 66.33 hrs (3980 min) |
| `score_rms_z` | 3.5 | 7 | 383 | 601 | 983 | 63.83 hrs (3830 min) | 63.67 hrs (3820 min) |
| `score_max_m` | 3.0 | 101 | 453 | 531 | 983 | 75.50 hrs (4530 min) | 75.33 hrs (4520 min) |
| `score_max_m` | 3.5 | 67 | 453 | 531 | 983 | 75.50 hrs (4530 min) | 75.33 hrs (4520 min) |
| `score_rms_m` | 3.0 | 6 | 407 | 577 | 983 | 67.83 hrs (4070 min) | 67.67 hrs (4060 min) |
| `score_rms_m` | 3.5 | 8 | 384 | 600 | 983 | 64.00 hrs (3840 min) | 63.83 hrs (3830 min) |

## 6. First Observed Threshold Crossings

First snapshot index, timestamp, exact score value, and partition where each score exceeds $T=3.0$ and $T=3.5$:

| Metric | Threshold ($T$) | First Index | Timestamp | Score Value | Partition |
|---|---|---|---|---|---|
| `score_max_z` | 3.0 | 0 | `2004-02-12 10:32:39` | 11.4089 | Baseline Fitting (0-159) |
| `score_max_z` | 3.5 | 0 | `2004-02-12 10:32:39` | 11.4089 | Baseline Fitting (0-159) |
| `score_rms_z` | 3.0 | 0 | `2004-02-12 10:32:39` | 4.4525 | Baseline Fitting (0-159) |
| `score_rms_z` | 3.5 | 0 | `2004-02-12 10:32:39` | 4.4525 | Baseline Fitting (0-159) |
| `score_max_m` | 3.0 | 0 | `2004-02-12 10:32:39` | 27.4768 | Baseline Fitting (0-159) |
| `score_max_m` | 3.5 | 0 | `2004-02-12 10:32:39` | 27.4768 | Baseline Fitting (0-159) |
| `score_rms_m` | 3.0 | 0 | `2004-02-12 10:32:39` | 9.0038 | Baseline Fitting (0-159) |
| `score_rms_m` | 3.5 | 0 | `2004-02-12 10:32:39` | 9.0038 | Baseline Fitting (0-159) |

> **Key Observation on First Crossings vs. Sustained Degradation:**
> All four scores exceed $T=3.0$ and $T=3.5$ at **Snapshot 0** (`2004-02-12 10:32:39`) due to an isolated startup transient (`score_rms_m` = 9.00). Following this initial transient, scores quickly settle back to baseline levels (~1.0).
> The **first sustained exceedance run** in the sequential evaluation partition begins at:
> - **Snapshot 531** for `score_max_z` ($T=3.0$, duration 453 snapshots / 75.5 hrs).
> - **Snapshot 577** for `score_rms_m` ($T=3.0$, duration 407 snapshots / 67.83 hrs).
> - **Snapshot 585** for `score_rms_z` ($T=3.0$, duration 399 snapshots / 66.5 hrs).

---

## 7. Quantitative Comparison: Standard vs. Robust Scores

1. **RMS Composite Pair (`score_rms_z` vs. `score_rms_m`):**
   - **Pearson Correlation:** $r = 0.999637$ ($p < 0.001$) across all 984 snapshots ($r = 0.9627$ in Baseline Fit, $r = 0.9228$ in Reference Eval, $r = 0.9997$ in Sequential Eval).
   - Both metrics track overall system energy deviation closely, with modified Z-score exhibiting slightly higher median baseline values (1.09 vs 0.92 in fitting).

2. **Max Feature Pair (`score_max_z` vs. `score_max_m`):**
   - **Pearson Correlation:** $r = 0.999973$ ($p < 0.001$) across all 984 snapshots ($r = 0.9427$ in Baseline Fit, $r = 0.7817$ in Reference Eval, $r = 0.99998$ in Sequential Eval).
   - Modified max Z-score (`score_max_m`) exhibits higher peak sensitivity to transient individual feature spikes (e.g., snapshot 0 reaches 27.48 vs 11.41) due to MAD scaling on narrow-IQR features.

---

## 8. Figure Artifacts

The following analysis figures are saved under [`reports/figures/baseline_analysis/`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/figures/baseline_analysis/):

1. [`reports/figures/baseline_analysis/score_trajectories_comparison.png`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/figures/baseline_analysis/score_trajectories_comparison.png)
2. [`reports/figures/baseline_analysis/score_distributions_by_partition.png`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/figures/baseline_analysis/score_distributions_by_partition.png)
3. [`reports/figures/baseline_analysis/threshold_exceedances_seq_eval.png`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/figures/baseline_analysis/threshold_exceedances_seq_eval.png)
4. [`reports/figures/baseline_analysis/consecutive_exceedance_runs.png`](file:///d:/Projects/MachineMind%20AI/ml-service/reports/figures/baseline_analysis/consecutive_exceedance_runs.png)

---

## 9. Methodological Limitations & Future Scope

- **Unverified Baseline Health:** The reference fitting and evaluation periods represent chronological splits; their physical health is unverified.
- **Uncalibrated Thresholds:** Boundaries 3.0 and 3.5 are exploratory mathematical lines, not calibrated failure alerts.
- **Single Failure Dataset:** IMS Set 2 features a single outer-race defect trajectory; no failure time or RUL estimation is established.
