# Baseline Evaluation & Next-Stage Planning Report — NASA IMS Bearing Dataset (Set 2)

**Project:** MachineMind AI  
**Component:** `ml-service`  
**Phase:** Phase 6, Step 5 — Baseline Evaluation & Next-Stage Planning  
**Date:** 2026-09-28  
**Document Status:** Complete Baseline Evaluation & Synthesis Report  

---

## 1. Executive Summary & Objective

This report evaluates the statistical baseline developed in Phase 6 for **Set 2 of the NASA IMS Bearing Dataset**. 

The goal of Step 5 is to synthesize what the non-machine-learning statistical baseline demonstrates, systematically examine its scientific and technical limitations, clearly separate supported conclusions from unsupported claims, and define a defensible validation plan for subsequent machine learning model development.

> **Crucial Methodological Scope Reminder:**
> This document is an evaluation and planning synthesis. It does **not** train new models, alter baseline parameters, or claim operationally validated failure-detection performance.

---

## 2. Existing Baseline Methodology Summary

The statistical baseline module [`src/baseline.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/baseline.py) implements a non-ML, scale-guarded statistical reference model:

1. **Chronological Partitioning (No Temporal Data Leakage):**
   - **Baseline Fitting Period (Snapshots 0–159, N=160, ~26.7 hrs):** Used strictly to compute frozen baseline distribution parameters ($\mu, \sigma, \text{median}, \text{MAD}, \text{IQR}, \text{P2P}$).
   - **Reference Evaluation Period (Snapshots 160–199, N=40, ~6.7 hrs):** Evaluated with frozen parameters to test baseline stability on unseen reference data.
   - **Sequential Evaluation Period (Snapshots 200–983, N=784, ~130.7 hrs):** Evaluated sequentially with frozen parameters to track long-term degradation trajectories to test completion.

2. **Scale-Aware Guarding & Feature Eligibility:**
   Lower variance cutoffs ($\text{Cutoff}_{\text{std}} = \max(10^{-6}, 10^{-4} \cdot \text{P2P}_{\text{fit}})$ and $\text{Cutoff}_{\text{mad}} = \max(10^{-6}, 10^{-4} \cdot \text{IQR}_{\text{fit}})$) prevent zero-division. All 36 features across 4 channels exhibited non-zero variance during fitting and were eligible.

3. **Four Composite Anomaly Metrics:**
   - `score_max_z`: Maximum absolute Z-score across 36 features.
   - `score_rms_z`: RMS composite Z-score across 36 features ($\sqrt{\frac{1}{36}\sum Z_i^2}$).
   - `score_max_m`: Maximum absolute modified Z-score across 36 features.
   - `score_rms_m`: RMS composite modified Z-score across 36 features ($\sqrt{\frac{1}{36}\sum M_i^2}$).

---

## 3. Main Observed Patterns

1. **Baseline Stability (Snapshots 0 to 199):**
   - During fitting (0–159) and reference evaluation (160–199), composite RMS scores (`score_rms_z` and `score_rms_m`) fluctuated closely around **~0.92 – 1.09** with zero exceedance of exploratory reference line $T=3.0$ during the reference evaluation period.
2. **Initial Transient Spike at Snapshot 0:**
   - Snapshot 0 (`2004-02-12 10:32:39`) exhibited an isolated spike (`score_rms_m` = 9.00), representing a startup mechanical/electrical transient before settling back to normal levels (~1.0).
3. **Emergence of Sustained Degradation (~Snapshot 531 to 983):**
   - The first sustained run exceeding $T=3.0$ begins at **Snapshot 531** (`score_max_z`) and **Snapshots 577–585** (`score_rms_m` / `score_rms_z`), lasting continuously for **399 to 453 snapshots** (~66.5 to 75.5 hours) until test completion.
4. **Extreme Pearson Correlation ($r > 0.999$):**
   - Standard Z-score and robust modified Z-score composite metrics correlate almost perfectly ($r = 0.9996$ for RMS pair, $r = 0.99997$ for Max pair across 984 snapshots).

---

## 4. Methodological & Technical Limitations

1. **Unverified Baseline Health Status:**
   The assumption that snapshots 0–159 represent un-degraded, healthy machinery is an unverified hypothesis. No physical inspection records or ground-truth health labels exist per snapshot.
2. **Exploratory Reference Thresholds ($T=3.0, 3.5$):**
   The thresholds 3.0 and 3.5 are mathematical reference lines, **not** operationally calibrated alarm thresholds. False alarm rates and detection precision under field operating conditions are unverified.
3. **Single Run-to-Failure Trajectory:**
   IMS Set 2 documents a single failure run (bearing 1 outer-race defect under constant speed and load). Evaluation on a single trajectory cannot establish statistical generalization across different machines, loads, or failure modes.
4. **Sensitivity of Max Features to Narrow-IQR Signals:**
   Max metrics (`score_max_m`) are sensitive to transient feature spikes on features with very low baseline dispersion (IQR).

---

## 5. Supported Conclusions vs. Unsupported Claims

### A. Directly Supported Conclusions (Confirmed by Empirical Evidence)
- The statistical baseline provides a deterministic, zero-leakage benchmark for feature deviation.
- Composite RMS scores effectively smooth single-channel noise, providing a stable baseline mean of ~0.92–1.09.
- A clear, multi-day sustained statistical deviation occurs starting around snapshot 531–585 and persists until test termination.
- Standard Z-score and modified Z-score composite metrics produce virtually identical overall degradation trajectories ($r > 0.999$).

### B. Unsupported Claims (Explicitly Excluded from Project Scope)
- **NO Failure Prediction or RUL Claims:** The baseline anomaly score quantifies statistical distance from initial behavior; it does **not** predict remaining useful life (RUL) or time-to-failure.
- **NO Exact Failure Onset Claim:** Snapshot 531 is the first observed sustained threshold crossing under an arbitrary line ($T=3.0$); it is **not** a confirmed physical failure onset date.
- **NO Operational Reliability Claim:** Results from laboratory data under constant load cannot be generalized to industrial machinery subject to variable operational regimes.

---

## 6. Proposed Validation Plan for Next Phase (Phase 7: ML Model Development)

To evaluate unsupervised machine learning models (e.g., **Isolation Forest**, **One-Class SVM**, or **Mahalanobis Distance**) in Phase 7 against this statistical baseline, the following validation framework is recommended:

1. **Strict Chronological Split Enforcement:**
   - All ML models, scalers, and dimensionality reduction tools must be trained **strictly** on the baseline fitting partition (snapshots 0–159).
   - Hyperparameter tuning and validation checks must use the reference evaluation partition (snapshots 160–199).
   - Sequential evaluation partition (snapshots 200–983) must remain strictly held back until final scoring.

2. **Fitting Window Sensitivity Analysis:**
   - Test model robustness across alternative baseline fitting window sizes (e.g., 10% / 100 snapshots, 15% / 150 snapshots, 20% / 200 snapshots) to verify parameter stability.

3. **Persistence-Based Alert Evaluation ($N$-Consecutive Exceedances):**
   - Evaluate persistence filtering ($N \ge 3, 5, 10$ consecutive exceedances) to quantify the reduction of isolated false-trigger spikes (such as snapshot 0) while preserving early detection lead time.

4. **Multi-Channel & Feature-Subset Sensitivity:**
   - Compare full 36-feature inputs against channel-specific or time-domain-only feature subsets to determine feature redundancy.

---

## 7. Recommended Next Step & Justification

### Recommended Action: Proceed to **Phase 7 — ML Model Development**
- **Justification:** The statistical baseline is fully implemented, verified, audited, and documented. We have established a transparent benchmark ($r > 0.999$, stable RMS baseline ~1.0, sustained elevation at snapshot 531–585). We are now prepared to build unsupervised ML anomaly detection algorithms (such as Isolation Forest) to evaluate whether machine learning provides superior feature interaction modeling or earlier anomaly detection relative to this statistical baseline.

---

## 8. Open Questions & Key Assumptions

1. **Unverified Sensor Calibration:** Absolute vibration amplitudes remain in uncalibrated sensor voltage units; all models must rely on relative statistical scale growth.
2. **Shared Shaft Coupling:** All 4 bearings share a single shaft under 6,000 lb load; signals on bearings 2–4 may reflect cross-talk from bearing 1's outer race defect.
