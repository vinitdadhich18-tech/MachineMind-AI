# Exploratory Data Analysis (EDA) Report — NASA IMS Bearing Dataset (Set 2)

**Project:** MachineMind AI  
**Component:** `ml-service`  
**Phase:** Phase 3 — Exploratory Data Analysis (EDA)  
**Date:** 2026-09-26  
**Document Status:** Complete EDA Report (Verified via Full Dataset Scan & Notebook Execution)

---

## 1. Executive Summary

This report documents the **Exploratory Data Analysis (EDA)** performed on **Set 2** of the **NASA IMS Bearing Dataset**. The primary goal of Phase 3 is to evaluate signal data quality, verify timestamp cadence, inspect time-domain vibration waveforms, and trace chronological statistical trends across all 984 snapshot recordings without altering raw data or making premature claims about failure onset.

### Key Summary Findings
- **Data Quality (100% Valid):** All 984 raw snapshot text files passed the automated quality scan. Exactly $80,609,280$ scalar numeric values ($984 \text{ files} \times 20,480 \text{ rows} \times 4 \text{ channels}$) were validated with **zero missing (NaN) values**, **zero infinite values**, **zero corrupted files**, and **zero duplicate timestamps**.
- **Cadence & Timeline:** All 983 consecutive recording intervals equal **exactly 600.0 seconds (10.0 minutes)**. The dataset represents a continuous 163.83-hour test run from `2004-02-12 10:32:39` to `2004-02-19 06:22:39` with zero recording pauses or day gaps.
- **Observed Signal Evolution:** Channel 1 exhibits a notable increase in standard deviation and RMS amplitude during late snapshots, peaking at file index 979 (`2004-02-19 05:42:39`, $\text{std} = 0.7250$). In the final two snapshots (`06:12:39` and `06:22:39`), all four channels drop to a uniform low amplitude ($\text{std} \approx 0.0010$).

---

## 2. Dataset Quality & Integrity Audit

### 2.1 Comprehensive Scan Results (All 984 Files)

An exhaustive programmatic scan was conducted across every file in `data/raw/IMS/2nd_test/2nd_test/`:

| Data Quality Check | Expectation | Observed Result | Status |
|---|---|---|---|
| **Total Snapshot Files** | 984 ASCII files | **984 files** | PASS |
| **Matrix Dimensions** | 20,480 rows × 4 columns | **(20480, 4) for all 984 files** | PASS |
| **Data Types** | Numeric floating-point (`float64`) | **`float64` for all channels** | PASS |
| **Missing Values (`NaN`)** | 0 missing values | **0 NaNs across all 80.6M points** | PASS |
| **Infinite Values (`Inf`)** | 0 positive/negative infinity | **0 Infs across all 80.6M points** | PASS |
| **Unreadable / Empty Files** | 0 corrupted files | **0 unreadable files** | PASS |
| **Duplicate Timestamps** | 0 duplicate timestamps | **0 duplicate timestamps** | PASS |

---

## 3. Timestamp & Cadence Analysis

### 3.1 Chronological Recording Cadence
- **Earliest Timestamp [VERIFIED]:** `2004-02-12 10:32:39`
- **Latest Timestamp [VERIFIED]:** `2004-02-19 06:22:39`
- **Total Duration:** 163 hours, 50 minutes (163.83 hours)
- **Consecutive Intervals Count:** 983 total intervals
- **Cadence Statistics:**
  - Minimum Interval: **10.0 minutes** ($600.0\text{ s}$)
  - Maximum Interval: **10.0 minutes** ($600.0\text{ s}$)
  - Mode Interval: **10.0 minutes** ($600.0\text{ s}$)
  - Intervals matching 10-minute cadence: **983 / 983 (100.0%)**

> **Implication for Signal Processing:** Because timestamp cadence is 100% uniform, time-domain feature trends can be indexed directly by snapshot sequence without time interpolation or non-uniform resampling.

---

## 4. Time-Domain Signal Observations & Chronological Waveform Inspection

### 4.1 Single Snapshot Baseline (First File: `2004.02.12.10.32.39`)

The initial snapshot represents baseline operational vibration:

| Channel | Mean (DC Offset) | Std Dev (`std`) | Min | Max | RMS | Skewness | Kurtosis (Fisher) |
|---|---|---|---|---|---|---|---|
| **Channel 1** | -0.010196 | 0.073477 | -0.386 | 0.454 | 0.074179 | 0.084000 | 0.629209 |
| **Channel 2** | -0.012695 | 0.090056 | -0.513 | 0.464 | 0.090944 | 0.126924 | 0.507217 |
| **Channel 3** | -0.014541 | 0.108436 | -0.911 | 1.023 | 0.109404 | 0.204855 | 3.214152 |
| **Channel 4** | -0.010026 | 0.053168 | -0.264 | 0.193 | 0.054103 | -0.022082 | 0.066268 |

*Observation:* All channels contain a slight negative static DC offset ($\text{mean} \approx -0.010$).

### 4.2 Chronological Waveform Progression Across 4 Stages

Snapshots at indices 0 (First), 328 (~33%), 656 (~66%), and 983 (Final) were compared:
- **First Stage (0%, File 0):** Stable background vibration ($\text{std} \approx 0.05 - 0.10$).
- **Middle-Early Stage (~33%, File 328):** Signal amplitudes remain comparable to baseline ($\text{std} = 0.0776$ on Channel 1).
- **Middle-Late Stage (~66%, File 656):** Channel 1 amplitude shows moderate expansion ($\text{std} = 0.1060$).
- **Final Stage (100%, File 983):** All four channels show a low amplitude ($\text{std} \approx 0.0010$).

---

## 5. Chronological Trend Analysis Across All 984 Snapshots

Saved figures in `reports/figures/eda/`:
- `trend_rms.png` (RMS amplitude trend across all 984 files)
- `trend_std.png` (Standard deviation trend across all 984 files)
- `trend_mean.png` (Mean DC offset trend across all 984 files)

### 5.1 Observed Numerical Findings
1. **Initial Baseline Phase (Files 0 to ~700):** RMS amplitude remains stable across all channels (Channel 1 RMS $\approx 0.07 - 0.10$).
2. **Late Amplitude Growth Phase (Files ~700 to 981):** Channel 1 RMS increases significantly, reaching a peak standard deviation of **`0.7250`** at file index 979 (`2004-02-19 05:42:39`) with maximum peak amplitudes reaching `4.998`.
3. **Tail Low-Amplitude Snapshots (Files 982 & 983):** In the final two files (`06:12:39` and `06:22:39`), standard deviation across all four channels drops to $\approx 0.0010$.

---

## 6. Strict Separation of Observations vs. Hypotheses

To preserve scientific rigor, empirical facts are strictly separated from unconfirmed hypotheses:

| Category | Finding / Statement | Scientific Status |
|---|---|---|
| **Observed Fact** | Channel 1 RMS and std increase late in the run, peaking at index 979 (`05:42:39`). | **[VERIFIED]** |
| **Observed Fact** | All 4 channels drop to $\text{std} \approx 0.0010$ in the final two snapshots (`06:12:39` and `06:22:39`). | **[VERIFIED]** |
| **Observed Fact** | Mean static DC offset remains near $-0.010$ throughout early and middle snapshots. | **[VERIFIED]** |
| **Hypothesis** | Late amplitude growth in Channel 1 corresponds to bearing outer-race defect progression. | **[UNCONFIRMED HYPOTHESIS]** |
| **Hypothesis** | Low amplitude in final two snapshots represents post-failure machine shutdown or sensor disconnect. | **[UNCONFIRMED HYPOTHESIS]** |

> **Safeguard:** We do not assert exact failure onset timing or machine shutdown state as proven facts. These are hypotheses to be evaluated against baseline threshold models in Phase 6.

---

## 7. Scientific Limitations & Implications for Preprocessing (Phase 4)

### 7.1 Key Scientific Limitations
1. **Absence of Ground-Truth Health Labels:** The dataset provides no per-file health/fault labels. Any definition of a "healthy period" is an explicit analytical assumption.
2. **Uncalibrated Sensor Units:** Acceleration units ($g$ vs. voltage) are unspecified in raw text files. Amplitude metrics represent arbitrary relative units.
3. **Single Failure Run:** Set 2 documents one degradation run, making evaluation a single-case study.

### 7.2 Recommendations for Preprocessing (Phase 4)
1. **DC Offset Removal:** Subtract per-snapshot mean ($\text{mean} \approx -0.010$) to center signals at zero prior to feature calculation.
2. **Feature Extraction Focus:** Standard deviation, RMS, peak-to-peak amplitude, skewness, and kurtosis provide clear trend signals for tracking anomaly progression.
3. **Baseline Training Period:** Use the initial stable phase (e.g. first 200–300 snapshots) to fit baseline scalers and statistical anomaly thresholds in Phase 6.
