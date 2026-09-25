# Signal Preprocessing Notes & Strategy Report — NASA IMS Bearing Dataset (Set 2)

**Project:** MachineMind AI  
**Component:** `ml-service`  
**Phase:** Phase 4 — Signal Preprocessing  
**Date:** 2026-09-26  
**Document Status:** Complete Preprocessing & Windowing Documentation (Foundation, Manifest, & Windowing Utility)

---

## 1. Executive Summary

This report documents the design, implementation, and empirical verification of the **Signal Preprocessing & Windowing Foundation** for **Set 2** of the **NASA IMS Bearing Dataset**.

The objective of Phase 4 is to establish a reproducible, deterministic preprocessing and windowing pipeline that creates a dataset manifest, provides optional mean-centering (DC-offset removal), and enables parameter-driven signal windowing without applying harmful per-file normalization or unapproved digital filtering.

---

## 2. Dataset Manifest Audit (`data/processed/manifest_set2.csv`)

### 2.1 Manifest Structure
The dataset manifest is stored at [`ml-service/data/processed/manifest_set2.csv`](file:///d:/Projects/MachineMind%20AI/ml-service/data/processed/manifest_set2.csv). It contains 984 rows sorted chronologically by parsed timestamp.

| Column Name | Data Type | Description |
|---|---|---|
| `file_index` | `int64` | Chronological sequence index (0 to 983). |
| `filename` | `string` | Original snapshot ASCII text file name (`YYYY.MM.DD.HH.MM.SS`). |
| `timestamp` | `string` | Parsed ISO timestamp (`YYYY-MM-DD HH:MM:SS`). |
| `file_size_bytes` | `int64` | Raw file size in bytes (stat check). |
| `rows` | `int64` | Row count per file (Expected: 20,480). |
| `cols` | `int64` | Column count per file (Expected: 4). |
| `is_valid` | `boolean` | Structural validity flag (`True` = file readable with valid matrix shape). |
| `notes` | `string` | Descriptive audit notes (flags unusual signal properties without altering file status). |

### 2.2 Empirical Manifest Summary
- **Total Registered Snapshots [VERIFIED]:** 984
- **Structurally Valid Files (`is_valid = True`) [VERIFIED]:** 984 / 984 (100.0%)
- **Notes Breakdown:**
  - `Structurally valid snapshot.`: **982 files**
  - `Structurally valid snapshot; Note: Low-amplitude signal observed (all channel std < 0.005).`: **2 files** (final two snapshots `2004.02.19.06.12.39` and `2004.02.19.06.22.39`).

> **Flagging Principle:** Low-amplitude snapshots are flagged neutrally in `notes` while keeping `is_valid = True`. No file is deleted, moved, or altered.

---

## 3. Preprocessing Functions (`src/preprocessing.py`)

The preprocessing module lives in [`ml-service/src/preprocessing.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/preprocessing.py).

### 3.1 Implemented Functions
1. `create_dataset_manifest(raw_dir, output_csv_path, low_std_threshold)`:
   Scans raw snapshot files, validates matrix shapes, logs observation notes, and outputs `manifest_set2.csv` deterministically.
2. `remove_dc_offset(df)`:
   Subtracts the arithmetic mean independently for each channel ($x_{\text{centered}} = x - \mu_{\text{snapshot}}$). Returns a new DataFrame without mutating the input DataFrame.
3. `process_snapshot(file_path, remove_dc=False)`:
   Loads a snapshot and optionally applies `remove_dc_offset()`.
4. `extract_windows(df, window_length, overlap_ratio, drop_incomplete)`:
   Slices a snapshot DataFrame into contiguous sub-windows while preserving all 4 channels and original column names.

---

## 4. Windowing Strategy & Trade-offs Analysis

### 4.1 Window Length Comparison ($N_{\text{window}}$ at $20.48\text{ kHz}$)

| Window Length ($N_{\text{window}}$) | Duration (ms) | Frequency Resolution ($\Delta f$) | Windows / Snapshot | Total Dataset Windows | Trade-offs & Suitability |
|---|---|---|---|---|---|
| **1,024 samples** | $\approx 50\text{ ms}$ | $20.0\text{ Hz}$ | 20 windows | 19,680 | High time resolution; higher computational overhead and statistical variance per window. |
| **2,048 samples** | $\approx 100\text{ ms}$ | $10.0\text{ Hz}$ | 10 windows | 9,840 | **Recommended Sub-windowing:** Excellent balance of statistical stability and frequency resolution. |
| **4,096 samples** | $\approx 200\text{ ms}$ | $5.0\text{ Hz}$ | 5 windows | 4,920 | High statistical stability; lower time resolution. |
| **20,480 samples** | $1,000\text{ ms}$ | $1.0\text{ Hz}$ | 1 window | 984 | **Recommended Baseline:** 1 window per snapshot; zero segmentation overhead, matches 10-minute snapshot recording interval. |

### 4.2 Overlapping vs. Non-Overlapping Windows
- **Non-overlapping ($0\%$ overlap):** Step size $S = N_{\text{window}}$. Zero data redundancy, 10 windows per snapshot ($N=2048$).
- **Overlapping ($50\%$ overlap):** Step size $S = 1024$. Smooths transitions and captures impact events near window boundaries, yielding 19 windows per snapshot ($N=2048$).

---

## 5. Distinction Between Raw & Centered Signals

### 5.1 Statistical Comparison Across Representative Snapshots

| Stage | Snapshot | Representation | Channel 1 Mean | Channel 1 Std | Channel 1 RMS | Peak-to-Peak (P2P) |
|---|---|---|---|---|---|---|
| **Early Baseline** | `2004.02.12.10.32.39` | **Raw** | -0.010196 | **0.073477** | 0.074179 | **0.840000** |
| | | **Centered** | 0.000000 | **0.073477** | 0.073477 | **0.840000** |
| **High Amplitude** | `2004.02.19.05.42.39` | **Raw** | -0.010972 | **0.725049** | 0.725114 | **9.996000** |
| | | **Centered** | 0.000000 | **0.725049** | 0.725049 | **9.996000** |
| **Final Low Amplitude** | `2004.02.19.06.22.39` | **Raw** | -0.001162 | **0.001000** | 0.001533 | **0.007000** |
| | | **Centered** | 0.000000 | **0.001000** | 0.001000 | **0.007000** |

### 5.2 Impact of Mean Centering on Metrics
- **Mean:** Forced to $0.000000$ for all channels.
- **Standard Deviation (`std`):** **100% Identical** before and after mean centering.
- **Peak-to-Peak (P2P):** **100% Identical** before and after mean centering.
- **RMS:** Shifted slightly from $\sqrt{\mu^2 + \sigma^2}$ to $\sigma$ (removing DC offset energy).

---

## 6. Preprocessing Constraints & Safeguards

- **NO Per-Snapshot Normalization:** Per-file z-score scaling ($x / \sigma$) is strictly prohibited because it sets $\sigma = 1.0$ for all files, destroying the RMS degradation trend.
- **NO Digital Filtering:** No low-pass or high-pass filtering is applied, preserving the full $0–10.24\text{ kHz}$ Nyquist spectrum.
- **NO Assumptions of Shutdown or Failure Timing:** Low-amplitude tail snapshots and late amplitude spikes are documented as numerical facts without claiming confirmed machine shutdown or failure onset timing.

---

## 7. Preprocessing Pipeline Validation & Verification Results

### 7.1 Validation Methodology & Representative Snapshots
The complete preprocessing pipeline (`load_snapshot` -> `manifest validation` -> `remove_dc_offset` -> `extract_windows`) was empirically validated across three representative real dataset snapshots:
1. **Early Baseline Snapshot (`2004.02.12.10.32.39`, File 0):** Unaffected healthy baseline state.
2. **Late High-Amplitude Snapshot (`2004.02.19.06.02.39`, File 981):** High-amplitude vibration state.
3. **Final Low-Amplitude Snapshot (`2004.02.19.06.22.39`, File 983):** Low-amplitude end-of-test state.

### 7.2 Validation Checkpoints & Actual Verification Results

| Checkpoint | Tested Condition | Expected Result | Actual Result | Verification Status |
|---|---|---|---|---|
| **1. File Loading & Shape** | Matrix dimensions per snapshot | `(20480, 4)` | `(20480, 4)` | **PASSED** |
| **2. Channel Ordering & Names** | Channel column standardization | `["Channel_1", "Channel_2", "Channel_3", "Channel_4"]` | `["Channel_1", "Channel_2", "Channel_3", "Channel_4"]` | **PASSED** |
| **3. Manifest Consistency** | Cross-reference with `manifest_set2.csv` | Registered & `is_valid = True` | Registered & `is_valid = True` (File 983 notes flagged low-amplitude tail) | **PASSED** |
| **4. Sub-Window Slicing** | Default $N=2048$, $0\%$ overlap | 10 sub-windows of shape `(2048, 4)` | 10 sub-windows of shape `(2048, 4)` | **PASSED** |
| **5. Exact Sample Reconstruction** | Concatenate non-overlapping windows | `pd.concat(wins).equals(df_raw)` | `True` ($100\%$ sample-exact match across all 20,480 rows) | **PASSED** |
| **6. Raw Data Immutability** | Input DataFrame memory state after processing | Input DataFrame unmodified (`df.equals(df_orig)`) | `True` (Zero mutation) | **PASSED** |
| **7. Error Handling & Validation** | Invalid inputs (string, NaN, negative $N$, out-of-range overlap) | Raise `TypeError` / `ValueError` | Raised expected Python exceptions with informative error messages | **PASSED** |

### 7.3 Quantitative Sub-Window Statistical Range Comparison (Channel 1)

| Snapshot Stage | Snapshot Filename | Raw Snapshot Stats | Sub-Windows Range (10 Sub-Windows, $N=2048$) |
|---|---|---|---|
| **Early Baseline** | `2004.02.12.10.32.39` | Mean: -0.010196<br>Std: 0.073477<br>RMS: 0.074179<br>P2P: 0.840000 | Mean: [-0.012755, -0.008230]<br>Std: [0.070467, 0.078605]<br>RMS: [0.071242, 0.079229] |
| **Late High-Amplitude** | `2004.02.19.06.02.39` | Mean: -0.001703<br>Std: 0.483844<br>RMS: 0.483835<br>P2P: 7.197000 | Mean: [-0.023626, 0.010911]<br>Std: [0.462621, 0.509710]<br>RMS: [0.462616, 0.509703] |
| **Final Low-Amplitude** | `2004.02.19.06.22.39` | Mean: -0.001162<br>Std: 0.001000<br>RMS: 0.001533<br>P2P: 0.007000 | Mean: [-0.001387, -0.000929]<br>Std: [0.000930, 0.001017]<br>RMS: [0.001369, 0.001670] |

---

## 8. Summary of Preprocessing Pipeline Status & Next Steps

Phase 4 (Signal Preprocessing) is fully validated. The pipeline provides:
- A deterministic manifest (`data/processed/manifest_set2.csv`) tracking all 984 snapshots.
- Reproducible DC-offset removal (`remove_dc_offset()`).
- Modular window extraction (`extract_windows()`) supporting parameterizable lengths ($N=2048$ or $N=20480$) and overlap ratios ($0\%$ or $50\%$).
- 100% sample preservation and zero modification of raw files.

**Ready for Phase 5 — Feature Engineering.**

