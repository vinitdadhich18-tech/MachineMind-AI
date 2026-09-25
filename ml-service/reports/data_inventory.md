# Dataset Inventory and Verification Report — NASA IMS Bearing Dataset (Set 2)

**Project:** MachineMind AI  
**Component:** `ml-service`  
**Phase:** Phase 2 — Dataset Acquisition and Organization  
**Date:** 2026-09-25  
**Document Status:** Complete Data Inventory (Verified via Empirical Inspection & Verification Script)

---

## 1. Executive Summary

This report documents the acquisition, storage layout, file integrity, checksums, time series intervals, and initial read-only verification for **Set 2** of the **NASA IMS Bearing Dataset**.

All raw data files have been manually downloaded, organized under `ml-service/data/raw/`, and preserved in their original, unmodified state. Programmatic verification confirms 984 recording files spanning from **2004-02-12 10:32:39** to **2004-02-19 06:22:39** with a strictly consistent sampling interval of **600 seconds (10 minutes)**.

---

## 2. Dataset Acquisition & Immutability Verification

### 2.1 File Provenance & Archives
- **Dataset Source [VERIFIED]:** NASA Prognostics Data Repository (`https://data.nasa.gov/docs/legacy/IMS.zip`).
- **Primary Archive [VERIFIED]:** `ml-service/data/raw/IMS.zip`
- **Nested Archive [VERIFIED]:** `ml-service/data/raw/IMS/2nd_test.rar`
- **Primary Archive SHA-256 Checksum [VERIFIED]:**  
  `6CB42C263B0281C725ABF99F4B9FCF49915C949F31DBD2333877DC2E06CE9EC2`

> **Immutability Guarantee:** The raw archive and extracted data files have been preserved in their original, unmodified state. No raw file has been renamed, modified, formatted, or deleted.

---

## 3. Extracted Dataset Statistics & File Integrity

### 3.1 Directory Location & File Count
- **Extracted Data Path [VERIFIED]:** `ml-service/data/raw/IMS/2nd_test/2nd_test/`
- **Number of Snapshot Recordings [VERIFIED]:** `984` ASCII text files
- **Total Storage Size [VERIFIED]:** `544,618,480` bytes (~544.6 MB)
- **Minimum File Size [VERIFIED]:** `512,002` bytes
- **Maximum File Size [VERIFIED]:** `557,958` bytes
- **Average File Size [VERIFIED]:** ~`553,474` bytes (553,474.065 bytes)

### 3.2 Recording Data Structure
- **Snapshot Matrix Shape [VERIFIED sample]:** 20,480 rows × 4 columns
- **Sampling Parameters [VERIFIED doc / REPORTED metadata]:** 20.48 kHz sampling frequency, 1-second duration per snapshot (20,480 data points per channel).
- **Data Format [VERIFIED]:** Whitespace-separated numeric floating-point values.
- **Missing Values in Inspected Samples [VERIFIED sample]:** 0 NaN or null values detected in the sampled boundary files (`2004.02.12.10.32.39` and `2004.02.19.06.22.39`). This sampled check does not constitute an exhaustive scan of all 984 recordings; full matrix verification across all files is reserved for Phase 3 EDA.

---

## 4. Timeline and Sampling Consistency

### 4.1 Chronological Bounds
- **First Snapshot Timestamp [VERIFIED]:** `2004-02-12 10:32:39`
- **Last Snapshot Timestamp [VERIFIED]:** `2004-02-19 06:22:39`

### 4.2 Recording Interval Analysis
- **Consecutive Recording Gap [VERIFIED]:** Exactly **600 seconds (10 minutes)** between consecutive snapshot files.
- **Interval Consistency [VERIFIED]:** All 983 consecutive timestamp intervals across the entire 984-file dataset are strictly equal to 600.0 seconds with zero missing timestamps or out-of-order files.

---

## 5. Version Control & Git Safety Verification

### 5.1 Git Exclusion Check
- **Exclusion Rule [VERIFIED]:** `ml-service/data/raw/*` defined in repository `.gitignore`.
- **Git Check-Ignore Verification [VERIFIED via `git check-ignore -v`]:**
  - `ml-service/data/raw/IMS.zip` -> **IGNORED**
  - `ml-service/data/raw/IMS/2nd_test.rar` -> **IGNORED**
  - `ml-service/data/raw/IMS/2nd_test/2nd_test/2004.02.12.10.32.39` -> **IGNORED**

---

## 6. Categorization of Knowledge & Unresolved Issues

To maintain strict scientific transparency, all aspects of this dataset are classified into verified facts, unverified assumptions, and open risks:

### 6.1 Verified Facts `[VERIFIED]`
1. Set 2 consists of exactly 984 valid text files with timestamp-encoded filenames.
2. Every snapshot file has a 600-second (10-minute) interval with strictly monotonic increasing order.
3. Inspected boundary sample snapshots contain 20,480 rows and 4 columns without null values (sampled check only, not an exhaustive scan of all 984 files).
4. Primary archive SHA-256 hash matches `6CB42C263B0281C725ABF99F4B9FCF49915C949F31DBD2333877DC2E06CE9EC2`.
5. All raw dataset files are ignored by Git rules.

### 6.2 Unverified Assumptions & Boundary Caveats `[ASSUMPTION / UNRESOLVED]`
1. **Exhaustive Null Scan [UNRESOLVED]:** Only sample boundary files have been verified for zero missing values. An exhaustive scan across all 984 recordings—comprising 20,152,320 total rows ($984 \text{ files} \times 20,480 \text{ rows/file}$) and 80,609,280 total scalar numeric values ($20,152,320 \text{ rows} \times 4 \text{ columns}$)—has not yet been conducted and is explicitly reserved for Phase 3 EDA.
2. **Sensor-to-Bearing Mapping [UNRESOLVED]:** Secondary sources state columns 1 to 4 correspond to Bearings 1 to 4 respectively (1 channel per bearing). Channel assignment to specific physical bearing locations and failure modes (Bearing 1 outer-race defect) must be verified during Exploratory Data Analysis.
3. **Sensor Units & Calibration [UNRESOLVED]:** The exact amplitude unit (e.g., voltage vs. $g$ acceleration calibration multiplier) is not specified in raw text files and remains an uncalibrated signal representation.
4. **Failure Time Boundary [ASSUMPTION]:** The end of recording (`2004-02-19 06:22:39`) is assumed to be the point of test termination following outer-race failure of Bearing 1, per dataset documentation.
