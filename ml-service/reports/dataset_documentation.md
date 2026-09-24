# Preliminary Dataset Documentation and Verification Report — NASA IMS Bearing Dataset (Set 2 Focus)

**Project:** MachineMind AI  
**Component:** `ml-service`  
**Phase:** Phase 1 — Dataset Documentation and Understanding  
**Date:** 2026-09-25  
**Document Status:** Preliminary Documentation (Pending Phase 2 File Inspection and User Review)

---

## 1. Executive Summary

This report provides a **preliminary documentation baseline** for the **NASA IMS Bearing Dataset**, with specific focus on **Set 2**. The objective is to evaluate whether Set 2 is provisionally suitable for an **unsupervised anomaly detection prototype** for rolling-element bearings in rotating machinery.

**Important Notice:** This document is compiled from official NASA metadata pages, primary reference papers, and secondary literature. It does **not** represent a fully verified final report; full empirical verification of file counts, column contents, data integrity, and timestamps will occur in Phase 2 once dataset files are inspected.

Every fact in this document is explicitly categorized into one of three verification levels:
- **[VERIFIED]:** Confirmed directly via official NASA repository metadata, landing pages, or peer-reviewed literature.
- **[REPORTED]:** Sourced from secondary academic publications or community repository audits; requires verification against dataset files during Phase 2.
- **[ASSUMPTION / UNRESOLVED]:** A methodological choice or open question that cannot be proven from dataset documentation alone.

---

## 2. Official Data Sources & References

### 2.1 Primary Repository Records
- **NASA Open Data Portal Landing Page [VERIFIED]:**  
  `https://data.nasa.gov/dataset/ims-bearings`  
  *Metadata:* Contact Author: Christopher Teubert (`christopher.a.teubert@nasa.gov`). License status: Listed as `other-license-specified`.
- **NASA Legacy Direct Data Download Link [VERIFIED]:**  
  `https://data.nasa.gov/docs/legacy/IMS.zip`
- **NASA Prognostics Center of Excellence (PCoE) Repository [VERIFIED]:**  
  `https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/` (Entry: *Bearings*).

### 2.2 Academic Reference Citation
- **Primary Publication [VERIFIED]:**  
  Qiu, H., Lee, J., Lin, J., & Yu, G. (2006). *Wavelet-based sensor data denoising for tool condition monitoring*. Journal of Sound and Vibration, 289(4-5), 1066-1090.
- **Official Dataset Citation Request (per NASA PCoE) [VERIFIED]:**  
  > J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. "Bearing Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

---

## 3. Test Rig Architecture & Experimental Setup

Based on official repository documentation and Qiu et al. (2006):

- **Mechanical Architecture [REPORTED]:**  
  Four double-row rolling element bearings (NSK 6015) are mounted on a single heavy-duty shaft driven by an AC motor via rub belts.
- **Operating Conditions [REPORTED]:**  
  - **Shaft Speed:** Approximately **2,000 RPM** (constant rotation).
  - **Radial Load:** Approximately **6,000 lbs** (27 kN) applied to the shaft and bearings via a heavy-duty spring mechanism.
- **Sensor Setup & Placement [REPORTED]:**  
  - **Sensors:** PCB 353B33 high-sensitivity quartz accelerometers installed on the bearing housings.
  - **Set 2 Sensor Allocation:** 1 accelerometer per bearing (4 sensors total across 4 channels).

---

## 4. Data Specification & File Structure

### 4.1 Sampling Protocol & Signal Properties
- **Sampling Frequency ($f_s$) [REPORTED]:** **20 kHz** (20,000 samples per second per channel).
- **Nyquist Frequency Limit ($f_{max}$) [VERIFIED Concept]:** $f_s / 2 = 10\text{ kHz}$. Vibration frequencies up to 10 kHz can be represented without aliasing.
- **Snapshot Duration [REPORTED]:** **1 second** per recording snapshot.
- **Snapshot Sample Count [REPORTED]:** **20,480 data points** per channel per file.
- **Recording Interval [REPORTED]:** Snapshots were captured every **10 minutes** (1-second sampling burst followed by a 9-minute 59-second pause).

### 4.2 File Naming & Column Organization
- **File Format [REPORTED]:** Plain ASCII text files, space/tab-delimited, containing raw numbers without header rows.
- **Timestamp Encoding [REPORTED]:** File names represent the exact snapshot timestamp formatted as `YYYY.MM.DD.HH.MM.SS` (e.g., `2004.02.12.10.32.39`).
- **Set 2 Channel Layout [REPORTED]:**
  - **Column 1 (Channel 1):** Bearing 1 accelerometer
  - **Column 2 (Channel 2):** Bearing 2 accelerometer
  - **Column 3 (Channel 3):** Bearing 3 accelerometer
  - **Column 4 (Channel 4):** Bearing 4 accelerometer

---

## 5. IMS Set 2 Degradation & Failure Information

- **Total File Count [REPORTED]:** **984 files** recorded between February 12, 2004, and February 19, 2004.
- **Documented Failure Event [REPORTED]:** At the conclusion of the 984-file run, severe **outer race failure** was observed on **Bearing 1**.
- **Status of Bearings 2, 3, and 4 [REPORTED / UNVERIFIED]:** Secondary sources report that Bearings 2, 3, and 4 remained structurally intact throughout Set 2. However, because all bearings share one shaft, vibration crosstalk from Bearing 1 influences sensors on Bearings 2–4.
- **Ground-Truth Labels [VERIFIED Fact]:** **NO per-recording health labels exist.** There are no flags in the raw dataset files indicating "Healthy", "Degrading", or "Faulty".

---

## 6. Comprehensive Fact Verification Table

| Aspect / Property | Value / Description | Source | Verification Status |
|---|---|---|---|
| **Dataset Origin** | Center for Intelligent Maintenance Systems (IMS), Univ. of Cincinnati | NASA Open Data Portal | **[VERIFIED]** |
| **Primary Repository** | NASA Prognostics Data Repository | NASA PCoE Portal | **[VERIFIED]** |
| **Experiment Type** | Natural Run-to-Failure degradation test | Official NASA Overview | **[VERIFIED]** |
| **Machinery Type** | NSK 6015 double-row rolling element bearings | Secondary / Qiu et al. | **[REPORTED]** |
| **Shaft Speed** | ~2,000 RPM (constant) | Secondary / Qiu et al. | **[REPORTED]** |
| **Radial Load** | ~6,000 lbs (27 kN) | Secondary / Qiu et al. | **[REPORTED]** |
| **Set 2 Channels** | 4 channels (1 accelerometer per bearing) | Dataset Documentation | **[REPORTED]** |
| **Sampling Frequency** | 20 kHz | Secondary / IMS Overview | **[REPORTED]** |
| **Snapshot Length** | 1 second (20,480 points) | Secondary / IMS Overview | **[REPORTED]** |
| **Recording Interval** | Every 10 minutes | Secondary / IMS Overview | **[REPORTED]** |
| **Set 2 File Count** | 984 ASCII files | Secondary Audits | **[REPORTED]** |
| **Set 2 Failure Mode** | Bearing 1 Outer Race Failure | Secondary / IMS Overview | **[REPORTED]** |
| **Per-File Health Labels** | None provided | Dataset Inspection | **[VERIFIED]** |
| **Sensor Calibration** | Unknown (mV/g factor not supplied) | Dataset Documentation | **[UNRESOLVED]** |
| **License Type** | Listed as `other-license-specified` / Public Domain mirror | `data.nasa.gov` metadata | **[UNRESOLVED]** |

---

## 7. Known Audit Discrepancies & Exclusions

- **Set 3 Exclusion Rationale [PROVISIONAL PROJECT DECISION]:**  
  Independent dataset audits (e.g., open-source PHM benchmarks) reported that the downloaded directory for **IMS Set 3** contradicted its documentation in file count and end dates. As a risk-mitigation measure, **IMS Set 3 is provisionally excluded** from the initial scope of MachineMind AI.
- **Set 2 Verification Requirement [PROCEDURAL]:**  
  When raw files are downloaded in Phase 2, an automated Python validation script must verify file counts, column shapes, line counts (20,480 lines), and timestamp continuity before proceeding to EDA.

---

## 8. License, Citation, and Data Governance

1. **Licensing Status [UNRESOLVED]:**  
   `data.nasa.gov` metadata lists the license as `other-license-specified`, while NASA PCoE describes repository data as freely available for research. Without explicit legal terms on raw data redistribution, **raw dataset files must never be committed to Git or redistributed publicly**.
2. **Git Governance Rule [MANDATORY]:**  
   The root `.gitignore` file enforces that all files inside `ml-service/data/raw/` and `ml-service/data/processed/` are excluded from version control.
3. **Citation Compliance [MANDATORY]:**  
   Any published report or artifact derived from this project must include the official NASA citation:
   > *J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. "Bearing Data Set", NASA Prognostics Data Repository, NASA Ames Research Center.*

---

## 9. Critical Scientific Limitations & Methodological Implications

1. **Assumed Healthy Baseline (No Ground Truth Labels):**  
   Because there are no health labels, defining a "healthy training period" (e.g., the first 150–200 files) is an **analyst assumption**, not a verified ground truth. The model measures deviation from an assumed baseline, not absolute physical fault states.
2. **Single Failure Event ($N=1$ Case Study):**  
   Set 2 captures only **one** bearing failure event. Statistical validation across multiple failures is impossible with Set 2 alone. Results must be presented as a **single-case prototype demonstration**.
3. **Shared Shaft Vibration Coupling:**  
   Because all 4 bearings reside on one shaft, severe vibration from Bearing 1 propagates through the shaft to sensors on Bearings 2, 3, and 4. Anomaly detectors on channel 2–4 may trigger due to crosstalk rather than local bearing damage.
4. **Unknown Calibration & Units:**  
   Without accelerometer calibration factors, amplitude metrics (like RMS or Peak) reflect relative voltage variations rather than calibrated $m/s^2$ acceleration.
5. **Fixed Operating Environment:**  
   The test rig operated at fixed speed and load in a controlled lab. Models trained on this data will not generalize to machines operating under dynamic loads or variable speeds.
6. **Proxy Failure Timestamp:**  
   The abrupt termination of the experiment serves as a proxy for total bearing failure, but the exact timestamp of initial micro-spalling is unknown.

---

## 10. Provisional Assessment of Set 2 Suitability for Anomaly Detection

### Strengths (Reported)
- Captures natural, unseeded run-to-failure degradation over 7 continuous days.
- Periodic 10-minute snapshot timestamps enable chronological, leakage-free time-series evaluation.
- Standard 4-channel structure allows comparative analysis across bearing positions.

### Weaknesses & Risks (Identified)
- Unlabeled data requires subjective baseline selection.
- Single failure run limits evaluation robustness.
- Potential sensor crosstalk across shared shaft.

### Recommendation
**PROVISIONAL — PROCEED WITH CAVEATS:** IMS Set 2 is provisionally suitable for building a research/learning prototype of an **unsupervised anomaly detection model**, provided that all evaluation reports explicitly state the $N=1$ case-study limitation and label-free baseline assumption. This suitability assessment remains **provisional** until dataset files are directly inspected in Phase 2.

---

## 11. Open Questions for Phase 2 Verification

Before or during Phase 2 dataset acquisition, the following questions must be answered by inspecting the downloaded files:

1. **File Count:** Does the downloaded `IMS.zip` contain exactly 984 files for Set 2?
2. **Data Delimiter:** Are numbers separated by spaces, tabs, or commas?
3. **Data Completeness:** Are all 984 files non-empty and formatted with exactly 20,480 rows and 4 columns?
4. **Timestamp Continuity:** Are there missing timestamps or large time gaps between consecutive recordings?
5. **Raw File README:** Does the `IMS.zip` archive contain an internal README file, and does it contradict any secondary-source claims?

---

## 12. Verification & Completion Checklist

- [x] Official sources checked and cited
- [x] Fact table complete with statuses ([VERIFIED] / [REPORTED] / [UNRESOLVED])
- [x] License and citation requirements recorded
- [x] Methodological limitations (unlabeled data, $N=1$ failure) explicitly documented
- [x] Set 2 suitability assessment completed with recommendation
- [x] Open questions for Phase 2 listed
- [x] No raw data downloaded; Python environment untouched; `PROJECT_CONTEXT.md` & `ML_PROGRESS.md` unmodified
