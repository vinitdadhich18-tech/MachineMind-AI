# ML_PROGRESS.md — MachineMind AI

> Live progress tracker. Update it after each **verified** phase (see "How to update").

## 1. Project name
MachineMind AI

## 2. Current project objective
Build a vibration-based condition-monitoring prototype for rolling-element bearings using **unsupervised anomaly detection** on the NASA IMS Bearing Dataset (provisional subset: **Set 2**). The model outputs an anomaly score and an alert. It must **not** claim to predict failure time or remaining useful life.

Dataset and subset stay **provisional** until Phases 1-2 verify them. Full details: `PROJECT_CONTEXT.md`.
## 3. Current phase
**Phase 10 Completed — Model Packaging.**

## 4. Overall progress checklist
Progress: **10 of 11 phases completed**

- [x] Phase 1: Dataset Documentation and Understanding
- [x] Phase 2: Dataset Acquisition and Organization
- [x] Phase 3: Exploratory Data Analysis
- [x] Phase 4: Signal Preprocessing
- [x] Phase 5: Feature Engineering
- [x] Phase 6: Statistical Baseline
- [x] Phase 7: ML Model Development
- [x] Phase 8: Evaluation
- [x] Phase 9: Model Improvement
- [x] Phase 10: Model Packaging
- [ ] Phase 11: ML Completion and Integration Readiness

Pre-project setup (not a development phase):
- [x] Phase 0: Setup & Context Verification

## 5. Status of all 11 phases
Status values: **Not Started**, **In Progress**, **Blocked**, **Completed**

| # | Phase | Status | Started | Verified (date) | Notes |
|---|---|---|---|---|---|
| 1 | Dataset Documentation and Understanding | Completed | 2026-09-25 | 2026-09-25 | Preliminary report created & approved |
| 2 | Dataset Acquisition and Organization | Completed | 2026-09-25 | 2026-09-25 | Set 2 acquired, SHA-256 verified, read-only script executed, inventory report written |
| 3 | Exploratory Data Analysis | Completed | 2026-09-26 | 2026-09-26 | 984 files quality scanned (100% valid), waveforms & trend plots generated, EDA report written |
| 4 | Signal Preprocessing | Completed | 2026-09-26 | 2026-09-26 | Manifest created (984 files valid), remove_dc_offset & extract_windows implemented, tested & validated |
| 5 | Feature Engineering | Completed | 2026-09-26 | 2026-09-26 | Time & frequency domain utilities implemented in src/feature_extraction.py, unit tested, 984 snapshots extracted to features_set2.csv |
| 6 | Statistical Baseline | Completed | 2026-09-28 | 2026-09-28 | Baseline module implemented in src/baseline.py (11/11 unit tests passed), 984 baseline scores exported, notebook & 3 reports completed |
| 7 | ML Model Development | Completed | 2026-09-29 | 2026-09-29 | Modular pipelines (iForest & PCA) implemented in src/models.py, 4 unit tests passed, 984 model scores exported, notebook & audited report completed |
| 8 | Evaluation | Completed | 2026-09-29 | 2026-09-29 | Modular evaluation module & tests implemented in src/evaluation.py (14/14 unit tests passed, 36/36 total passed), notebook & report written, 6 figures generated |
| 9 | Model Improvement | Completed | 2026-09-30 | 2026-09-30 | Executed EXP-04..EXP-12, reconciled audit, selected 28-feature set & Logical OR system alert policy, user approved final summary |
| 10 | Model Packaging | Completed | 2026-09-30 | 2026-09-30 | Packaged iForest & PCA pipelines (28 time-domain feats), created model_metadata.json, implemented src/export_models.py & src/inference.py, 56/56 tests passed, verified reference consistency (IF max diff 0.00e+00, PCA 1.77e-14), executed notebook 08 & written model_card.md |
| 11 | ML Completion and Integration Readiness | Not Started |  |  |  |

## 6. Completed deliverables
Add a row only after the file exists **and** you have checked it.

| Phase | Deliverable (file path) | Verified how |
|---|---|---|
| 1 | `ml-service/reports/dataset_documentation.md` | User review and approval of preliminary report |
| 2 | `ml-service/reports/data_inventory.md`, `ml-service/src/verify_dataset.py` | Verification script execution output & user approval |
| 3 | `ml-service/notebooks/01_eda.ipynb`, `ml-service/reports/eda_report.md`, `ml-service/reports/figures/eda/` | Full dataset scan execution, trend plot generation & user review |
| 4 | `ml-service/src/preprocessing.py`, `ml-service/data/processed/manifest_set2.csv`, `ml-service/notebooks/02_preprocessing.ipynb`, `ml-service/reports/preprocessing_notes.md` | Synthetic unit testing, manifest validation, empirical pipeline execution & user review |
| 5 | `ml-service/src/feature_extraction.py`, `ml-service/src/test_feature_extraction.py`, `ml-service/data/processed/features_set2.csv`, `ml-service/reports/feature_engineering_notes.md` | Unit test suite execution (7/7 passed), 984-file dataset extraction (984x40, 0 NaNs) & documentation |
| 6 | `ml-service/src/baseline.py`, `ml-service/src/test_baseline.py`, `ml-service/data/processed/baseline_scores_set2.csv`, `ml-service/notebooks/04_statistical_baseline.ipynb`, `ml-service/reports/statistical_baseline_report.md`, `ml-service/reports/baseline_score_analysis.md`, `ml-service/reports/baseline_evaluation.md`, `ml-service/reports/figures/baseline/`, `ml-service/reports/figures/baseline_analysis/` | Unit test suite execution (11/11 passed), notebook execution (top-to-bottom clean run), dataset verification & user review |
| 7 | `ml-service/src/models.py`, `ml-service/src/test_models.py`, `ml-service/data/processed/model_scores_set2.csv`, `ml-service/notebooks/05_model_development.ipynb`, `ml-service/reports/model_development.md`, `ml-service/reports/figures/models/` | Unit test suite execution (4/4 passed), dataset score export verification, technical audit & user approval |
| 8 | `ml-service/reports/evaluation_protocol.md`, `ml-service/src/evaluation.py`, `ml-service/src/test_evaluation.py`, `ml-service/notebooks/06_evaluation.ipynb`, `ml-service/reports/evaluation_report.md`, `ml-service/reports/figures/evaluation/` | Unit test suite execution (14/14 passed), top-to-bottom notebook execution (nbconvert), 6 figure generation & report audit |
| 9 | `ml-service/reports/experiment_log.md`, `ml-service/reports/improvement_summary.md`, `ml-service/notebooks/07_model_improvement.ipynb` | Controlled experiment suite execution (EXP-04..EXP-12), unit test suite execution (36/36 passed), top-to-bottom notebook execution, reconciliation audit & user approval |
| 10 | `ml-service/src/export_models.py`, `ml-service/src/test_export_models.py`, `ml-service/src/inference.py`, `ml-service/src/test_inference.py`, `ml-service/models/model_metadata.json`, `ml-service/models/iforest_pipeline_v1.joblib`, `ml-service/models/pca_pipeline_v1.joblib`, `ml-service/notebooks/08_inference_check.ipynb`, `ml-service/reports/model_card.md` | Full unit test suite execution (56/56 passed), reference numerical consistency check (IF max diff 0.00e+00, PCA 1.77e-14 <= 1e-5), top-to-bottom notebook execution (nbconvert clean), fresh process artifact loading & audit |
| 11 | *(none yet)* | |

## 7. Important decisions
| Date | Decision | Status (Provisional / Confirmed) | Reason |
|---|---|---|---|
| 2026-09-25 | Dataset: NASA IMS Bearing Dataset | Provisional | Only compared dataset with natural run-to-failure degradation; documentation still to verify |
| 2026-09-25 | Initial subset: IMS Set 2 | Provisional | Smallest set (984 files reported), one documented failure; Set 3 excluded because of a reported documentation mismatch |
| 2026-09-25 | Task: unsupervised anomaly detection | Provisional | No per-file labels exist; no RUL or failure-time claims |
| 2026-09-28 | Chronological Baseline Partition | Provisional Reference | Baseline fitting (0–159), Reference evaluation (160–199), Sequential evaluation (200–983). Physical health status remains unverified hypothesis. |
| 2026-09-29 | Per-Channel Unsupervised Anomaly Detection Pipelines | Confirmed | Isolation Forest and PCA Reconstruction Error implemented per channel (4 models each), fit strictly on snapshots 0–159, with validation thresholds derived from snapshots 160–199. |
| 2026-09-29 | Persistence-Filtered Causal Evaluation Framework | Confirmed | Enforced frozen evaluation protocol, persistence filtering (k=3) eliminates 100% of reference false alarms, online confirmation timing separated from retrospective event starts. |
| 2026-09-30 | 28-Feature Time-Domain Feature Set Selection | Confirmed | Selected 28-feature time-domain set per EXP-04 decision rule (0 sustained FAs, 25% lower raw noise sensitivity, 7 features/channel). |
| 2026-09-30 | Logical OR System Alert Aggregation Policy | Confirmed | System alert active if ANY channel meets k=3 sustained threshold condition, preserving single-bearing defect sensitivity. |
| 2026-09-30 | Versioned Model Packaging & AnomalyInferenceEngine Interface | Confirmed | Packaged fitted scalers & models into versioned joblib artifacts (`iforest_pipeline_v1.joblib`, `pca_pipeline_v1.joblib`) with human-readable `model_metadata.json` and a stateful $k=3$ persistence inference engine. |

## 8. Experiment results
Record every experiment, including unsuccessful ones. Results must come from actually running the code.

| ID | Phase | What was tried | Configuration (features, model, seed, periods) | Result | Conclusion / caveat |
|---|---|---|---|---|---|
| EXP-01 | Phase 6 | Non-ML Statistical Baseline Scoring | 36 features, ref_end_idx=160, frozen baseline | `score_rms_z` mean=0.92 (fit), 0.97 (ref eval), 12.30 (seq eval); sustained run >3.0 starts at index 531-585 | Baseline provides stable ~1.0 reference; Pearson r between standard Z and modified Z is >0.999 |
| EXP-02 | Phase 7 | Per-Channel Isolation Forest & PCA Anomaly Detection | 36 features (9 per channel), RobustScaler + iForest (n_est=100) / PCA (k=2), seed=42, fit 0..159, val 160..199 | Both models show sustained score elevation on ch1 starting at snapshot 532; ch2–ch4 exhibit elevated scores in second half due to mechanical shaft coupling | Per-channel models successfully track degradation trajectory; cross-channel shaft coupling documented without causal failure claims |
| EXP-03 | Phase 8 | Persistence-Filtered (k=1..5) & Multi-Channel Evaluation | Baseline T=3.0, iForest/PCA validation P99 thresholds, k=1,3,5 | Persistence k=3 eliminates 100% of reference false alarms (R_FA=0.0%); iForest & PCA confirm ch1 alert at snapshot 534 (3.1 days lead); OR shaft alert confirmed at snapshot 349 (4.4 days lead) | Persistence filtering guarantees false-alarm suppression on baseline period while preserving early detection lead time |
| EXP-04 | Phase 9 | Feature Set Ablation | Full 36 features vs Time-Domain 28 features, Fit 0..159, Cal 160..179, Diag 180..199 | Sustained FA (k=3) tied at 0/80; Raw FA reduced from 12/80 (36-feat) to 9/80 (28-feat) | 28-feature time-domain set selected per decision rule (25% noise reduction, feature parsimony) |
| EXP-05 | Phase 9 | Feature Scaler Method | RobustScaler vs StandardScaler | Identical scores across all channels (max \|delta s\| = 0.0) | Isolation Forest tree split logic is rank-invariant to monotonic scaling; RobustScaler retained |
| EXP-06 | Phase 9 | PCA Component Count | PCA k=1 vs k=2 vs k=3 components | Raw FA reduced to 3/80 (36-feat) / 4/80 (28-feat); Cal MSE reduced to 0.1525 (vs 0.2581 for k=2) | PCA k=3 accepted as improved component rank |
| EXP-07 | Phase 9 | iForest Hyperparameter Grid | n_estimators in {50,100,200} x max_samples in {0.5,1.0,"auto"} | All 9 grid configurations achieved 0 sustained false alarms (k=3) | Retain control n_estimators=100, max_samples="auto" to avoid parameter churn |
| EXP-08 | Phase 9 | Threshold Strategy | Fit mu+3s/4s vs Cal P99/P99.5/mu+3s | Cal P99 and P99.5 yielded identical cutoffs; Fit cutoffs overly conservative | Retain Cal P99 as non-parametric baseline threshold cutoff |
| EXP-09 | Phase 9 | Persistence Filter | k=1 vs k=3 vs k=5 consecutive exceedances | k=3 and k=5 eliminated 100% of diagnostic FAs; k=3 requires 20 min delay vs 40 min for k=5 | Retain k=3 filter for optimal noise suppression vs latency trade-off |
| EXP-10 | Phase 9 | Fit Window Sensitivity | N_fit=100 vs N_fit=160 | N_fit=160 provided lower raw noise sensitivity (12 vs 21 on 36-feat; 9 vs 12 on 28-feat) | N_fit=160 retained. Calibration windows differed (100..119 vs 160..179), introducing calibration shift |
| EXP-11 | Phase 9 | Modeling Architecture | Per-Channel (4 models) vs Pooled (1 joint model) | Both achieved 0 sustained FAs. Per-channel raw FA 12/80 (36-feat) / 9/80 (28-feat); Pooled raw FA 5/20 / 3/20 | Retain per-channel modeling to preserve spatial bearing defect localization |
| EXP-12 | Phase 9 | Master Synthesis & Alert Policy | Final pipeline selection & Logical OR system alert policy | Synthesized EXP-04..EXP-11, selected 28-feature set & Logical OR system alert rule | Final Phase 9 pipeline specified & user approved |

## 9. Known issues and blockers
| # | Issue | Type (Issue / Blocker) | Status |
|---|---|---|---|
| 1 | IMS licence metadata is inconsistent; usage for public sharing unresolved | Issue | Open |
| 2 | Set 2 raw file details verified (984 files, 600s intervals, 20480x4 shape) | Issue | Verified in Phase 2 |
| 3 | Sensor units and calibration unknown | Issue | Open |
| 4 | Python 3.14.2 compatibility of any additional package unverified | Issue | Open |
| 5 | Only one documented failure in Set 2; evaluation will be a single-case study | Limitation | Accepted |

## 10. Next action
**Prepare for Phase 11 — ML Completion and Integration Readiness when ready.**


---

## How to update this file
After **each** phase:
1. Read the agent's summary and check the files it claims to have created. Re-run the notebook or check yourself; do not rely on the summary alone.
2. Confirm the phase's completion checklist is honestly satisfied.
3. Set the phase status: **In Progress** while working, **Blocked** if you cannot continue (write why in Known issues), **Completed** only after you verified it.
4. Fill in the Verified date, add rows to Completed deliverables, and record experiments (Section 8) and decisions (Section 7).
5. Update Section 3 (Current phase), the checklist and the "x of 11" count, then write the Next action.
6. If a decision changed, also update `PROJECT_CONTEXT.md`.
7. **Git Commit Reminder:** Run `git status`, verify `.gitignore` safety (no raw data, venv, secrets, or data binaries staged), explain changed files, suggest a meaningful commit message, and present exact `git add` and `git commit` commands for manual execution. Never execute git commit automatically.
Never mark a phase Completed because the environment works or because code was generated; only because it was verified.
