# MachineMind AI — Phase 5 → Phase 10 Continuation Prompt Pack

**Project:** MachineMind AI — Industrial Predictive Maintenance & Vibration Monitoring Platform
**Branch:** `feature/industrial-stack-migration`
**Audience:** Antigravity (coding agent)
**Scope of this file:** Phases 5–10 only. Phases 0–4 are already done and are summarized in Part A as baseline context. Do not ask for the old prompt files.

---

## HOW TO USE THIS FILE (read first)

1. This file contains six phases (5, 6, 7, 8, 9, 10). **You execute exactly ONE phase per session**, the one the user names (e.g. "Execute Phase 5"). If the user does not name a phase, start at Phase 5.
2. Read Part A (baseline) and Part B (global rules) in full before every phase. They apply to all phases.
3. After finishing a phase you **commit, push, verify, report, and STOP**. Never start the next phase on your own.
4. **Repository evidence beats this document.** If the code, Git history, or docs disagree with anything written here, trust the repository, record the discrepancy in your report, and proceed on the evidence.
5. You are implementing the current phase only. Do not redesign unrelated systems, do not jump ahead.

---

# PART A — COMPLETED WORK / BASELINE

## A.1 Project purpose

Build an industrial-style predictive-maintenance pipeline using **historical NASA IMS bearing vibration data replayed as telemetry through MQTT**. It demonstrates: industrial MQTT telemetry, streaming ingestion, vibration signal processing, feature engineering, supervised predictive maintenance, RUL/degradation prediction, prediction persistence, machine health state, production-style architecture, reproducible testing, and a professional Git workflow.

## A.2 Honesty requirement (non-negotiable)

NASA IMS is **historical run-to-failure research data**. No physical live sensors are connected to this project. Always describe the source as:

> "replayed NASA IMS historical telemetry" or "historical NASA IMS vibration data replayed through MQTT to simulate a live telemetry source."

Never claim the system receives data from real physical industrial sensors — in code comments, docs, logs, README, reports, or viva material.

## A.3 Teacher's actual requirement

| Component | Status |
|---|---|
| MQTT (HiveMQ Cloud) | **MANDATORY** — core of the system |
| InfluxDB | **OPTIONAL** — useful time-series persistence, never a dependency of core inference |
| Grafana | **NOT REQUIRED** — do not add it, do not build dashboards, do not spend time on it |

The core path must work with InfluxDB switched off and with Grafana nonexistent.

## A.4 Dataset facts (NASA IMS Set 2, established in Phase 0)

- 984 snapshot files, chronological run-to-failure sequence
- 20,480 rows per snapshot, 4 vibration channels
- Sampling rate 20,480 Hz → ~1 s of vibration per snapshot
- ~10 minutes between snapshots (approximate; verify from filenames/timestamps before relying on it)
- Raw dataset is preserved and must never be modified

Original (legacy) architecture: NASA IMS → signal processing/features → RobustScaler → Isolation Forest + PCA → Flask → MongoDB → Streamlit.

## A.5 Completed phases (do NOT redo)

**Phase 0 — Audit & migration design.** Repository audited, industrial-stack migration designed. Baseline commit before migration: `e1c4f18`.

**Phase 1 — HiveMQ Cloud MQTT telemetry publisher.** HiveMQ Cloud broker, TLS, port 8883, credentials via environment variables only. Topics: `machinemind/v1/telemetry/{machine_id}` and `machinemind/v1/status/{machine_id}`. Machine ID `ims_set2_rig`. QoS 1, telemetry retain=false, status/LWT implemented, source tag `replayed_nasa_ims`. Verified: TLS, auth, publish/subscribe, QoS 1, sequence numbers, original timestamps, 4 channels × 20,480 samples, label-leakage protection, no credential leakage. Commits: `37d5d3a`, `61da8cc`, `69b0082`.

**Phase 2 — MQTT ingestion consumer.** `ml-service/src/ingestion_consumer.py`. TLS, env-var auth, subscribes to telemetry/status, QoS 1, reconnect handling, duplicate/out-of-order/sequence-gap tracking, telemetry validation, original timestamps preserved, ingest timestamps created, standardized `TelemetryRecord` (machine_id, snapshot_sequence, original_timestamp, ingest_timestamp, sampling_rate_hz, sample_count, ch1–ch4, ingestion metadata). Verified live against HiveMQ Cloud. Commit: `0d1581a`.

**Phase 3 — InfluxDB integration (optional layer).** InfluxDB v2. Files: `ml-service/src/influx_writer.py`, `docker-compose.yml`, `ml-service/src/test_phase3_influx.py`, `ml-service/reports/influxdb_schema_documentation.md`. Env vars: `INFLUXDB_URL`, `INFLUXDB_ORG`, `INFLUXDB_BUCKET`, `INFLUXDB_TOKEN`. Bucket `machinemind_telemetry`; measurements `vibration_features`, `pipeline_health`. Point timestamps use the replay/ingest timeline; original NASA timestamps are kept as provenance. Raw 20,480×4 arrays are NOT stored. Verified live end to end. Commit: `13d02fb` (`feat(storage): add InfluxDB telemetry persistence`).

**Phase 4 — Canonical signal processing & feature pipeline.** Delivered (per user) but **its commit hash is not known to this document.** Intended result: `TelemetryRecord → canonical signal processing → canonical feature engineering → deterministic ML-ready feature vector`, all 4 channels, deterministic ordering, finite output (no NaN/Inf), offline/online parity, no future-data leakage, training preprocessing separated from inference preprocessing, verified live HiveMQ → TelemetryRecord → feature pipeline. No XGBoost, no invented labels, legacy Flask/MongoDB/Streamlit untouched, no Grafana.
**You must locate Phase 4 yourself** via `git log`, file inspection, and tests. Do not assume a hash, module name, function signature, or feature list from this document. If Phase 4 is missing or incomplete, report it and propose a minimal, clearly-scoped fix before continuing.

## A.6 Target architecture (Phase 5 onward)

```
NASA IMS historical dataset
        ↓
Replay Simulator
        ↓
HiveMQ Cloud MQTT/TLS
        ↓
MQTT Ingestion Consumer
        ↓
TelemetryRecord
        ↓
Canonical Signal Processing
        ↓
Canonical Feature Vector
        ↓
XGBoost predictive model
        ↓
RUL / degradation / health prediction
        ↓
Prediction state / persistence
        ↓
Optional InfluxDB persistence
```

Flask, MongoDB, Streamlit, Isolation Forest, and PCA are **legacy**. They stay in the repo until Phase 9 and only if Phase 9 proves them obsolete.

---

# PART B — GLOBAL RULES (apply to every phase)

## B.1 Start-of-phase protocol (mandatory, do it before writing any code)

1. `git status`, `git branch --show-current`, `git log --oneline -n 20`. Confirm you are on `feature/industrial-stack-migration` with a clean tree. If the tree is dirty, stop and report; do not commit unrelated changes.
2. Inspect the actual implementation of everything the current phase depends on (modules, tests, docs, config, artifacts).
3. Confirm the previous phase's status from evidence (commit, files, passing tests). Run the existing test suite once to establish a baseline pass/fail count.
4. List discrepancies between this prompt/documentation and the real code. Resolve in favor of the repository and mention them in your report.
5. Write a short plan (files to create/modify, tests to add) before coding.

## B.2 Behavior during a phase

- Implement only the current phase. Do not jump ahead. Do not redesign unrelated systems.
- Reuse existing modules and conventions (naming, folder layout, config style, logging, test framework). Do not create parallel duplicate implementations.
- Do not remove working components unless the phase explicitly allows it.
- **Do not invent data, metrics, labels, thresholds, or results.** Every number in docs/reports must come from a command you actually ran. Record the command and date/commit next to it.
- Do not claim unverified functionality. "Imports successfully" is not "works".
- If something cannot be verified (e.g. HiveMQ credentials unavailable in your environment), say so explicitly and mark that part **UNVERIFIED** rather than substituting a mock and calling it done.

## B.3 Testing philosophy

Every phase must:

- add appropriate new tests, preserve existing ones
- run phase-specific tests, then the **full** test suite
- include **real integration testing** where practical, particularly against HiveMQ Cloud for anything touching MQTT, using real NASA IMS snapshots
- use mocks only for unit isolation, never as the sole evidence for a core claim
- keep tests deterministic (fixed seeds, no reliance on wall-clock timing except where latency is the subject)
- clean up after themselves (no stray topics, no leftover credentials, no large temp files in the repo)

## B.4 Secrets

Never commit or print: `.env`, MQTT usernames/passwords, HiveMQ credentials, InfluxDB tokens, API keys, private certificates. Use `.env.example` with placeholder values only. Never log credentials. Never paste real credentials into docs. Before every commit run `git diff --cached` and a grep for credential-like strings and confirm nothing sensitive is staged.

## B.5 ML integrity rules (apply to Phases 5–8)

- XGBoost is a **supervised** model. It is never to be presented as an anomaly detector.
- No random splits of the chronological snapshots. Chronological only.
- No future information in preprocessing, feature selection, hyperparameter tuning, thresholding, or model selection.
- The final test set is touched **once**, after all choices are frozen.
- Training preprocessing is *fit* on training data only. Inference preprocessing only *applies* the persisted fit.
- Keep these concepts strictly separate in code, naming, and docs: **predicted RUL**, **anomaly score** (Isolation Forest), **health index**, **machine state**.

## B.6 Git workflow (end of every phase)

1. `git status` and `git diff` — review every change.
2. Verify no secrets and no unintended large files are staged (`git diff --cached --stat`).
3. Run phase tests, then the full suite. All must pass (or failures must be pre-existing, documented, and justified).
4. Commit with a professional Conventional-Commit message (e.g. `feat(ml): ...`, `feat(inference): ...`, `docs: ...`). One phase = one commit unless a genuinely separable second commit is cleaner (explain why).
5. `git push` the branch. Verify the push succeeded (`git status`, `git log origin/feature/industrial-stack-migration -n 1`).
6. Confirm the working tree is clean.
7. **STOP.** Output the phase report (template in Part D).

## B.7 Repository policy for artifacts

Check `.gitignore` and existing conventions before committing any binary/model/data file. Do not commit large or regenerable artifacts without a clear reason. Prefer committing: code, configs, schemas, small metadata JSON, docs. If a small model artifact must be committed for reproducibility, state its size and justify it; otherwise commit the training script plus a documented command that regenerates it and keep the artifact out of Git.

---

# PHASE 5 — SUPERVISED DATASET + XGBOOST TRAINING

## 5.1 Objective

Introduce a legitimate supervised predictive model: build an ML dataset from the canonical feature pipeline, define a defensible RUL/degradation target, split chronologically with leakage controls, train and evaluate an XGBoost regressor, and persist a versioned, reproducible model artifact set.

## 5.2 Pre-flight specific to Phase 5

- Locate the Phase 4 canonical feature pipeline: its entry point, the feature schema/ordering, the training-vs-inference preprocessing separation, and existing tests.
- Locate the NASA IMS Set 2 raw data and how snapshot sequence/timestamps map to files. Confirm the count (expected 984) and the chronological order from filenames.
- Check whether other IMS sets are present in the repo or data directory. **Do not download or assume other sets exist.** If they are absent, work with Set 2 only.
- Inspect any existing legacy code that touches labels, thresholds, or the Isolation Forest/PCA training so you can reuse what is genuinely reusable and avoid contradicting existing docs.
- Check whether `xgboost`, `scikit-learn`, `joblib`, etc. are already dependencies; add only what is needed and pin consistently with repo conventions.

## 5.3 Task A — Feature matrix construction

- Build the offline feature matrix by running **all 984 raw snapshots** through the **same canonical feature function** used online (via a `TelemetryRecord` or the pipeline's offline-equivalent entry point). Do not write a second feature implementation.
- Each row is keyed by `snapshot_sequence` and original timestamp.
- Verify: 984 rows, deterministic column order, no NaN/Inf, columns match the Phase 4 feature schema exactly.
- Any rolling/temporal/cumulative feature must be **causal** (uses only snapshot ≤ t). Prove it with a test: the feature vector for snapshot t must be identical whether or not snapshots > t exist.
- Persist the matrix in a reproducible way (see 5.9) with a clear regeneration command.

## 5.4 Task B — Target construction (RUL), documented and justified

The dataset has **no explicit per-snapshot RUL labels**. The target is *derived* from the known chronological run-to-failure structure, and docs must say so plainly.

Candidate formulations (evaluate, do not blindly adopt):

- Linear RUL (in snapshots): `RUL(t) = (N − 1) − t`, where `N` = total snapshots (984 → `983 − t`)
- Capped / piecewise-linear RUL: `RUL(t) = min(RUL_linear(t), cap)` (e.g. cap = 500 as a *candidate*; the cap value must be justified, not assumed)
- A normalized degradation target (e.g. fractional life consumed) if the analysis shows it is more defensible

Required analysis before choosing:

1. Confirm what "end of run" means for Set 2 from repository docs and dataset documentation. State what is established (run-to-failure) and what is assumed (that the last snapshot corresponds to functional failure). Do not overclaim.
2. Examine whether early-life snapshots are essentially indistinguishable (flat healthy behavior). Linear RUL implies the model should distinguish snapshot 5 from snapshot 300 even if the bearing looks identical in both, which is physically questionable. This is the usual reason capped RUL is used in PHM. Show evidence from the actual features (e.g. plot/summary of a few key features over time), not from memory.
3. Report the unit: RUL in **snapshots**; if you also express it in minutes/hours, label it "approximate (≈10 min/snapshot)" and verify the spacing from timestamps.
4. **Target-range extrapolation analysis (critical).** Tree-based models like XGBoost cannot predict target values outside the range seen in training. With a chronological split where training ends well before failure, the test region contains *lower* RUL values than anything the model trained on. Quantify this from the real split: training target range vs validation range vs test range. State clearly how this limits what the model can claim, and document the mitigation chosen (for example: a capped target, a degradation-stage/normalized target, a walk-forward/expanding-window evaluation, or additional run-to-failure sequences **only if they truly exist in the repo**). Whichever choice is made, it must be reported honestly, including the consequence that a model trained only on the early part of a single run cannot be expected to predict near-failure RUL accurately.
5. Choose the final target using **training/validation evidence only**, never the test set. Write the decision, alternatives considered, and rationale into the target-definition document (5.10).

## 5.5 Task C — Chronological split and leakage audit

- Candidate split inherited from earlier design: **train 0–600, embargo/validation 601–750, test 751–983**. Treat it as a *starting hypothesis*. Verify it against the extrapolation analysis above and the data; adjust only with documented justification and without looking at test performance.
- Splits are index-contiguous and strictly ordered; no overlap; no shuffling. Define an embargo gap if rolling features span windows so that no window straddles a split boundary.
- Write an **automated leakage audit** (script + test) that asserts:
  - split boundaries are contiguous, disjoint, and chronologically ordered
  - scaler/preprocessing statistics are computed from training rows only (fit-only-on-train check, e.g. refit on train and compare, or assert fit was called with train indices)
  - causal-feature check from 5.3
  - no feature column is a direct function of the target or of `snapshot_sequence`/timestamp/file index (explicitly test that sequence index, filename, and original timestamp are **not** model inputs)
  - feature selection (if any) uses training rows only
  - hyperparameter tuning uses training+validation only, never test
  - test rows are not read until the final evaluation step
- The audit output is saved as a report.

## 5.6 Task D — Training-only preprocessing

- Fit the scaler/preprocessor (follow Phase 4's separation; reuse `RobustScaler` or whatever Phase 4 established unless there's a documented reason) on **training rows only**.
- Persist the fitted preprocessor. Validation/test/live data only ever pass through `transform`.
- Note: tree models are largely scale-invariant. If you keep a scaler for parity with the inference path, say so honestly; if you decide it adds nothing, document that and make training/inference consistent either way.

## 5.7 Task E — XGBoost regression

- `xgboost` regressor with an explicit configuration object (stored in a config file, not scattered literals): objective, `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, regularization, `random_state`, early-stopping rounds, etc.
- Fixed random seed(s) for reproducibility; document single-thread/deterministic settings if needed to get repeatable results.
- Hyperparameter search (if any) must be **chronology-aware** (e.g. time-series/expanding-window CV inside the training+validation region). No random K-fold. Keep the search modest; do not claim a big search you did not run.
- Early stopping uses the validation split, never test.
- Include simple, honest baselines for context (e.g. predict the training mean; linear regression on the same features) so XGBoost's result is interpretable. Baselines must be measured, not assumed.

## 5.8 Task F — Evaluation

- Metrics: **RMSE, MAE, R²** on train, validation, and (once, at the end) test.
- Optionally add a PHM-style asymmetric score (e.g. the PHM08-style scoring function that penalizes late predictions more than early). If you add it, document the exact formula and units. Do not mention it if you do not implement it.
- Report errors in the target's units, and additionally in snapshots/approximate minutes where meaningful.
- Add a view of error vs. time/true-RUL (table or saved figure) so near-failure behavior is visible, not hidden by a global average.
- Report feature importance but **do not present it as causal/physical proof**.
- **Report only measured numbers.** If test performance is poor, report it as poor and explain the likely cause (for example, the extrapolation problem) rather than adjusting the split or the metric until it looks good. Never tune on test.
- The test set is evaluated exactly once after configuration is frozen. Record the exact command, commit, and timestamp of that evaluation.

## 5.9 Task G — Model artifacts and reproducibility

Persist a **versioned artifact bundle** (follow repo conventions for location, e.g. under `ml-service/models/` or `ml-service/artifacts/` — check first):

- XGBoost model file (native XGBoost format preferred; avoid pickle-only if a native format is available)
- fitted preprocessor/scaler
- feature schema (ordered feature names + dtypes)
- target definition (formula, cap, units)
- split definition (index boundaries, embargo)
- training configuration and seed
- library versions (python, xgboost, sklearn, numpy, pandas)
- data fingerprint (e.g. hash/count of snapshot files used, hash of the feature matrix)
- metrics from the actual run
- model version string and creation timestamp
- git commit of the code that produced it

Provide a single documented command to regenerate everything from raw data (e.g. `python -m ... train`). Add a test that re-running with the same seed reproduces the same predictions (within a stated tolerance) on a small fixture.

## 5.10 Task H — Documentation

Create under `ml-service/reports/` (or the repo's established docs location):

- `rul_target_definition.md` — target choice, alternatives, rationale, extrapolation analysis, limitations
- `xgboost_training_report.md` — dataset, split, leakage audit results, config, metrics (measured only), baselines, limitations
- update the README/architecture docs section for the ML stage, with the honest "replayed NASA IMS historical telemetry" wording

## 5.11 Tests for Phase 5

New tests (names to follow repo convention, e.g. `test_phase5_*.py`): feature-matrix shape/order/finiteness; causal-feature test; target construction (known values, off-by-one at both ends, cap behavior); split contiguity/ordering; leakage audit; fit-on-train-only; artifact save/load round trip; schema mismatch is rejected; deterministic retraining; metric functions against hand-computed values. Then the full existing suite.

## 5.12 Out of scope for Phase 5

No live MQTT inference (Phase 6), no health-state thresholds (Phase 7), no deletion/modification of legacy Isolation Forest/Flask/MongoDB/Streamlit, no Grafana, no InfluxDB changes.

## 5.13 Acceptance criteria

- [ ] Phase 4 pipeline located and reused; no duplicate feature code
- [ ] 984-row feature matrix, schema-exact, finite, causal (tested)
- [ ] Target defined, justified, documented; extrapolation analysis present with real numbers
- [ ] Chronological split verified; leakage audit passes and is automated
- [ ] Preprocessor fit on train only
- [ ] XGBoost trained; baselines measured; metrics are measured and reproducible
- [ ] Test set evaluated once, after freeze
- [ ] Versioned artifact bundle + regeneration command + reproducibility test
- [ ] Docs written; no invented numbers
- [ ] Full test suite green; no secrets staged; committed; pushed; tree clean
- [ ] **STOP**

---

# PHASE 6 — LIVE XGBOOST INFERENCE

## 6.1 Objective

Run the trained model on live (replayed) telemetry:

```
HiveMQ → TelemetryRecord → canonical features → trained preprocessing → trained XGBoost → predicted RUL
```

## 6.2 Pre-flight specific to Phase 6

- Locate the Phase 5 artifact bundle, its loader (if any), and its schema/version metadata.
- Re-read the Phase 2 consumer's extension points (callbacks/handlers/queues) and the Phase 4 online feature entry point. Extend them; do not fork them.
- Check how Phase 3 attaches to the consumer so the inference stage composes cleanly with the optional InfluxDB writer.

## 6.3 Requirements

1. **Model loader.** Loads the artifact bundle once at startup; validates artifact integrity (files present, versions readable) and the feature schema; refuses to start (clear error) if the schema or required artifacts are missing or mismatched. Model path/version selected via configuration/env var.
2. **Inference stage.** Takes a `TelemetryRecord`, runs the *same* canonical feature code and the *persisted* preprocessor, then XGBoost. **No retraining. No scaler fitting. No schema drift.** A test must prove the online features for a given snapshot equal the offline training-matrix features for the same snapshot (parity within a tight, stated tolerance).
3. **Prediction record.** A standardized `PredictionRecord` with at least: machine_id, snapshot_sequence, original_timestamp, ingest_timestamp, prediction_timestamp, predicted_rul (with units), model_version, feature_schema_version, target_definition id, processing/inference latency fields, validity flags. It must **not** carry ground-truth labels from the replay (label-leakage protection from Phase 1 stays intact).
4. **Prediction validation.** Reject/flag non-finite predictions, out-of-schema features, wrong sample counts, and clamp or flag predictions outside the plausible target range according to a documented rule. Failures are counted and logged without crashing the consumer.
5. **Latency measurement.** Measure per-record: feature extraction time, preprocessing time, model predict time, total processing time. Report mean/median/p95/max over a real run. Report only measured values; state hardware/context.
6. **Concurrency/ordering.** Heavy compute must not break the MQTT keepalive/loop (offload or structure appropriately). Keep the Phase 2 duplicate/out-of-order/gap semantics; define and test what inference does for a duplicate or out-of-order record (e.g. skip duplicates, flag out-of-order) rather than leaving it undefined.
7. **Optional InfluxDB.** If InfluxDB is enabled, write predictions (`predicted_rul`, model_version, latency, validity) through the existing writer pattern. If it is disabled or unreachable, inference must continue unaffected. Test both modes. Do not add Grafana.
8. **Entry point.** A runnable script/module (e.g. a `live_inference` runner) consistent with repo conventions, with clean shutdown and structured logs that never print credentials.

## 6.4 Real integration test (required)

Replay a real slice of NASA IMS snapshots through HiveMQ Cloud, consume them, run live inference, and verify: all expected records produce predictions, sequence numbers are preserved, predictions are finite, online-vs-offline parity holds for the replayed snapshots, latency stats are recorded, and no credentials appear in logs. If HiveMQ credentials are unavailable in your environment, mark this **UNVERIFIED** and give the user the exact command to run it.

## 6.5 Tests

Model-loader success/failure paths; schema-mismatch rejection; parity test; prediction validation edge cases (NaN, wrong length, extreme values); duplicate/out-of-order behavior; latency instrumentation; InfluxDB on/off; no-label-leakage check on `PredictionRecord`; full suite.

## 6.6 Out of scope

Health states/thresholds (Phase 7), retraining, changing the feature set or the model (if a Phase 5 defect is discovered, report it and propose a scoped fix rather than quietly changing training), legacy deletion, Grafana.

## 6.7 Acceptance criteria

- [ ] Real HiveMQ → TelemetryRecord → features → preprocessing → XGBoost → RUL path demonstrated (or marked UNVERIFIED with command)
- [ ] No retraining, no scaler fitting online; schema validated at startup
- [ ] Online/offline feature parity proven by test
- [ ] Prediction validation and latency stats implemented and measured
- [ ] Works with InfluxDB enabled and disabled
- [ ] Docs updated; full suite green; secrets clean; committed; pushed; tree clean
- [ ] **STOP**

---

# PHASE 7 — HEALTH STATE / DEGRADATION LOGIC

## 7.1 Objective

Turn raw predicted RUL into an understandable machine-health state, with defensible, documented thresholds and robust transition logic.

## 7.2 Concept separation (must be explicit in code and docs)

| Concept | Meaning | Source |
|---|---|---|
| Predicted RUL | Supervised estimate of remaining life | XGBoost |
| Anomaly score | Unsupervised deviation from normal | Isolation Forest (legacy) |
| Health index | Optional normalized 0–1 summary | defined here, if at all |
| Machine state | Discrete operational interpretation | defined here |

Never merge or relabel these. Never present the Isolation Forest score as RUL or vice versa.

## 7.3 Requirements

1. **State scheme.** Define states (candidate: `NORMAL`, `WATCH`, `CRITICAL`; add `UNKNOWN`/`INVALID` for missing or invalid predictions if useful). Justify the scheme.
2. **Threshold derivation.** Thresholds must be **derived and justified**, not invented. Acceptable approaches: thresholds on predicted RUL expressed as fractions/amounts of remaining life tied to the target definition and to a stated maintenance-lead-time assumption (clearly labeled as an assumption); calibrated using **training/validation** predictions only. Never choose thresholds by looking at test results. Record the rationale, the data used, and the assumptions in a threshold document. If a threshold is an engineering assumption rather than a data-derived value, say so plainly.
3. **Noise handling.** Single-snapshot predictions are noisy. Design smoothing/persistence/hysteresis from the predictive-maintenance logic (e.g. a causal rolling median/EMA on predicted RUL, enter/exit thresholds with a margin, minimum consecutive confirmations, and a rule that prevents rapid flapping). **Do not simply copy the old 3-snapshot Isolation Forest persistence rule;** if you reuse any of it, justify why. Smoothing must be strictly causal (no future snapshots).
4. **Transition rules.** Define allowed transitions (e.g. whether `CRITICAL → NORMAL` is allowed directly, whether recovery requires sustained evidence). For a degrading machine, document the policy honestly (e.g. latching/non-recovering vs. recoverable) and justify.
5. **Gaps and bad data.** Define behavior for missing/gapped/duplicate/out-of-order snapshots and invalid predictions (hold last state, mark stale, or degrade to `UNKNOWN`) and test it.
6. **State record.** A `MachineHealthRecord` with machine_id, snapshot_sequence, timestamps, predicted_rul, smoothed_rul, state, previous_state, transition flag, reason, thresholds version, model_version. No ground-truth leakage.
7. **Offline evaluation of the state logic.** Replay the stored validation/test predictions from Phase 5/6 through the state machine and report: first-WATCH and first-CRITICAL snapshot positions relative to the end of run, number of flaps, false-alarm behavior in the healthy region, and detection lead time. These are measured results; report them honestly even if unimpressive. The final test region is used for **reporting only**, not for choosing thresholds.
8. **Optional InfluxDB.** Persist state/transition information through the existing optional writer if enabled; core logic works without it.

## 7.4 Tests

State derivation at/around each threshold; hysteresis (no flapping on noisy RUL near a threshold); persistence confirmation; causality of smoothing (no future info); gap/duplicate/out-of-order/invalid handling; transition-table coverage; deterministic replay of a synthetic RUL sequence yielding exact expected transitions; separation of RUL vs anomaly score; full suite.

## 7.5 Out of scope

Retraining, new features, Grafana, dashboards, deleting legacy code, changing the Isolation Forest.

## 7.6 Acceptance criteria

- [ ] States, thresholds, and transition rules defined with documented, honest justification
- [ ] Causal smoothing/hysteresis implemented and tested
- [ ] Offline evaluation reported with measured numbers only
- [ ] RUL, anomaly score, health index, and state kept distinct
- [ ] Docs updated; full suite green; secrets clean; committed; pushed; tree clean
- [ ] **STOP**

---

# PHASE 8 — END-TO-END MQTT PREDICTIVE MAINTENANCE PIPELINE

## 8.1 Objective

Validate the entire system on real NASA IMS snapshots, measured end to end:

```
NASA IMS → simulator → HiveMQ Cloud → MQTT consumer → TelemetryRecord
        → signal processing → canonical features → XGBoost → RUL
        → health state → optional InfluxDB
```

## 8.2 Requirements

1. **Orchestration.** Provide a documented way to run the full pipeline (single command or a small, clear set of commands; docker-compose only where it already applies, e.g. InfluxDB). Include a configurable replay speed so a full 984-snapshot run is practical, and a smaller smoke-test mode.
2. **Full-run validation** using real snapshots (full 984 if time/quota allow; otherwise a clearly stated, representative subset — never imply a full run when it was a subset). Measure and report:
   - message delivery (published vs received vs processed)
   - sequence correctness (gaps, duplicates, out-of-order counts)
   - feature generation success rate
   - inference latency (mean/median/p95/max) and end-to-end latency (publish → state)
   - prediction generation count and validity
   - health-state trajectory over the run
   - persistence success (if InfluxDB enabled), and behavior with InfluxDB disabled
   - resource observations (CPU/memory) if practical
3. **Fault injection / robustness tests (real where practical):**
   - broker disconnect/reconnect mid-run (consumer resumes, no crash, state sane)
   - duplicate message injection
   - out-of-order message injection
   - sequence gap injection
   - malformed/invalid payloads (wrong channel count, wrong sample count, NaN)
   - InfluxDB unavailable (pipeline continues)
   - consumer restart behavior (document what happens to state)
   Use clearly labeled test topics or `machine_id` values so you never contaminate the production topics, and clean up afterwards.
4. **Consistency check.** Compare end-to-end live predictions to offline predictions for the same snapshots and report the maximum deviation. Explain any non-zero difference.
5. **Honest result reporting.** Produce `reports/end_to_end_validation_report.md` containing the commands run, commit hash, dates, environment, and **measured** results only. Anything not run is listed as not run. Failures found are reported, then either fixed (if small and in scope) or listed as known issues.
6. **Honesty about the data source** appears in the report and in any run banner: replayed NASA IMS historical telemetry.

## 8.3 Tests

Automated integration test(s) for the smoke mode; fault-injection tests; end-to-end parity test; confirm credentials never appear in logs/output; full suite.

## 8.4 Out of scope

Legacy cleanup (Phase 9), new modeling work, Grafana. If the end-to-end run reveals a defect in an earlier phase, make the minimal fix, add a regression test, and flag it clearly in the report.

## 8.5 Acceptance criteria

- [ ] Full path run on real NASA IMS snapshots through HiveMQ Cloud (scope stated honestly)
- [ ] Delivery, sequence, latency, prediction, persistence, reconnect, duplicate, and out-of-order behaviors measured and reported
- [ ] Online/offline consistency quantified
- [ ] Works with InfluxDB on and off
- [ ] Validation report contains only measured results
- [ ] Full suite green; secrets clean; committed; pushed; tree clean
- [ ] **STOP**

---

# PHASE 9 — LEGACY ARCHITECTURE MIGRATION / CLEANUP

## 9.1 Objective

Only now that the MQTT + XGBoost pipeline is proven, decide what to retain, deprecate, or remove from the legacy stack. **Evidence-based, conservative, reversible.**

## 9.2 Preconditions

Phase 8 validation report exists and shows the new path is stable. If it does not, report that and **do not delete anything**.

## 9.3 Procedure

1. **Inventory** all legacy components: Flask app/routes, MongoDB usage and prediction storage, Streamlit app, old inference paths, Isolation Forest/PCA training and artifacts, old anomaly routes, old Docker/config entries, and dependencies in `requirements*.txt` / package files.
2. **Dependency analysis** for each item: who imports/calls it (static search + tests), whether the new path or any test depends on it, whether docs/viva material depend on it.
3. **Decision table** with one row per component: `KEEP` / `DEPRECATE (mark, keep)` / `REMOVE` / `ARCHIVE`, with the evidence for the decision. Write it into `reports/legacy_cleanup_decision.md` **before** changing anything.
4. **Isolation Forest/PCA.** Decide with evidence whether to keep them as an offline baseline/comparison, a clearly labeled legacy anomaly engine, or to retire them. If kept, label them as such everywhere and do not present them as the main predictive model. If removed, ensure no test, doc, or dependency still references them.
5. **Remove/deprecate only what is proven obsolete.** Prefer deprecation notices and relocation to a `legacy/` or `archive/` area over outright deletion when in doubt. Do not delete the raw dataset, the model artifacts needed by the live path, Phase 1–8 code, or anything the new path imports.
6. **Dependencies.** Remove unused packages only after confirming (by running the tests and the end-to-end smoke path) that nothing needs them. Update `.env.example` and compose files to match; keep MQTT mandatory, InfluxDB optional, Grafana absent.
7. **Docs and tests.** Update README/architecture docs so they describe the *current* system only; remove or mark obsolete tests that tested removed code (justify each); keep coverage of everything that remains.
8. **Verification.** After cleanup: full test suite, plus the Phase 8 smoke test against HiveMQ Cloud, plus a fresh-environment install check (e.g. create a clean venv and install from the updated requirements) if practical.

## 9.4 Out of scope

New features, retraining, changing the health-state logic, Grafana.

## 9.5 Acceptance criteria

- [ ] Decision table with evidence exists and was written before deletions
- [ ] Only proven-obsolete code removed/deprecated; new path untouched
- [ ] No dangling imports, tests, docs, or dependencies referencing removed code
- [ ] Full suite and end-to-end smoke test pass after cleanup
- [ ] Committed (consider splitting "deprecate" and "remove" into separate commits if cleaner); pushed; tree clean
- [ ] **STOP**

---

# PHASE 10 — FINAL DOCUMENTATION, REPRODUCIBILITY & VIVA PREPARATION

## 10.1 Objective

Produce the final, accurate, defendable documentation and viva material. Everything must match the repository as it actually is. Verify each claim against the code before writing it.

## 10.2 Deliverables

### A. Final project documentation (README + `docs/` or the repo's convention)

Cover all of: problem statement; dataset (NASA IMS Set 2 facts as verified); overall architecture; MQTT architecture; HiveMQ Cloud usage (TLS, port 8883, auth by env vars, QoS 1, topic structure, status/LWT, retain policy); telemetry schema; signal processing; feature engineering; target construction (RUL); XGBoost model; chronological split and leakage prevention; RUL methodology and its limitations (including the extrapolation limitation measured in Phase 5); health-state logic and thresholds (with the stated assumptions); InfluxDB's optional role and schema (if retained); testing strategy and measured results; security (secrets handling, TLS, `.env.example`); setup and deployment instructions that a fresh machine can follow; reproducibility commands; Git history summary (phase-by-phase commits); known limitations; future improvements.

### B. Diagrams

Architecture diagram and data-flow diagram in a format that renders in the repo (Mermaid in Markdown preferred; keep source text, not only an image). Include the replay-vs-live honesty note in the diagram caption.

### C. Reproducibility check

Follow your own setup instructions in a clean environment as far as practical (clean venv, install, run tests, train from raw data, run the smoke pipeline). Fix doc errors you discover. Record what was actually verified.

### D. Viva preparation (`docs/viva_preparation.md`)

Concise, honest Q&A, answered from the real project, with the likely follow-up questions. At minimum:

- Why MQTT? Why HiveMQ Cloud? Why QoS 1? Why TLS? Why retain=false for telemetry?
- Why is NASA IMS *replayed* rather than live sensor data, and how do you describe that honestly?
- Why XGBoost, and why is it *not* an anomaly detector?
- How was the RUL target constructed, given that NASA IMS has no explicit RUL labels? Why capped/linear/normalized? What are the limitations?
- What is temporal leakage, how was it prevented, and how is it tested?
- Why can a tree model not extrapolate RUL beyond the training range, and how did you handle it?
- How were health thresholds chosen, and which parts are engineering assumptions?
- How do you handle duplicates, out-of-order messages, gaps, reconnects?
- What is InfluxDB's optional role, and why is Grafana not required?
- What happened to Isolation Forest/PCA and why?
- What are the actual measured results and what do they honestly show and not show?
- What would you do next (additional run-to-failure datasets, multi-machine support, model monitoring, etc.)?

Include a short "weak points an examiner may probe" list with honest answers (single run-to-failure trajectory, no real sensors, assumed failure semantics, small test region, etc.).

### E. Final consistency audit

Grep the repo/docs for stale or dishonest statements: claims of real live sensors, Grafana as mandatory, XGBoost as anomaly detection, Isolation Forest as the main model, invented metrics, references to deleted components, and any real secrets. Fix or report each finding.

## 10.3 Rules

- Documentation reports **only measured numbers** from the earlier phases' reports/commands. If a number cannot be traced to a report or a rerun, omit it or rerun it.
- No new features. Only minimal fixes needed to make docs and reality agree.

## 10.4 Acceptance criteria

- [ ] Complete documentation set matching the actual code
- [ ] Architecture + data-flow diagrams committed as text-source
- [ ] Clean-environment reproducibility check performed and recorded
- [ ] Viva Q&A document covering every required topic plus weak-point list
- [ ] Consistency audit completed; no secrets; no dishonest claims
- [ ] Full suite green; committed; pushed; tree clean
- [ ] **STOP — project complete.** Provide a final summary of the whole Phase 0–10 journey.

---

# PART C — CROSS-PHASE CHECKLIST (verify at the end of every phase)

- [ ] Started with the repository/Git inspection protocol (B.1)
- [ ] Reused existing modules; no duplicate implementations
- [ ] No invented data, labels, thresholds, metrics, or results
- [ ] Source described honestly as replayed NASA IMS historical telemetry
- [ ] MQTT remains mandatory; InfluxDB optional; no Grafana
- [ ] No temporal leakage; test set untouched until final evaluation (Phases 5–8)
- [ ] New tests added; phase tests + full suite run and green
- [ ] Real HiveMQ integration done where relevant, or explicitly marked UNVERIFIED
- [ ] No secrets in code, docs, logs, or staged diff
- [ ] Docs updated; discrepancies recorded
- [ ] Committed, pushed, push verified, working tree clean
- [ ] Stopped; did not start the next phase

---

# PART D — PHASE REPORT TEMPLATE (output this at the end of every phase)

```
PHASE <N> REPORT
Branch: feature/industrial-stack-migration
Start commit: <hash>      End commit: <hash>
Date: <date>

1. Repository state found at start (including discrepancies vs. this prompt)
2. What was implemented (files created/modified)
3. Key design decisions and justifications
4. Tests added; phase-test results; full-suite results (counts, pass/fail)
5. Real integration results (HiveMQ etc.) — commands run, outcomes; anything UNVERIFIED
6. Measured results (only numbers actually produced, with the commands that produced them)
7. Known limitations / issues / items deferred to later phases
8. Security check (what was verified about secrets)
9. Git: commit message, push verification, clean-tree confirmation
10. Acceptance-criteria checklist (ticked/unticked honestly)

STOPPED. Awaiting instruction for the next phase.
```

---

# PART E — FINAL REMINDER

Optimize for **correctness, reproducibility, explainability, temporal integrity, real MQTT integration, valid supervised learning, clean architecture, strong testing, and professional engineering practice** — not for adding technologies. When in doubt: inspect the repository, prefer evidence over assumption, report honestly, and stop at the end of the phase.
