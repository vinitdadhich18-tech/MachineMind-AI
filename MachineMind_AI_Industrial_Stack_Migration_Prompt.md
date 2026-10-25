# MachineMind AI — Industrial Stack Migration: Master Engineering Prompt

**Audience:** Antigravity (AI coding agent)
**Owner:** 3rd-year CS student, academic/research project
**Mandate:** Evolve the existing MachineMind AI repository into a system genuinely built around **MQTT + InfluxDB + XGBoost + Grafana**.
**Nature of this file:** An engineering specification and mentorship guide. It contains **no implementation code** and **no assumed filenames**. Every file, module, endpoint, collection, and script name must be discovered by inspecting the repository.

---

## 0. How to Read and Use This Document

1. Read the whole document before touching the repository.
2. Your **first deliverable is NOT code**. It is the Phase 0 audit and design report (Section 22). Stop after delivering it and wait for the owner's approval.
3. Where this document says "inspect and determine", you must actually inspect. Do not guess.
4. Where this document says "decide and justify", record the decision, the alternatives considered, and the reason in a decision log (Section 26).
5. The owner is a student who must be able to **defend every design choice orally**. Prefer clear, explainable designs over clever ones. Explain each architectural change in plain language as you go.
6. If a requirement is genuinely ambiguous or two requirements conflict, **ask**. Do not silently choose.

---

## 1. Project Context

**Project:** MachineMind AI — Industrial Predictive Maintenance & Vibration Monitoring Platform.

**Purpose:** Predictive maintenance and vibration monitoring using the NASA IMS Bearing Dataset (primarily Set 2).

**Current data situation:**
- The dataset is historical, run-to-failure vibration data from bearing test rigs.
- Set 2 is believed to contain 984 snapshot files, each holding multi-channel vibration measurements. **Verify all dataset facts against the repository's data and the dataset readme** (sampling rate, samples per snapshot, channel count, file-naming/timestamp convention, snapshot interval, which bearing failed, how the run ended). Do not trust this document or memory for these numbers; confirm them and record them in the audit.
- The project currently processes historical snapshots. It does not receive physical sensor telemetry.

**Honesty principle (non-negotiable):** The project must NOT claim to have physical industrial sensors. NASA IMS plays two legitimate roles:
1. The research and training dataset.
2. A **simulated telemetry source** (replayed historical data) that demonstrates a real-time MQTT architecture.

All documentation, dashboards, and demo scripts must describe the data as "replayed NASA IMS historical telemetry".

**Teacher's requirement (verbatim intent):** "The entire system must be built around MQTT + InfluxDB + XGBoost + Grafana."

"Built around" means each technology carries a real architectural responsibility on the main data path. Presence in `requirements.txt` or a decorative container does not count.

---

## 2. Existing Architecture (as described by the owner — verify it)

```
Raw NASA IMS snapshot
   -> signal preprocessing
   -> feature extraction
   -> RobustScaler
   -> Isolation Forest + PCA
   -> anomaly score
   -> threshold
   -> 3-snapshot persistence
   -> NORMAL / WATCH / ANOMALY CONFIRMED
   -> Flask REST API / MongoDB / Streamlit
```

Existing dashboard concepts to preserve in spirit: machine status, channel condition, anomaly score trend, threshold, persistence count, recent incidents, machine summary, history, alerts, model information.

This description is the owner's summary. The repository is the source of truth. Where they differ, report the difference.

---

## 3. Current Technology Stack (to be audited, not assumed)

| Layer | Technologies reportedly in use |
|---|---|
| ML / data | Python, NumPy, pandas, SciPy, scikit-learn, joblib, Isolation Forest, PCA, RobustScaler, vibration feature engineering, thresholding, persistence logic |
| Backend | Python, Flask, Flask-CORS, REST API, PyMongo |
| Frontend | Streamlit, Altair/charting |
| Database | MongoDB |
| Data | NASA IMS Bearing Dataset |

---

## 4. Mandatory Target Technology Stack

| Technology | Mandatory role (one-line) |
|---|---|
| **MQTT** (real broker) | Real-time telemetry transport decoupling producer from ML pipeline |
| **InfluxDB** | Primary time-series store for telemetry-derived data, features, predictions, health and state |
| **XGBoost** | Primary predictive model on a scientifically defined supervised target |
| **Grafana** | Primary industrial monitoring UI, querying InfluxDB |

All four are mandatory. All four must appear on the live path in Section 5.

---

## 5. Target Architecture

```
NASA IMS dataset (historical files)
        |
        v
Telemetry Simulator / Replay Engine
        |
        v
MQTT Publisher ──> MQTT Broker ──> MachineMind MQTT Consumer (ingestion)
                                         |
                           ┌─────────────┼───────────────┐
                           v                             v
                 Signal processing +              InfluxDB (telemetry-
                 feature extraction               derived measurements)
                           |                             ^
                           v                             |
                 XGBoost inference                       |
                  (RUL / degradation / risk)             |
                           |                             |
                           └── predictions + state ──────┘
                                         |
                                         v
                                      Grafana
```

Principles of this architecture:
- **The primary telemetry path is MQTT -> ingestion -> ML -> InfluxDB -> Grafana.** It must not pass through Flask, MongoDB, or Streamlit.
- The simulator is a *producer* that could later be replaced by a real gateway/sensor with **zero changes** to the consumer. Design the payload contract with that in mind.
- The consumer must be unaware that data is simulated, except for an explicit, honest `source` marker in the payload metadata (e.g., indicating replayed dataset vs live).

Exact implementation details are determined only after the Phase 0 audit.

---

## 6. Migration Philosophy

1. **Evolve, do not rewrite.** The existing project is a working baseline. Reuse proven domain logic (preprocessing, feature engineering, persistence concept, status vocabulary) wherever it is correct.
2. **Inspect before changing.** Nothing is modified before the audit is complete and approved.
3. **Incremental and reversible.** Small phases, each tested and committed. The old path keeps working until the new path is verified (strangler pattern).
4. **No silent semantic change.** If a metric's meaning changes (e.g., "anomaly score" -> "predicted RUL"), rename it, document it, and update every consumer.
5. **No cargo-cult technology.** Each of the four mandatory technologies must do real work. Each retained legacy component must have a justified role, or be removed after verification.
6. **Scientific honesty beats good-looking results.** If the data cannot support a claim, say so in the documentation rather than tuning until a number looks good.
7. **Reproducibility.** A reviewer should be able to clone the repo, follow the README, and reproduce the demo and the model training results.

---

## 7. MQTT — Detailed Responsibilities

**Role:** The transport layer for real-time telemetry between producer(s) and MachineMind.

**Responsibilities:**
- Carry telemetry messages from the simulator (and in future, real sensors/gateways) to the consumer.
- Decouple producer rate from processing rate.
- Support multiple machines via topic hierarchy.
- Optionally carry lightweight status/heartbeat messages (producer online/offline, e.g., via Last Will and Testament), if justified.

**Must be:**
- A **real MQTT broker** running locally (an established open-source broker; decide and justify). Document how to start and stop it.
- Actually exercised end to end. No in-process fake queue standing in for the broker. No hardcoded "connected" flags.

**Decisions Antigravity must make and document (see Section 12 for principles):** broker choice, client library choice, MQTT version, QoS per topic, retained-message policy, authentication/TLS approach for local dev, payload encoding, reconnect policy.

---

## 8. InfluxDB — Detailed Responsibilities

**Role:** The primary time-series database.

**Responsibilities:**
- Persist telemetry-derived measurements (see the size caveat below).
- Persist engineered features per snapshot per channel.
- Persist model predictions (predicted RUL, degradation score, failure-risk) with model version.
- Persist machine/channel health state and alert state transitions.
- Persist pipeline health metrics (last message received, ingestion latency, consumer errors) so Grafana can show data ingestion status.
- Serve Grafana's queries efficiently.

**Important caveat on raw waveform storage:** A NASA IMS snapshot is a long, high-frequency waveform per channel. Storing every raw sample point in InfluxDB is probably wasteful and unnecessary. Antigravity must decide and justify what goes in InfluxDB:
- Features and summary statistics per snapshot/channel (certainly).
- Raw or downsampled waveform (only if there is a real dashboard need, e.g., a "latest snapshot waveform" panel; consider a bounded retention or a separate bucket).

**Version decision:** InfluxDB has major versions with different query languages and client libraries. Inspect what is practical (client library maturity, Grafana datasource support, docs available to the student), check current documentation, **decide and justify** the version. Do not mix client APIs from different versions.

Schema principles are in Section 13.

---

## 9. XGBoost — Detailed Responsibilities

**Role:** The primary predictive ML model on the live inference path.

**Core rule:** XGBoost is **supervised**. Isolation Forest is **unsupervised**. They are not interchangeable. Replacing one with the other without defining a supervised target is scientifically invalid. Antigravity must establish the target first (Section 14).

**Responsibilities:**
- Train on engineered vibration features with a defensible, deterministically derived target.
- Produce predictions at inference time inside the consumer pipeline, for each incoming snapshot.
- Be versioned, serialized, and loaded reproducibly.
- Be evaluated honestly with a time-aware protocol.

**Preferred direction:** Degradation/RUL regression (predicted RUL, optionally a degradation index). A classification head (e.g., "failure within N snapshots") is acceptable **only if derived deterministically from the RUL definition**, not invented labels. If the repository or dataset structure argues for a different validated target, document why.

**Anti-requirements:** Do not add XGBoost as decoration (e.g., trained but unused at inference, or trained on synthetic or randomly assigned labels). Do not name an XGBoost output "anomaly score".

---

## 10. Grafana — Detailed Responsibilities

**Role:** The primary industrial monitoring and visualization layer.

**Responsibilities:**
- Query InfluxDB as a provisioned datasource.
- Show live and historical telemetry-derived measurements, features, predictions, health, alerts, and pipeline status.
- Be reproducible: datasource and dashboards should be **provisioned from version-controlled configuration**, not hand-clicked once and lost.
- Every panel must have a traceable query to real stored data.

Dashboard requirements are in Section 16.

---

## 11. NASA IMS Telemetry Simulator / Replay Engine — Design

**Purpose:** Turn historical snapshots into a controlled, chronological stream of MQTT messages that behaves like a machine producing telemetry.

**Requirements:**
- Replays snapshots in **strict chronological order** using the dataset's own timestamps.
- Controllable **replay speed** (real-time cadence is impractical, since original snapshot spacing is on the order of minutes, so a speed multiplier or fixed-interval mode is needed; document both semantics).
- Selectable machine/test run (dataset set, e.g., Set 2) and start/stop/resume position.
- Publishes each snapshot through the real MQTT broker.
- Does **not** fabricate, interpolate, smooth, or augment measurements. It reads real files and emits them.
- Emits clear metadata: machine ID, channel IDs, original timestamp, snapshot sequence number, sampling information, `source` indicator (replayed dataset).
- Does not leak labels or future information. The simulator must never publish RUL, failure time, or "this is the failing bearing" hints in the telemetry payload. The consumer must infer from measurements only.
- Handles end of dataset gracefully (stop, loop option clearly labeled, or signal completion).
- Handles broker disconnects (retry/backoff; no silent message loss without logging).

**Timestamp strategy (critical design decision, decide and justify):**
Original IMS timestamps are from 2004. Grafana's default time ranges are relative to "now". Options include:
1. Preserve original event timestamps as the InfluxDB point time and instruct dashboards to use absolute time ranges.
2. Map replay to a shifted/"virtual live" timeline (preserving relative spacing) and store the original dataset timestamp as a separate field, along with an ingest timestamp.
3. A hybrid.

Whichever is chosen, the original dataset timestamp, the replay (ingest) time, and the processing time must remain distinguishable and documented. Explain clearly in the README how a viewer sees "live" behavior in Grafana.

**Payload size caveat:** A raw snapshot can be large. Decide how to transport it (one message per snapshot vs. per channel vs. chunked; JSON vs. compact binary or compressed encoding), considering broker/client message size limits and parse cost. Justify with a measurement, not a guess.

---

## 12. MQTT Topic and Payload Design Principles

**Topic hierarchy:** Use a consistent, documented hierarchy. A pattern such as `machinemind/{machine_id}/telemetry` is a reasonable starting point, but Antigravity must decide whether channel appears in the topic or the payload, and whether separate topics are needed for status/heartbeat. Rules:
- Lowercase, stable, no spaces; no leading slash.
- Machine ID in topic enables wildcard subscription and per-machine ACLs later.
- Do not put high-cardinality or changing values (timestamps, sequence numbers) in topics.
- Define separate subtopics for telemetry vs. status/events.

**Payload contract (write it as a formal spec, e.g., a schema document, before implementing):**
- Schema version field (for evolution).
- Machine ID, channel identifiers.
- Event timestamp in an unambiguous format (UTC, ISO 8601 or epoch with declared unit).
- Snapshot sequence/ID for deduplication and gap detection.
- Sampling rate and sample count.
- Data representation and encoding.
- Source/provenance field (replayed dataset vs. live sensor).
- Optional integrity information (e.g., sample count check).

**Behavior decisions to document:**
- **QoS:** choose per topic with reasoning (e.g., consequences of at-most-once vs. at-least-once on telemetry; consequences of duplicates for InfluxDB writes and how idempotency is handled).
- **Retained messages:** only where meaningful (e.g., last known producer status); avoid retaining large telemetry payloads.
- **Persistent sessions / client IDs:** decide so that a consumer restart does not silently drop data unexpectedly, or document that it can.
- **Reconnect:** automatic reconnect with backoff, resubscribe on reconnect.
- **Validation:** consumer rejects malformed/unsupported-version/out-of-order-flagged messages, logs them, and does not crash. Consider a dead-letter style log or counter.
- **Idempotency/ordering:** define behavior for duplicates and out-of-order arrival (features computed with history depend on order, see Section 15).
- **Security:** local dev may use simple auth, but credentials come from environment config, never committed. Document how TLS/authentication would be enabled beyond local demo.

---

## 13. InfluxDB Schema Design Principles

**Write the schema document before implementing writes.** The document must be reviewed in Phase 0/3.

Principles:
- **Tags** are low-cardinality, indexed identifiers used to filter/group: e.g., machine ID, channel, source type, model version (if cardinality stays small), status category.
- **Fields** are the measured/computed values: feature values, predicted RUL, degradation score, risk probability, thresholds, counters.
- **Never** put unbounded or continuously changing values in tags (timestamps, sequence numbers, raw floats). Avoid high-cardinality tags.
- Separate **measurements** by semantic meaning and write cadence. Candidates for the schema document to evaluate (names illustrative, final names are Antigravity's decision):
  - raw/summary telemetry per snapshot/channel
  - engineered features per snapshot/channel
  - model predictions per snapshot (per machine; per channel if the model is per channel)
  - machine/channel health and alert state (written on each evaluation or on state change, decide)
  - pipeline/ingestion health metrics
- Decide **bucket(s)** and **retention policy** per data class (e.g., raw waveform, if stored at all, shorter retention than features and predictions).
- Define **timestamp semantics** (Section 11) and field naming conventions with units.
- Define write **idempotency**: writing the same point (same measurement, tags, timestamp) overwrites rather than duplicates; make sure replays do not create duplicate rows. Think carefully when tags differ between replays.
- Prefer **batched writes** with explicit error handling and retry; do not silently drop failed writes.
- Document **example queries** that Grafana panels will use, and confirm they are efficient (aggregations, windowing, limited time ranges).

---

## 14. XGBoost Target and Label Strategy

This section is the scientific heart of the migration. Antigravity must complete it **in writing, and have it approved by the owner**, before any training code is written.

### 14.1 Understand the data first
Record in the audit:
- How many run-to-failure experiments are available in the repository (IMS has multiple sets; Set 2 is believed to contain one run-to-failure sequence with a single bearing failure; **verify**).
- Which bearing(s)/channel(s) failed, and how the run ended (documented in the dataset readme).
- Whether other sets (e.g., Set 1, Set 3) are present, could be added, and what their channel layouts and failure modes are.

### 14.2 Define the target (do not invent labels)
Candidate: **RUL** expressed in a consistent unit (e.g., remaining snapshots or hours), derived from the run's known end time: `RUL(t) = t_end_of_run − t`. Variants to evaluate and justify:
- **Linear RUL** (simple, but unrealistically predicts decline during the healthy phase).
- **Piecewise-linear / capped RUL** (constant "healthy" plateau, then linear decline), common in PHM literature. The cap value is a hyperparameter. It must be chosen with documented reasoning and **not tuned on the test segment**.
- **Degradation index / health indicator** derived from features, only if its construction is transparent and non-circular.
- **Derived risk class** (e.g., "RUL below N snapshots") strictly a deterministic function of the RUL target.

Rules:
- The "end of run" is a property of the dataset documentation. Using it to construct training **labels** is legitimate. Using it as an **input** at inference is not.
- Do not randomly assign healthy/failure labels. Do not hand-label by eyeballing plots without documenting criteria.
- If a supervised target cannot be justified for the data at hand, say so, escalate, and propose alternatives. Do not fake it.

### 14.3 Scientific limitation Antigravity must confront honestly
A single run-to-failure sequence gives one trajectory. A chronological split within it means:
- The test segment is the *end of life*, with RUL values and feature magnitudes that may lie **outside the range seen in training** (tree models cannot extrapolate beyond training target range).
- Neighboring snapshots are highly autocorrelated, so naive random splits wildly overstate performance.

Antigravity must evaluate and recommend (with the owner's approval) among strategies such as:
1. **Multi-run training** using additional IMS sets/bearings, with **leave-one-run-out (or leave-one-bearing-out)** evaluation as the headline metric, if the additional data is available and its channel/failure semantics are documented and compatible.
2. **Chronological split within a single run** with explicit gaps/embargo between train, validation, and test segments, and clearly stated limits on what the result proves.
3. A combination, using chronological evaluation as a secondary check.

Whatever is chosen, the README and the final report must plainly state what the evaluation does and does not demonstrate. Do not oversell.

### 14.4 Evaluation
- Metrics for regression: RMSE, MAE, and R² where appropriate; if RUL is predicted, also a PHM-style asymmetric scoring function (state which definition and why late predictions are penalized more).
- Baselines to compare against (cheap and honest): a naive baseline (e.g., predict-the-mean/last-value approach) and, optionally, a simple linear model; and the legacy Isolation Forest signal as a *comparison* (not as a fair "RUL" predictor, just as a reference for how the health indicator evolves).
- Plots of predicted vs. true RUL over time for the held-out segment.
- Hyperparameter tuning only with time-aware validation (never random K-fold). Early stopping uses the validation segment, never the test segment.
- Track experiments (parameters, data version, feature list, metrics, model version) in a reproducible way (e.g., saved metadata alongside the model artifact).
- Set seeds; document environment/library versions.

### 14.5 Model packaging
- Serialized model artifact + feature schema (names, order, dtypes) + scaler/preprocessor artifact (if any, fit on training only) + training metadata + version identifier.
- The inference code must **refuse** to run (loud error) if the incoming feature vector does not match the saved feature schema.
- The version identifier must be written alongside each prediction in InfluxDB.

---

## 15. Data Leakage Prevention (critical)

Antigravity must implement **and test** the following, and document each decision:

1. **Chronological order is sacred.** Splits are by time (or by run), never random.
2. **Causal feature computation.** Any feature using history (rolling mean/std, trend/slope, EWMA, deltas, lagged features) must use **only past and current snapshots**. No centered windows, no future look-ahead, no whole-run normalization.
3. **Scaler/preprocessor fit on training data only.** Fit on the training split, then apply to validation/test/inference. Same for any imputation or feature selection.
4. **Target construction is isolated.** The RUL label is computed from the dataset end-of-run and used **only as a label**. No feature may be derived from it, directly or indirectly (e.g., normalizing features by "time to end of run", or using snapshot index/normalized time as a feature, which would encode RUL).
5. **No test-set influence.** Hyperparameters, thresholds, the RUL cap, early stopping, feature selection, and the choice of persistence parameters must not be informed by test results.
6. **Embargo/gap** between train and validation/test segments sized to cover the longest rolling-window dependence.
7. **Inference-time parity.** The streaming consumer must compute features exactly as the training pipeline did, using only data that has arrived so far. Implement a **parity test**: replay a segment through the streaming path and verify features equal the offline-pipeline features for the same snapshots (within numeric tolerance).
8. **Warm-up handling.** Define behavior when insufficient history exists at stream start (e.g., first few snapshots): emit "insufficient history" state rather than fabricated values, and make training consistent with that.
9. **Threshold calibration** for status/alerting uses training/validation data only.
10. **Leakage tests** (automated): shuffle-future test (altering future snapshots must not change past features/predictions), scaler-fit-source test, split-ordering test, and a target-isolation test.

Document all of the above in a leakage-prevention section of the final documentation.

---

## 16. Grafana Dashboard Requirements

Build an industrial, functional dashboard, not a decorative one. Provision datasource and dashboards from version-controlled config.

**Required panels (adapt layout after the audit):**
1. Machine status (NORMAL / WATCH / CONFIRMED-style state, derived from stored state data)
2. Live vibration telemetry / latest snapshot summary
3. Channel 1 trend (e.g., RMS and selected features)
4. Channel 2 trend
5. Channel 3 trend
6. Channel 4 trend
7. Risk / degradation trend (clearly labeled with what it is)
8. Predicted RUL (with units)
9. Degradation / health indicator trend
10. Recent alerts / status transitions
11. Model information (model version, training date, target definition)
12. Last telemetry timestamp (and "data freshness" indicator)
13. Data ingestion status (message rate, ingestion lag, error counters)

**Rules:**
- Every panel maps to a documented InfluxDB query. No hardcoded numbers, no dummy series.
- Where ground-truth RUL is known (replay of a dataset with known end), an *evaluation-mode* panel may overlay predicted vs. true RUL, but it must be **clearly labeled as simulation/evaluation-only** and derived from stored evaluation data, and never from the live path inputs. Decide with the owner whether to include it.
- Units, thresholds, and time ranges are explicit. Thresholds shown are the real ones used by the alert logic.
- Handle the "replay timeline" correctly (Section 11) so dashboards are not empty or misleading.
- Include variables (machine selector, channel selector) where sensible.
- Alerting: decide whether Grafana alert rules, application-level state, or both generate alerts; document why, and avoid two contradictory alert sources.

---

## 17. Streamlit Migration Strategy

- **Audit:** list every page/view/widget in the current Streamlit app and what data source and API call each uses.
- **Map each function to Grafana:** for each feature, mark MIGRATE (to a Grafana panel), RETAIN (with justification), or DROP (with justification). Produce a mapping table.
- **Grafana is the primary dashboard.** Do not run two competing dashboards without a justified purpose.
- Streamlit may remain only if it has a **clearly distinct role** (e.g., an offline research notebook-style tool for model experiments, or an admin/ops tool). Record the justification.
- Do not delete Streamlit until the Grafana dashboard demonstrably covers the migrated functionality (Phase 7 verified), then remove it in the cleanup phase (Phase 9), in its own commit.
- Keep any legacy-vocabulary UI semantics (e.g., NORMAL/WATCH/ANOMALY CONFIRMED) only if the new semantics (Section 21) justify them.

---

## 18. Flask Migration Strategy

- **Audit:** list each endpoint, its consumers, its data source, and whether it is on the telemetry path.
- Classify each endpoint: **REMOVE** (redundant after MQTT/InfluxDB/Grafana), **REPLACE**, **RETAIN** (with purpose).
- The live telemetry path must **not** depend on Flask.
- Acceptable retained roles (only if actually needed): model metadata/health endpoint, administrative/config operations, replay-control API, a health check for the ingestion service. Do not keep Flask solely because it exists.
- If retained, apply configuration via environment, validate input, and document each remaining endpoint.
- Remove only after verification; removal is a separate commit.

---

## 19. MongoDB Migration Strategy

- **Audit:** list collections, document shapes, writers, readers, and volume.
- Classify data:
  - **Time-series** (telemetry, features, scores, status history, alerts-as-events) -> migrate to InfluxDB.
  - **Application/document data** (e.g., user/config documents, if any) -> decide whether a document store is genuinely needed.
- Provide a **migration/backfill** approach for any historical data worth preserving (verify correctness by comparing counts and sampled values). If the data can be regenerated deterministically from the NASA files, regenerating is preferable to migrating.
- Do not keep MongoDB for telemetry. Do not delete it until the InfluxDB path is verified. If nothing genuinely requires it, remove the dependency cleanly (code, config, docs, environment variables) in the cleanup phase.

---

## 20. Isolation Forest + PCA Migration Strategy

- **Audit:** where used, what its output means, who consumes it, what thresholds were calibrated on what data.
- Decide its fate and document it:
  - **Baseline/comparison experiment** (offline research notebook/report comparing unsupervised anomaly detection against XGBoost), and/or
  - **Supplementary anomaly detector** on the live path, with a distinct name and distinct measurement/field (e.g., stored as "anomaly score" separately from XGBoost outputs), clearly explained in docs and dashboard, and/or
  - **Retired** after XGBoost supersedes it (only after comparison).
- **No confusion rule:** XGBoost outputs and Isolation Forest outputs must never share a field name, panel title, or threshold. The final architecture diagram must make XGBoost's role unambiguous as the primary model.
- If retained live, it must obey the same leakage rules (scaler/PCA/IF fit on training/healthy-baseline data only; document which data was used).

---

## 21. Persistence and Status Strategy

The legacy logic: 1/3 -> WATCH, 2/3 -> WATCH, 3/3 -> CONFIRMED.

Antigravity must **re-derive** this logic for the new model, not blindly carry it over:

1. Define precisely what the monitored quantity is (predicted RUL, risk probability, degradation index) and its units.
2. Define what **threshold** means for that quantity (e.g., "predicted RUL below X", "risk probability above p"), how it is calibrated (training/validation data only), and its precision/recall trade-off.
3. Define what a **persistence count** means (N consecutive snapshots breaching the threshold), the reset rules (what happens on a non-breach), and whether N=3 remains appropriate given the replay cadence and prediction noise. Evaluate on validation data; do not tune on test.
4. Define status vocabulary and mapping. If "ANOMALY CONFIRMED" no longer describes the semantics (it is a degradation/risk alert now), rename and document it. Do not label an XGBoost prediction an "anomaly score".
5. Persist **state** (current persistence counter, current status) so a consumer restart does not silently reset or corrupt it. Decide where that state lives (e.g., derivable from stored InfluxDB state vs. in-memory with documented behavior).
6. Make status transitions **auditable**: each transition (with reason and the inputs that caused it) is stored in InfluxDB.
7. Add unit tests for the state machine (including boundary cases, reset, restart, and warm-up).

---

## 22. Phase-by-Phase Implementation Roadmap

**Global rules for every phase:**
- Work on the migration branch only.
- Before starting a phase: verify the previous phase's verification commands pass, commit, push, record the commit hash in the progress log.
- A phase is **not complete** because code exists or the app starts. It is complete only when its tests and verification commands have run and the evidence (outputs) is recorded.
- Each phase report must include: Objective, Files affected, Architecture impact, Implementation steps, Tests, Expected output, Failure conditions, Verification commands, Git commit message, Rollback considerations.
- Exact filenames and commands are discovered from the repository. Do not invent them in advance.

---

### PHASE 0 — Repository Audit and Migration Design

**Objective:** Understand reality; produce the design that all later phases follow. **No functional code changes.**

**Files affected:** Documentation only (new design/audit documents, decision log). Determine locations by inspecting existing doc conventions.

**Architecture impact:** None at runtime.

**Implementation steps:**
1. Read all README/docs/notebooks/config. Map directory structure.
2. Catalog: entry points, modules, pipelines, models/artifacts, scripts, configs, tests, dependencies, environment setup.
3. Trace the actual data flow from raw files to dashboard. Note what is implemented vs. stubbed vs. dead code.
4. Audit Flask endpoints, MongoDB collections, Streamlit views, ML artifacts, existing tests, and test coverage.
5. Run the existing baseline (setup, tests, app) and record what works and what does not. Record versions of Python and key libraries.
6. Dataset audit (Section 14.1): file counts, channels, sampling, timestamps, which sets exist locally.
7. Produce the **keep / replace / migrate / deprecate** matrix for every component (Flask, MongoDB, Streamlit, IF, PCA, RobustScaler, persistence logic, preprocessing, feature code, etc.).
8. Draft: MQTT topic/payload spec, InfluxDB schema, timestamp strategy, XGBoost target/evaluation plan, persistence redefinition, config/env plan, local infrastructure plan (how broker, InfluxDB, Grafana run locally, e.g., containerized, if the owner's machine supports it, otherwise an alternative).
9. Produce a risk register (Section 27) and a list of **open questions for the owner**.
10. Create the migration branch from a clean main (`feature/industrial-stack-migration`) and record the baseline commit hash.

**Tests:** Baseline tests run and results recorded. If no tests exist, state that.

**Expected output:** An audit report + design package + decision log, delivered for owner review.

**Failure conditions:** Unverified claims about the repo; invented filenames; missing component in the matrix; unresolved ambiguity about the XGBoost target or evaluation strategy.

**Verification:** Owner approves the design. Baseline reproduction evidence attached.

**Git commit message:** `docs(migration): add repository audit and industrial stack migration design`

**Rollback:** Delete the docs/branch; main is untouched.

**STOP after this phase and wait for approval.**

---

### PHASE 1 — MQTT Infrastructure and NASA IMS Telemetry Simulator

**Objective:** Real broker running locally; simulator publishes chronological NASA IMS snapshots.

**Files affected:** New infrastructure/config and simulator modules; configuration templates (no secrets); dependency manifest; documentation. Locations decided in Phase 0.

**Architecture impact:** Introduces the broker and the producer. Legacy path untouched.

**Implementation steps:** Stand up the broker with documented, reproducible startup. Implement the payload contract and topic scheme from Phase 0. Implement the replay engine (ordering, speed control, start/stop/resume, graceful end, reconnect with backoff, structured logging). Implement an independent **throwaway verification subscriber tool** (or use a standard MQTT CLI client) to observe messages.

**Tests:** Payload construction/validation tests; ordering test (messages arrive in timestamp order); replay-speed test; no-label-leak test (payload contains no RUL/failure hints); broker-down/reconnect behavior test; message-size test against the measured limits.

**Expected output:** Running the simulator causes messages to appear on the real broker, observable by an independent subscriber, with correct topics, timestamps, sequence numbers, and metadata.

**Failure conditions:** Simulator works without a broker (fake transport); messages reordered; data altered vs. the source files; secrets committed; hardcoded broker address.

**Verification commands:** Determined by Antigravity. Must prove (a) the broker is up, (b) N snapshots published, (c) an independent subscriber received exactly N in order, (d) payload equals source data (spot-check checksums/statistics against the original files).

**Git commit message:** `feat(mqtt): add NASA IMS telemetry publisher and local broker setup`

**Rollback:** Revert the commit; no other component depends on it yet.

---

### PHASE 2 — MQTT Consumer / Ingestion Service

**Objective:** A service that subscribes, validates, and hands telemetry to the processing pipeline, robustly.

**Files affected:** New ingestion service module(s), config, logging setup, tests.

**Architecture impact:** MachineMind now consumes from MQTT. Processing initially may be a no-op/log sink, or write nothing yet, but it must not use fake data.

**Implementation steps:** Subscribe with the agreed QoS/session settings. Validate payloads against the contract (version, fields, sample counts, timestamps). Handle duplicates, out-of-order, malformed messages, and reconnects. Expose clean internal interfaces between *ingest -> process -> store* so later phases plug in without rewrites. Graceful shutdown. Structured logs with counters (received, rejected, processed). Add a pipeline health signal that later feeds InfluxDB (Phase 3).

**Tests:** Valid/invalid payload tests; reconnect/resubscribe test; duplicate handling test; malformed-message-does-not-crash test; backpressure/slow-consumer behavior test; shutdown test.

**Expected output:** With simulator + broker running, the consumer logs receipt of every snapshot, rejects bad messages visibly, and survives broker restarts.

**Failure conditions:** Consumer crashes on bad input; silent drops; unlogged rejections; processing logic tightly coupled to the transport.

**Verification:** Run simulator -> consumer; count in = count processed + count rejected; kill/restart broker mid-stream and demonstrate recovery behavior as documented.

**Git commit message:** `feat(ingestion): add MQTT telemetry consumer with validation and reconnect handling`

**Rollback:** Revert; Phase 1 remains independently valid.

---

### PHASE 3 — InfluxDB Integration and Time-Series Schema

**Objective:** Persist consumer outputs to InfluxDB under the approved schema.

**Files affected:** New storage layer module, schema documentation, config/env templates, infrastructure definition for InfluxDB, tests.

**Architecture impact:** InfluxDB becomes the live store. MongoDB is untouched at this point (parallel run is acceptable for verification).

**Implementation steps:** Provision InfluxDB locally (reproducible; credentials/tokens via environment). Implement the writer with batching, retry, error reporting, and idempotent point design. Write telemetry-derived data and pipeline health metrics. Implement a small read/query helper only for tests and verification. Decide retention and buckets as per the schema doc.

**Tests:** Write test; query test; **timestamp round-trip verification** (stored time equals intended time semantics); schema verification (measurement/tag/field names and types match the doc); duplicate-write idempotency test; failed-write handling test; no high-cardinality tag test (e.g., check tag cardinality assumptions).

**Expected output:** After a replay, queries return the expected point counts, timestamps, and values matching the source computations.

**Failure conditions:** Wrong time semantics; duplicates on replay; tokens in code; unbounded tag cardinality; silent write failures.

**Verification:** Replay N snapshots; query InfluxDB for N points per expected measurement/channel; compare sampled values against the offline computation from the raw files.

**Git commit message:** `feat(storage): integrate InfluxDB time-series storage and schema`

**Rollback:** Revert; ingestion still works with logging sink.

---

### PHASE 4 — Signal Processing and Feature Pipeline Integration

**Objective:** One feature pipeline used identically offline (training) and online (streaming).

**Files affected:** Existing preprocessing/feature modules (refactor carefully, preserve behavior), new streaming adapter, tests.

**Architecture impact:** The consumer computes features on each snapshot (and causal history features) and stores them in InfluxDB.

**Implementation steps:**
1. Inspect existing preprocessing and feature code. Document each feature (formula, unit, why it matters for bearing diagnostics).
2. Preserve existing correct behavior (e.g., DC bias removal, time-domain features such as RMS, std/variance, kurtosis, skewness, peak, crest factor; spectral/envelope features if they exist). Add new features only with justification (e.g., band-energy features) and keep the set manageable and explainable.
3. Refactor so **one implementation** serves both the offline pipeline and the streaming consumer (no duplicate logic drifting apart).
4. Implement causal history features (if used) with explicit window semantics and warm-up policy (Section 15).
5. Write features to InfluxDB under the schema.
6. Regression test: refactored features match the legacy outputs on a fixed sample of snapshots (document any intentional differences).

**Tests:** Feature unit tests with known synthetic signals (analytic expectations, e.g., a pure sine has known RMS/crest factor; used only to test code, never as pipeline data); legacy-parity test; **streaming/offline parity test**; causality (shuffle-future) test; NaN/short-signal/handling tests; determinism test.

**Expected output:** Stored features for a replayed segment equal the offline features for the same segment within tolerance.

**Failure conditions:** Divergent offline/online implementations; non-causal windows; silent NaN propagation; changed legacy feature semantics without documentation.

**Git commit message:** `feat(features): unify offline and streaming vibration feature pipeline`

**Rollback:** Revert; legacy feature code remains available from main history.

---

### PHASE 5 — XGBoost Target Engineering and Training Pipeline

**Objective:** A scientifically defensible, leakage-free, reproducible training pipeline.

**Files affected:** New training/evaluation modules, target-construction module, experiment metadata storage, model artifact directory handling, documentation, tests.

**Architecture impact:** Offline only. Produces a versioned model artifact consumed in Phase 6.

**Implementation steps:**
1. Implement the approved target construction (Section 14) as an isolated, tested module.
2. Implement the approved split strategy with embargo/gap, or leave-one-run-out if approved (Section 14.3).
3. Fit preprocessing/scaler on training data only (Section 15).
4. Train XGBoost with time-aware hyperparameter tuning (bounded search; seeds fixed; documented search space).
5. Evaluate against baselines with the metrics in Section 14.4; produce plots and a written evaluation report that states limitations honestly.
6. Optionally run the legacy Isolation Forest/PCA comparison experiment (Section 20).
7. Package model + feature schema + preprocessor + metadata + version; verify load/predict round trip.
8. Calibrate status thresholds and persistence parameters on training/validation data only (Section 21).

**Tests:** Target generation test (correct RUL values at start/end/known points; monotonic non-increasing behavior; cap behavior); split-ordering and embargo tests; leakage tests (Section 15 item 10); training smoke test (small deterministic run); serialization round-trip; feature-schema mismatch rejection; reproducibility (same seed -> same metrics within tolerance).

**Expected output:** A model artifact, an evaluation report with metrics and plots, and a documented list of limitations.

**Failure conditions:** Random splits; test-informed tuning; time/index features leaking RUL; unrealistically perfect scores without explanation (treat suspiciously good results as a leakage alarm and investigate); no baselines.

**Git commit message:** `feat(ml): implement XGBoost degradation/RUL training pipeline with leakage-safe evaluation`

**Rollback:** Revert; model artifacts are versioned and removable.

---

### PHASE 6 — XGBoost Inference and Prediction Persistence

**Objective:** The consumer produces XGBoost predictions per snapshot and stores them with status logic.

**Files affected:** Inference module, consumer integration, state machine (persistence), storage layer additions, tests.

**Architecture impact:** The live path now includes XGBoost. MQTT -> features -> XGBoost -> InfluxDB works.

**Implementation steps:** Load the versioned model and schema at startup (fail loudly on mismatch). Run inference per snapshot once required history/warm-up is satisfied. Write predictions (predicted RUL / degradation / risk) with model version to InfluxDB. Implement the redefined persistence/status state machine (Section 21) and write state/alert transitions to InfluxDB. Log prediction latency; store it as a pipeline metric.

**Tests:** Inference unit test; schema mismatch test; warm-up behavior test; state machine tests; model-version-in-record test; end-to-end replay test of a segment verifying stored predictions equal the offline predictions on the same features; inference latency sanity check.

**Expected output:** Replaying data yields prediction and state series in InfluxDB, matching offline results.

**Failure conditions:** Predictions on incomplete features; status computed from a different quantity than documented; model version missing; state lost on restart without documentation.

**Git commit message:** `feat(inference): add streaming XGBoost inference and prediction persistence`

**Rollback:** Revert; Phases 1–5 remain valid (ingest + features still stored).

---

### PHASE 7 — Grafana Integration and Industrial Dashboards

**Objective:** Grafana, provisioned from code, visualizes real stored data.

**Files affected:** Grafana provisioning configuration (datasource, dashboards), infrastructure definition, documentation of every panel's query, tests/verification scripts.

**Architecture impact:** Grafana becomes the primary UI.

**Implementation steps:** Provision the InfluxDB datasource via environment-driven config. Build the dashboard (Section 16) with variables and units. Document each panel -> query -> measurement/field mapping. Verify timeline behavior under the chosen timestamp strategy. Decide alerting approach (Section 16). Capture screenshots as evidence for the documentation. Fill in the Streamlit -> Grafana mapping table (Section 17) with status per item.

**Tests:** Datasource health check; per-panel query execution against real data (scripted where feasible); provisioning reproducibility (destroy and recreate the stack, dashboard returns); "no fake values" review checklist; empty-state behavior (what the panel shows when there is no data).

**Expected output:** During a replay, dashboard panels update from real InfluxDB data; after a replay, history is browsable.

**Failure conditions:** Hand-built dashboard not in version control; hardcoded values; panels showing data that has no documented query; credentials in provisioning files.

**Git commit message:** `feat(grafana): add provisioned industrial monitoring dashboards`

**Rollback:** Revert; data in InfluxDB unaffected.

---

### PHASE 8 — End-to-End Integration Testing

**Objective:** Prove the full chain works together, repeatedly and from a clean start.

**Files affected:** Integration test suite, demo scripts/runbook, test documentation.

**Implementation steps:** Automate a clean-start scenario (fresh infra -> replay a defined segment -> assert on InfluxDB contents). Include failure-injection scenarios (broker restart, InfluxDB temporarily down, malformed message, consumer restart mid-stream). Measure end-to-end latency (publish -> stored prediction). Test long-run stability on the full Set 2 replay at an accelerated speed.

**Tests:** Full pipeline test: NASA snapshot -> MQTT -> consumer -> features -> XGBoost -> InfluxDB -> (Grafana query). Count/ordering/value assertions; data-gap detection; restart/resume correctness; no duplicate points after a replayed segment; resource usage sanity.

**Expected output:** Green integration suite and a recorded full-replay demonstration.

**Failure conditions:** Tests that mock the broker/DB while claiming E2E; flaky tests; unverified Grafana linkage.

**Git commit message:** `test(e2e): validate telemetry to prediction pipeline across the full stack`

**Rollback:** Tests only; revert freely.

---

### PHASE 9 — Cleanup and Deprecation of Obsolete Architecture

**Objective:** Remove or formally demote legacy components that the audit marked obsolete, with evidence it is safe.

**Implementation steps:** For each REMOVE/DEPRECATE item (per the Phase 0 matrix): confirm no remaining consumers, remove code/config/dependencies/docs references in **separate small commits**. Keep Isolation Forest/PCA only per the Section 20 decision. Confirm Streamlit/Flask/MongoDB decisions have been executed or justified. Remove unused dependencies and environment variables. Re-run the entire test suite and the E2E suite after each removal.

**Failure conditions:** Removal without verified replacement; dangling imports/config; broken docs.

**Git commit messages (examples):**
`refactor(cleanup): remove MongoDB dependency after InfluxDB migration verified`
`refactor(cleanup): retire Streamlit dashboard replaced by Grafana`
(Adjust to actual decisions.)

**Rollback:** Each removal is its own revertable commit.

---

### PHASE 10 — Documentation, Reproducibility, and Final Demonstration

**Objective:** A reviewer can understand and reproduce everything.

**Deliverables:** Updated README (setup, run, demo); architecture diagram of the final system; data-flow documentation; MQTT contract; InfluxDB schema; model card (target, features, splits, metrics, limitations); leakage-prevention document; configuration reference (all env vars, no secrets); troubleshooting; demo runbook (Section 29); decision log; limitations and future work (e.g., real sensor integration path); dependency/version pins; one-command (or few-command) local startup if feasible.

**Verification:** A clean-clone dry run following only the README succeeds. Record the final commit hash.

**Git commit message:** `docs(final): add reproducible setup, model card and demonstration runbook`

**Rollback:** Docs only.

---

## 23. Testing Requirements (summary)

| Area | Minimum tests |
|---|---|
| MQTT | Publisher test, subscriber test, payload validation, reconnect/error handling, ordering, size limits |
| InfluxDB | Write test, query test, timestamp verification, schema verification, idempotency, failure handling |
| Features | Known-signal unit tests, legacy parity, streaming/offline parity, causality, determinism |
| ML | Target generation, split ordering/embargo, leakage checks, training smoke test, inference test, serialization round trip, schema-mismatch rejection |
| State machine | Threshold, persistence, reset, warm-up, restart behavior |
| Grafana | Datasource health, per-panel query verification, provisioning reproducibility |
| End-to-end | NASA snapshot -> MQTT -> consumer -> features -> XGBoost -> InfluxDB -> Grafana demonstrated with real infra |

Rules: tests must not mock away the thing they claim to verify. Unit tests may use mocks; integration/E2E tests must use the real broker and database. Report real test output, including failures. Do not delete or weaken tests to make them pass.

---

## 24. Git Workflow

- `main` is the stable baseline. **Never modify main during the migration.**
- Create and use: `feature/industrial-stack-migration` (record the base commit hash).
- One logical change per commit; conventional-commit style, e.g.:
  - `feat(mqtt): add NASA IMS telemetry publisher`
  - `feat(ingestion): add MQTT telemetry consumer`
  - `feat(storage): integrate InfluxDB time-series storage`
  - `feat(ml): implement XGBoost degradation prediction`
  - `feat(grafana): add industrial monitoring dashboards`
  - `test(e2e): validate telemetry to prediction pipeline`
- No giant "migration" commit. No committed secrets, datasets (unless already tracked), model binaries beyond what the repo convention allows, or generated junk. Inspect `.gitignore` and update it if needed.
- Before each new phase: verify previous phase -> commit -> push -> record the commit hash in a progress log.
- Optionally tag milestone commits after phases verified.
- If a commit introduces a regression, fix forward in a new commit or revert; do not rewrite pushed history without owner consent.

---

## 25. Acceptance Criteria (Final System)

The final project demonstrates, locally and reproducibly:

```
NASA IMS snapshot -> replay/simulator -> MQTT publisher -> MQTT broker
  -> MachineMind MQTT consumer -> signal processing -> feature extraction
  -> XGBoost inference -> prediction/RUL/risk -> InfluxDB -> Grafana dashboard
```

A reviewer can answer, from the documentation and the running system:
- Where does the data originate (and that it is replayed historical data)?
- How does MQTT transport it (topics, payload, QoS)?
- Where is telemetry-derived data stored (InfluxDB schema)?
- How are features generated, and why are they the same offline and online?
- What does XGBoost predict, how is the target defined, and how was it evaluated?
- How are predictions, status, and alerts stored and visualized?
- How does the system avoid data leakage (with test evidence)?
- How would real sensor telemetry plug in later (producer replacement, payload contract)?

Additional checks:
- All four mandatory technologies are on the live path with real responsibilities.
- No secrets or hardcoded endpoints in the repo; configuration is environment-driven.
- All tests pass; E2E evidence recorded.
- Limitations are stated honestly (single-run caveat, simulated telemetry, etc.).
- Any retained legacy component has a documented, non-confusing role.

---

## 26. Antigravity Operating Rules

1. **Inspect the repository before changing anything.**
2. Read all existing documentation and architecture notes.
3. Identify existing functionality and its true status (working, partial, dead).
4. Avoid unnecessary rewrites; prefer refactors that preserve tested behavior.
5. **Never fabricate** successful infrastructure connections (broker, InfluxDB, Grafana). Connection state must reflect reality; failures must surface.
6. **Never use fake telemetry in the final pipeline.** Synthetic signals are allowed only inside unit tests with analytic expectations, clearly labeled as such.
7. Never hardcode values to make a UI look right.
8. Never silently change ML semantics; rename and document when meanings change.
9. Never remove working components without justification and verification.
10. **Ask for clarification** if a requirement is genuinely ambiguous or contradictory. Offer concrete options with trade-offs.
11. Prefer incremental implementation and small commits.
12. Run tests after every meaningful change; report real outputs.
13. Keep Git history clean and professional.
14. Explain architectural changes in plain language. The owner must be able to defend them.
15. **Record assumptions** and decisions in a decision log (decision, alternatives, rationale, date, phase).
16. Maintain reproducibility (pinned dependencies, seeds, documented environment).
17. **Do not claim a feature is complete without verification evidence.**
18. Never commit secrets; use environment variables and committed *example* config templates with placeholders.
19. Do not "fix" suspicious results by tuning until they look good. Investigate leakage first.
20. Mentorship duty: at the end of each phase, add a short "What was done / why / what to be ready to explain in the viva" note.
21. Do not trust dataset or library facts from memory. Check the dataset readme and current library documentation when details matter (broker/client versions, InfluxDB client APIs, Grafana provisioning format).
22. Treat the owner's time as limited: batch questions, and propose a default for each.

**Required report format at the end of every phase:** Objective · What changed (files) · Architecture impact · Tests run (with results) · Verification evidence · Assumptions/decisions · Commit hash · Known issues · Next phase readiness.

---

## 27. Risks and Migration Pitfalls

| Risk | Mitigation |
|---|---|
| **Single run-to-failure trajectory** limits the supervised RUL claim; end-of-life test region is out-of-distribution for tree models | Section 14.3: consider additional IMS sets with leave-one-run-out; state limits honestly |
| **Data leakage** via non-causal windows, full-run scaling, time-index features, test-informed tuning | Section 15; automated leakage tests; treat too-good metrics as alarms |
| **Mislabeled semantics** (calling RUL/risk an "anomaly score") | Section 21 renaming and documentation |
| **Large payloads** overwhelming broker/client or Influx | Measure; choose encoding/chunking; store features, not raw waveforms, by default |
| **Replay timeline mismatch** (2004 timestamps vs. Grafana "now") produces empty dashboards | Section 11 timestamp strategy decided early |
| **Duplicate points / non-idempotent writes** on replay | Schema idempotency design and tests |
| **High-cardinality tags** degrading InfluxDB | Tag/field discipline (Section 13) |
| **Offline/online feature drift** | Single implementation + parity test |
| **Two competing dashboards / two competing alert sources** | Section 16/17 decisions |
| **Scope creep** (adding auth, Kubernetes, cloud, etc.) | Stay within the four-technology mandate; log extras as future work |
| **Silent failure modes** (consumer swallowing errors) | Counters, structured logs, health metrics |
| **Version/API confusion** across InfluxDB and client library versions | Decide version once, document, pin |
| **Removing legacy too early** | Cleanup only in Phase 9 after E2E verified |
| **Overstated claims** in the report/presentation | Honesty principle; limitations section |
| **Environment/setup fragility** for a student machine | Document resource needs; provide a reproducible local setup and a troubleshooting guide |

---

## 28. Rollback Strategy

- **Baseline safety:** `main` is untouched; the owner can always return to the working baseline.
- **Phase-level rollback:** each phase is a small set of commits; revert the phase's commits to return to the previous verified state. Record each phase's commit hash for this purpose.
- **Parallel-run period:** keep the legacy path functional until Phase 9. Never delete legacy data before the new store is verified.
- **Data safety:** do not delete the MongoDB data or the original NASA files. Back up before any destructive step; the backfill is repeatable from source files.
- **Model artifacts:** versioned; keep the previous version available. Switching a model version must not require code changes beyond configuration.
- **Infrastructure state:** InfluxDB/Grafana local volumes can be recreated from provisioning plus a replay. Document how to wipe and rebuild.
- **Cleanup commits:** each removal is a separate commit so it can be reverted independently.
- **Failure policy:** if a phase fails verification, stop, report with evidence, and propose fix or rollback. Do not proceed to the next phase.

---

## 29. Final End-to-End Demonstration Procedure

Antigravity must produce a runbook with exact, tested commands (discovered from the repo) implementing the following flow. Each step states the expected observable result.

1. **Start infrastructure:** MQTT broker, InfluxDB, Grafana (reproducible, from config; secrets from environment). Expected: all three report healthy.
2. **Show configuration:** brief tour of the environment variables/config (no secrets displayed).
3. **Start the MachineMind consumer** (ingestion -> features -> XGBoost -> InfluxDB). Expected: logs show subscription established, model version loaded, schema validated.
4. **Open Grafana** and show the dashboard in its initial/empty or baseline state (honest empty state).
5. **Start the NASA IMS replay** (select machine/run, set replay speed). Expected: messages appear on the broker (optionally observed with an independent subscriber).
6. **Observe live updates:** machine status, channel trends, risk/degradation trend, predicted RUL, last-telemetry timestamp, ingestion status updating as data flows.
7. **Show a status transition** (e.g., WATCH -> alert state) later in the replay; show the alert/status history panel and the underlying InfluxDB records that justify it.
8. **Failure demo (optional but valuable):** stop/restart the broker or consumer; show recovery behavior and ingestion status reflecting it.
9. **Traceability walkthrough:** pick one dashboard value and trace it: Grafana panel -> query -> InfluxDB point -> consumer pipeline -> MQTT message -> original NASA snapshot file.
10. **Model walkthrough:** target definition, split strategy, leakage controls, evaluation metrics and plots, baselines, honest limitations.
11. **Legacy comparison (if retained):** Isolation Forest/PCA baseline versus XGBoost, clearly labeled.
12. **Replacement-readiness statement:** explain that a real sensor gateway publishing the same payload contract would replace the simulator without changing the consumer.
13. **Teardown and rebuild** from a clean state to prove reproducibility.

---

## 30. Immediate Instructions to Antigravity (Start Here)

1. Acknowledge this specification and list any ambiguities you see (propose defaults).
2. Perform **Phase 0** only: inspect the repository, run the baseline, and deliver the audit report, keep/replace/migrate/deprecate matrix, design drafts (MQTT contract, InfluxDB schema, timestamp strategy, XGBoost target/evaluation plan, persistence redefinition, infrastructure plan), risk register, and open questions.
3. Create the branch `feature/industrial-stack-migration` and record the baseline commit hash.
4. **Stop and wait for the owner's approval** before starting Phase 1.

No production code is to be written before the Phase 0 deliverable is approved.
