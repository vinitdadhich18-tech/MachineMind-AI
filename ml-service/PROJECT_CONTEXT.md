# PROJECT_CONTEXT.md — MachineMind AI

> Permanent project reference. Reuse this file as context in Claude and Antigravity.
> Created: 2026-09-25. Update it only when a decision genuinely changes (see Section 14).

**How to read this file:** every fact is labeled **CONFIRMED** (stated by the project owner or checked against an official source), **REPORTED** (found in secondary sources, must be verified in Phase 1), or **PROVISIONAL** (a decision that may change).

---

## 1. Project name and objective

- **Name:** MachineMind AI (CONFIRMED)
- **Objective:** Build a vibration-based condition-monitoring prototype for rotating machinery using machine learning. (CONFIRMED)
- **Owner profile:** A Computer Science student learning ML. The goal is to *understand* the code, the ML concepts and the evaluation process, not to accept generated code blindly. (CONFIRMED)

## 2. Scope and intended functionality

- Initial machinery: **rolling-element bearings**. (CONFIRMED)
- The prototype learns normal vibration behavior from an assumed healthy period and flags later measurements that deviate significantly from it. (CONFIRMED)
- This is a **research/learning prototype** built on laboratory data. It is not an industrial monitoring product. (CONFIRMED)
- The ML work lives in `ml-service/`. A backend or frontend may be added *after* the ML work is complete, and is out of scope now. (PROVISIONAL)

## 3. Dataset and initial subset

- **Dataset:** NASA IMS Bearing Dataset (University of Cincinnati Center for Intelligent Maintenance Systems, hosted in the NASA Prognostics Data Repository). (PROVISIONAL, chosen after a comparison of CWRU, Paderborn, IMS, MFPT and C-MAPSS)
- **Initial subset:** **Set 2**. (PROVISIONAL, subject to verification of documentation, data availability and suitability in Phases 1-2)
- **Official source (CONFIRMED to exist):** NASA Prognostics Data Repository, entry "Bearings": https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ (landing page: https://data.nasa.gov/dataset/ims-bearings)
- **Citation requested by NASA (CONFIRMED from the repository page):** J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. "Bearing Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA. NASA also asks users to acknowledge the repository and the data donors, and states that data is used at the user's own risk.
- **Licence: UNRESOLVED.** Repository metadata is inconsistent (one record links to the US government works page, a mirror lists "other-license-specified"). Do not publish derived data until this is clarified. Never commit raw data to Git.

**Details REPORTED by secondary sources (not yet verified against the official README; Phase 1 must verify each):**

| Item | Reported value |
|---|---|
| Test rig | 4 bearings on one shaft, ~2000 rpm, ~6000 lb radial load |
| Experiment type | Test-to-failure runs (3 sets) |
| Snapshot | 1 second, 20,480 points, 20 kHz sampling |
| File format | ASCII text, one file per snapshot, timestamp encoded in the file name |
| Recording interval | Every 10 minutes; larger timestamp gaps mean the experiment resumed the next working day |
| Set 2 size | 984 files, 4 channels (one per bearing) |
| Set 2 dates | About 12 Feb 2004 to 19 Feb 2004 |
| Set 2 outcome | Outer-race failure in bearing 1 |

**NOT known / unverified:** sensor units and calibration, file delimiter, exact first and last timestamps, bearing geometry, whether bearings 2-4 stayed healthy for the whole run, and whether any Set 2 files are damaged.

**Known concern:** one independent audit found that a downloaded Set 3 directory did not match its README (file count and end date). **Set 3 is excluded from the initial scope.**

## 4. Initial ML task

- **Task:** Unsupervised anomaly detection. (CONFIRMED as the intended first task; the dataset it runs on is still PROVISIONAL)
- **Goal:** Learn normal vibration behavior from an **assumed** healthy period, then identify unusual deviations in later measurements.
- **Key point:** the dataset has **no per-file healthy/faulty labels**. "Healthy" is an assumption made by us, and must always be described that way.

## 5. Expected model input and output (PROVISIONAL design)

- **Input:** one 1-second vibration snapshot from one bearing channel, reduced to a small feature vector (time-domain features first: RMS, standard deviation, kurtosis, skewness, crest factor; frequency-domain features only if justified by the data).
- **Output:** (a) a continuous **anomaly score** and (b) an **alert** when the score stays above a threshold learned from the baseline (a persistence rule may require several consecutive exceedances).
- **The model must NOT claim** to predict the time of failure or the remaining useful life (RUL).

## 6. Current technical environment (CONFIRMED)

- OS: Windows
- IDE: Antigravity
- Python: 3.14.2, virtual environment `venv` (already created and verified; do not recreate it)
- Installed libraries: NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter
- Ask before installing anything new. Python 3.14 is recent, so compatibility of any additional package is **unverified**.

## 7. Existing project structure (CONFIRMED as provided by the owner)

```
MachineMind AI/
│
├── ml-service/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── notebooks/
│   ├── models/
│   ├── reports/
│   ├── src/
│   ├── venv/
│   ├── requirements.txt
│   └── README.md
│
└── .gitignore
```

An earlier note used the folder name `MachineMind-AI`. The agent must inspect the real workspace and use the real names.

## 8. Current project status

- Environment setup: **complete** (CONFIRMED).
- Dataset research and comparison: done in conversation; the dataset choice is **provisional**.
- No dataset has been downloaded, no ML code written, no model trained. (CONFIRMED as of 2026-09-25)
- ML development phases 1-11: **all Not Started.** The live status is tracked in `ML_PROGRESS.md`.

## 9. Important scientific limitations

1. **No ground-truth health labels.** The healthy period is an assumption. False alarms can only be judged relative to that assumption.
2. **Very few failure events.** Set 2 documents one failure (bearing 1), so any evaluation is a single-case study, not a statistical validation.
3. **Single rig and operating condition** (fixed speed and load), in a laboratory. No claim of industrial reliability may be made.
4. **Bearings share one shaft**, so the four channels are not independent samples.
5. **Detection timing is measured against the end of the recording,** which is a proxy for failure time, not an exact failure time.
6. **Anomaly is not the same as fault.** An alert means "different from the baseline", not "this specific defect is present".
7. **Unknown units and calibration** may limit interpretation of absolute amplitudes.
8. This is anomaly detection / early warning, not failure-time prediction and not RUL prediction.

## 10. Development workflow

1. Work on **one phase at a time**, using the prompts in `MachineMind_AI_ML_Prompt_Pack.md`.
2. Each phase: inspect the workspace, present a plan, get approval, implement in small steps, explain the concepts, verify, summarize.
3. The owner verifies the results personally, then updates `ML_PROGRESS.md`. A phase is only "Completed" after verification.
4. Do not start a phase before the previous one is verified.
5. If a required fact is unknown, stop and ask instead of guessing.
6. **Git Commit Reminder Workflow:** After a phase is verified and `ML_PROGRESS.md` is updated, the agent must inspect `git status`, verify `.gitignore` exclusions, explain the changed files, suggest a descriptive commit message, and provide exact `git add` and `git commit` commands for the owner to execute manually.

## 11. Coding and evaluation rules

**Coding**
- Reusable logic goes in `src/` as small, documented functions. Notebooks are for exploration and explanation.
- Use relative paths (for example via `pathlib`), never hard-coded absolute Windows paths.
- Fix random seeds and record configurations and library versions.
- Raw data in `data/raw/` is never modified, renamed or deleted.
- Never delete or overwrite an existing file without explicit approval.
- Ask before installing libraries or running commands that change the system.

**Git & Version Control**
- Always verify `.gitignore` before suggesting a commit (ensure raw datasets, `venv`, `.env`, binary data, and model binaries are excluded).
- Check `git status` before proposing a commit; if no meaningful changes exist, do not suggest a commit.
- Provide clear explanations of changed files and suggest descriptive commit messages.
- **Never execute `git commit` or `git push` automatically.** Provide the exact commands for the project owner to run manually.

**Evaluation**
- Preserve chronological order; never shuffle time-series data or use random splits.
- Fit scalers, statistics, thresholds and models **only** on permitted training data. Choose thresholds on a validation period, never on the final evaluation period.
- Evaluate per bearing/channel (group-based).
- Do not invent labels, metrics or results. If proxy labels are ever used, define them in advance and report them as proxies.
- Report false-alarm behavior and detection timing, and state the limitations next to every result.

## 12. Included in the initial version

- Dataset documentation and verification, EDA, signal preprocessing, feature engineering
- A statistical baseline and 1-3 justified unsupervised scikit-learn models
- A leakage-aware chronological evaluation and controlled improvement experiments
- Saved model artifacts with metadata and a reproducible inference module
- Documentation of limitations, intended use and an interface description for future integration

## 13. Explicitly excluded from the initial version

- Predicting failure time or remaining useful life
- Supervised fault classification
- Deep learning models
- Other datasets or IMS Sets 1 and 3 (unless approved later after their own documentation check)
- Backend, API, frontend, dashboard, deployment, real-time streaming, notification systems
- Any claim of industrial reliability

## 14. Conditions under which decisions may change

Revisit the provisional decisions if any of these happens (record it in `ML_PROGRESS.md` under Important decisions, then update this file):

- Set 2 documentation is contradictory or missing, or the data is unavailable or corrupted.
- The licence or usage terms prevent the intended use.
- No defensible healthy period can be justified from the data.
- Python 3.14 blocks a needed package and there is no reasonable workaround.
- Results show a single failure run is too weak, so an additional dataset (for example another IMS set, or a dataset with more failures) should be considered.
- The supervisor or course requirements change the goal.
