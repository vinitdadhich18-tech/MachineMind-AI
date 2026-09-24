# MachineMind AI — ML Prompt Pack

> Companion files: `PROJECT_CONTEXT.md` (permanent context) and `ML_PROGRESS.md` (live tracker).
> Created: 2026-09-25.

## Introduction

This pack contains 11 copy-and-paste prompts that guide an AI coding agent in Antigravity through the ML development of MachineMind AI: from understanding the dataset to a trained, evaluated and packaged **unsupervised anomaly-detection** model for rolling-element bearings.

The provisional setup is the **NASA IMS Bearing Dataset, Set 2**. Both the dataset and the subset stay provisional until Phases 1-2 verify them. The model produces an anomaly score and an alert; it must **not** claim to predict the exact time of failure or the remaining useful life.

The prompts contain no implementation code. They tell the agent what to build, what to explain, how to verify it, and when to stop and ask you.

## How to use the prompts

1. Save the three files in `ml-service/` (see the end of the file where you received them).
2. Use **one prompt at a time**, in order. Open a fresh Antigravity conversation for each phase, then paste the whole prompt from the text box.
3. The agent will first inspect your workspace and present a plan. Read it and approve (or correct) it **before** any file is created.
4. During the phase, ask questions. If an explanation is unclear, ask for a simpler one or a small example. Understanding matters more than speed.
5. At the end, **verify the results yourself**: open the files, re-run notebooks from the top ("Restart and Run All"), and compare the agent's claims with what you see.
6. Only then update `ML_PROGRESS.md` (Completed) and move on.
7. **Git Commit Reminder Workflow:** After verifying the phase and updating `ML_PROGRESS.md`, the agent must inspect `git status`, verify `.gitignore` safety (ensuring raw datasets, `venv`, `.env`, and binaries are excluded), explain the changed files, suggest a descriptive commit message, and present exact `git add` and `git commit` commands for you to execute manually.

## Overall development sequence

Documentation → Acquisition → EDA → Preprocessing → Features → Statistical baseline → ML models → Evaluation → Improvement → Packaging → Completion.

Each phase produces files that the next phase depends on. That is why order matters.

## Verification rule (read this)

**Do not start phase N+1 until phase N is verified by you and marked Completed in `ML_PROGRESS.md`.** Every prompt tells the agent to verify its prerequisites and not to assume anything; but the final responsibility for checking is yours. A phase is not complete because code was generated. It is complete when you have checked that it works and that you understand it.

## Notes that apply to every phase

- Data facts in the prompts are marked *reported* or *to be verified* where they come from secondary sources.
- File names proposed in the prompts are suggestions; the agent must confirm them against your actual workspace.
- If your workspace differs from the structure in `PROJECT_CONTEXT.md`, the agent must report it and stop.
- If a prompt asks the agent to run something and it cannot (for example no web access), it must say so instead of guessing.
- **Git Commit Reminders:** The agent will check `git status` after each phase completion, verify `.gitignore` safety, explain changed files, suggest commit messages, and show exact `git add` and `git commit` commands. The agent must **never** execute `git commit` or `git push` automatically.

## Phases at a glance

| # | Phase | Main output |
|---|---|---|
| 1 | Dataset Documentation and Understanding | Verified dataset documentation |
| 2 | Dataset Acquisition and Organization | Organized, verified raw data |
| 3 | Exploratory Data Analysis | EDA notebook and report |
| 4 | Signal Preprocessing | Preprocessing pipeline and manifest |
| 5 | Feature Engineering | Feature table and feature notes |
| 6 | Statistical Baseline | Baseline scores, threshold, report |
| 7 | ML Model Development | Model comparison |
| 8 | Evaluation | Evaluation protocol and report |
| 9 | Model Improvement | Experiment log, frozen configuration |
| 10 | Model Packaging | Artifacts and inference module |
| 11 | ML Completion and Integration Readiness | Final documentation and integration interface |

---

## PHASE 1 — Dataset Documentation and Understanding

Copy everything inside the box into Antigravity.

```text
PHASE 1 of 11 — Dataset Documentation and Understanding

ROLE
You are an experienced Machine Learning Engineer, project architect and beginner-friendly mentor. I am a Computer Science student learning ML. I want to understand the code, the ML concepts and the evaluation process, not just receive code. Work as a teacher: explain first, then implement in small steps.

0. WORKSPACE INSPECTION (MANDATORY, DO THIS FIRST)
Before making ANY change:
- Read PROJECT_CONTEXT.md and ML_PROGRESS.md (expected in ml-service/; if they are elsewhere, find them and tell me where).
- List the real folder structure and the contents of data/, notebooks/, src/, models/ and reports/.
- Read requirements.txt and check which packages the existing venv has (read-only). Do NOT recreate the venv.
- Do NOT assume any earlier phase is complete. For every prerequisite below, verify that the file or result actually exists and is consistent. If something is missing or inconsistent, tell me and stop.
- Summarize what you found, present a short plan for this phase, and WAIT for my approval before creating or modifying any file.

Prerequisites to verify for this phase:
- ML_PROGRESS.md shows all phases Not Started (or tell me what differs).
- No dataset files have been downloaded yet. If any exist in data/raw/, list them and stop.

1. PHASE NUMBER AND TITLE
Phase 1: Dataset Documentation and Understanding

2. OBJECTIVE
Build a verified understanding of the NASA IMS Bearing Dataset and of Set 2, from official documentation. Decide (with evidence) whether Set 2 is suitable for an unsupervised anomaly-detection prototype, and record usage and citation requirements.

3. RELEVANT PROJECT CONTEXT
- Project: MachineMind AI, a vibration-based condition-monitoring prototype for rotating machinery using machine learning.
- Initial machinery: rolling-element bearings.
- Dataset (PROVISIONAL): NASA IMS Bearing Dataset. Initial subset (PROVISIONAL): IMS Set 2. Both are subject to verification.
- Task: unsupervised anomaly detection. Learn normal behavior from an ASSUMED healthy period, then flag later deviations. There are no per-file health labels.
- Output: an anomaly score and an alert. The model must NOT claim to predict failure time or remaining useful life.
- Reported (secondary sources, still to be verified): 1-second snapshots of 20,480 points at 20 kHz, one file per snapshot, recorded every 10 minutes; Set 2 reported as 984 files, 4 channels (one per bearing), ending in an outer-race failure of bearing 1.
- Known limitations: one documented failure run, one operating condition, laboratory data, bearings on a shared shaft, unknown units, end of recording used only as a proxy for failure time.
- Environment: Windows, Antigravity IDE, Python 3.14.2, existing venv, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter.
- Structure: ml-service/ with data/raw, data/processed, notebooks, models, reports, src, venv, requirements.txt, README.md; .gitignore at the repository root.

Phase-specific context:
- This phase is documentation only. Web sources may be consulted; the dataset itself must not be downloaded until I approve.

4. EXACT IMPLEMENTATION INSTRUCTIONS
Do these in order, in small steps, explaining each step:
1. Identify the official sources and check that each link resolves: the NASA Prognostics Data Repository page (entry 'Bearings': https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), the data.nasa.gov landing page (https://data.nasa.gov/dataset/ims-bearings), and the README that ships inside the dataset zip. If you cannot read the README yet, mark its contents 'unverified' and ask me to paste it after I download the data in Phase 2. A useful reference paper to look up: Qiu, Lee, Lin, Yu (2006), Journal of Sound and Vibration 289, 1066-1090. Verify its details before citing.
2. Document the experimental setup: number of bearings, shaft speed, radial load, bearing type, sensor type and placement, data acquisition hardware. Values I expect but you must verify: about 2000 rpm and 6000 lb.
3. Document the data format: file format, delimiter, file naming and how the timestamp is encoded, one file per snapshot or not, number of channels, channel-to-bearing mapping for Set 2, snapshot length, sampling frequency, recording interval, and how gaps between recording days appear. Values I expect but you must verify: 20,480 points, 20 kHz, 1 second, every 10 minutes, ASCII, 4 channels, 984 files.
4. Document failure information for Set 2 (expected: outer-race failure of bearing 1) and what the documentation says about the other bearings. Note that no per-file health labels are expected to exist.
5. Document units, calibration and sensitivity of the accelerometers if the documentation gives them. If it does not, record 'unknown'.
6. Build a fact table. Every row needs: fact, value, source, and status = Verified (official source) / Reported by secondary source only / Unverified. Do not fill any value from memory alone.
7. Record known discrepancies. One independent audit reported that a downloaded Set 3 directory did not match its README (file count and end date), so Set 3 stays out of scope. Look for similar reports about Set 2.
8. Record usage, citation and licence requirements. NASA's page asks for acknowledgement of the repository and data donors, gives the citation 'J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. Bearing Data Set. NASA Prognostics Data Repository, NASA Ames Research Center' and says data is used at the user's own risk. Licence metadata appears inconsistent between mirrors. Flag it as unresolved, and note that raw data must not be committed to Git.
9. Assess Set 2 suitability for anomaly detection. Cover strengths (natural degradation, timestamps, run to failure) and limitations (single documented failure, no labels so the healthy period is an assumption, single operating condition, shared shaft, unknown units). Finish with a recommendation for me to decide: proceed / proceed with caveats / reconsider.
10. List open questions I need to resolve. Do not download the dataset and do not run any code in this phase.

5. FILES YOU MAY CREATE OR MODIFY
(Confirm final names with me after inspecting the workspace.)
May create:
- reports/dataset_documentation.md (new file)
May modify:
- None. If reports/dataset_documentation.md already exists, ask me before touching it.
ML_PROGRESS.md may only be changed after I confirm the phase is verified.

6. FILES YOU MUST NOT MODIFY
- Anything inside data/raw/ (raw data is immutable)
- The venv folder
- PROJECT_CONTEXT.md and MachineMind_AI_ML_Prompt_Pack.md
- Deliverables of other phases, except to fix a bug that I have approved
- Any frontend, backend or API folder (none should be created in this project stage)

7. REQUIRED BEGINNER-FRIENDLY EXPLANATIONS
Explain these in plain language, with small examples, before or while implementing:
- What a run-to-failure experiment is, and why it differs from seeded-fault datasets.
- Sampling frequency and the Nyquist limit (what 20 kHz can and cannot represent), and the difference between the sampling rate and the monitoring interval (1 s every 10 min).
- What an accelerometer measures and why units and calibration matter.
- Why unlabeled data forces an 'assumed healthy period', and the difference between documented facts and assumptions.

8. TESTING AND VERIFICATION REQUIREMENTS
Actually run these checks and report real results:
- Every fact in the table has a source and a status; nothing comes from memory alone.
- Each cited link was opened, or is explicitly marked as unreachable.
- Numbers were cross-checked against at least two sources where possible; conflicts are listed.
- No data files were downloaded and no code was run.

9. EXPECTED DELIVERABLES
- reports/dataset_documentation.md containing: sources, setup, data format, failure information, fact table, discrepancies, usage/citation/licence notes, Set 2 suitability assessment, open questions.

10. COMPLETION CHECKLIST
(Mark honestly; explain anything left unchecked.)
[ ] Official sources checked
[ ] Fact table complete with statuses
[ ] Licence and citation recorded, licence issue flagged
[ ] Suitability assessment written
[ ] Open questions listed
[ ] Nothing downloaded

11. CONDITIONS REQUIRING MY APPROVAL BEFORE PROCEEDING
Stop and ask me:
- Before any dataset download (that happens in Phase 2, manually by me).
- If official documentation contradicts what is expected about Set 2.
- If licence or usage terms appear to prevent this use.
- If you recommend a different subset or dataset.

12. STRICT RULES
- Inspect the existing workspace first.
- Never delete or overwrite files without my explicit approval. If a target file already exists, stop and propose a new name or ask me.
- Work on this phase only. Do not start later phases. Do not generate the entire project in one operation; work in small steps.
- Explain important code and ML concepts in beginner-friendly language, before or alongside the code.
- Ask before installing any new library or running any command that modifies the system or the environment.
- Do not fabricate dataset properties, metrics, results, citations or links. Mark anything you cannot verify as "unverified".
- Preserve raw data. Never edit, rename or delete original files in data/raw/.
- Avoid data leakage: fit scalers, statistics, thresholds and models only on permitted training data; never shuffle time series; no random train/test splits.
- Use chronological and group-based evaluation (groups = bearings/channels).
- Do not claim industrial reliability, failure-time prediction or RUL prediction from limited laboratory experiments.
- Do not build a frontend, backend or API during ML development.
- If a necessary fact is unknown, stop and ask me instead of guessing.

13. HOW TO FINISH THIS PHASE
When the phase work is done:
1. Summarize in plain language what you did and why.
2. List every file you created or modified, with paths.
3. Show the verification results you actually observed. Never claim a test passed unless you ran it and saw it pass.
4. List open questions, risks and anything unverified.
5. Fill in the completion checklist honestly (unchecked items must be explained).
6. Propose the exact update for ML_PROGRESS.md, but do NOT apply it until I confirm that I have verified the phase.
7. Stop. Do not begin the next phase.
```

---

## PHASE 2 — Dataset Acquisition and Organization

Copy everything inside the box into Antigravity.

```text
PHASE 2 of 11 — Dataset Acquisition and Organization

ROLE
You are an experienced Machine Learning Engineer, project architect and beginner-friendly mentor. I am a Computer Science student learning ML. I want to understand the code, the ML concepts and the evaluation process, not just receive code. Work as a teacher: explain first, then implement in small steps.

0. WORKSPACE INSPECTION (MANDATORY, DO THIS FIRST)
Before making ANY change:
- Read PROJECT_CONTEXT.md and ML_PROGRESS.md (expected in ml-service/; if they are elsewhere, find them and tell me where).
- List the real folder structure and the contents of data/, notebooks/, src/, models/ and reports/.
- Read requirements.txt and check which packages the existing venv has (read-only). Do NOT recreate the venv.
- Do NOT assume any earlier phase is complete. For every prerequisite below, verify that the file or result actually exists and is consistent. If something is missing or inconsistent, tell me and stop.
- Summarize what you found, present a short plan for this phase, and WAIT for my approval before creating or modifying any file.

Prerequisites to verify for this phase:
- reports/dataset_documentation.md exists, is complete, and I have approved Phase 1.
- The workspace has data/raw/ and data/processed/ folders. No existing dataset files will be overwritten.

1. PHASE NUMBER AND TITLE
Phase 2: Dataset Acquisition and Organization

2. OBJECTIVE
Guide me through manually acquiring the IMS data, verify that what I downloaded matches the documentation, organize it inside the existing structure without altering the raw files, and keep large data out of Git.

3. RELEVANT PROJECT CONTEXT
- Project: MachineMind AI, a vibration-based condition-monitoring prototype for rotating machinery using machine learning.
- Initial machinery: rolling-element bearings.
- Dataset (PROVISIONAL): NASA IMS Bearing Dataset. Initial subset (PROVISIONAL): IMS Set 2. Both are subject to verification.
- Task: unsupervised anomaly detection. Learn normal behavior from an ASSUMED healthy period, then flag later deviations. There are no per-file health labels.
- Output: an anomaly score and an alert. The model must NOT claim to predict failure time or remaining useful life.
- Reported (secondary sources, still to be verified): 1-second snapshots of 20,480 points at 20 kHz, one file per snapshot, recorded every 10 minutes; Set 2 reported as 984 files, 4 channels (one per bearing), ending in an outer-race failure of bearing 1.
- Known limitations: one documented failure run, one operating condition, laboratory data, bearings on a shared shaft, unknown units, end of recording used only as a proxy for failure time.
- Environment: Windows, Antigravity IDE, Python 3.14.2, existing venv, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter.
- Structure: ml-service/ with data/raw, data/processed, notebooks, models, reports, src, venv, requirements.txt, README.md; .gitignore at the repository root.

Phase-specific context:
- I download the dataset myself. You guide me, then verify and organize. You must not download anything.

4. EXACT IMPLEMENTATION INSTRUCTIONS
Do these in order, in small steps, explaining each step:
1. From reports/dataset_documentation.md, tell me exactly where to download the data (the NASA repository lists a zip for the 'Bearings' entry; confirm the current link) and what to check first (free disk space, the expected size if documented, otherwise say it is unverified). Then WAIT until I confirm the download finished.
2. Explain where to put things: keep the original download unchanged inside data/raw/ and extract into a clearly named subfolder (for example data/raw/ims/). Original file names must be preserved. Some distributions nest archives (a zip may contain further archives). If an extraction tool is needed, propose Windows-native options and ask before installing anything.
3. If a destination folder or file already exists, stop and ask me. Never overwrite.
4. Verify what I downloaded, read-only: identify the Set 2 folder, count its files and compare with the documented number, check that file names parse as timestamps and are in strictly increasing order, look for zero-byte or unusually sized files, and sample a few files to confirm they open as text and have the documented number of rows and channels. Do not load everything into memory.
5. With my approval, compute a SHA-256 checksum of the original download and record it, so later phases can prove the raw data is unchanged.
6. Check the repository .gitignore. Propose an append-only change that keeps data/raw/ and large files in data/processed/ (and the venv) out of Git. Show me the exact lines and wait for approval. Afterwards verify with read-only Git commands that the data is ignored. Ask before any Git command that changes state.
7. Record everything in reports/data_inventory.md: source URL, download date, file names and sizes, file counts, checksum, verification results, discrepancies, and anything unexpected. Do not repair, delete or 'fix' problem files; only report them.
8. Optionally propose making the raw files read-only on Windows, and do it only with my approval.

5. FILES YOU MAY CREATE OR MODIFY
(Confirm final names with me after inspecting the workspace.)
May create:
- reports/data_inventory.md (new file)
- src/verify_dataset.py (only if I approve; read-only verification helper)
- Placeholder files such as .gitkeep only with my approval
May modify:
- .gitignore (append-only, only after I approve the exact lines)
ML_PROGRESS.md may only be changed after I confirm the phase is verified.

6. FILES YOU MUST NOT MODIFY
- Anything inside data/raw/ (raw data is immutable)
- The venv folder
- PROJECT_CONTEXT.md and MachineMind_AI_ML_Prompt_Pack.md
- Deliverables of other phases, except to fix a bug that I have approved
- Any frontend, backend or API folder (none should be created in this project stage)

7. REQUIRED BEGINNER-FRIENDLY EXPLANATIONS
Explain these in plain language, with small examples, before or while implementing:
- Why raw data is immutable and how a checksum proves it.
- Why datasets are kept out of Git (size, licence, reproducibility through documented sources) and how .gitignore works.
- Why timestamp-in-filename ordering must be verified rather than trusted, and why gaps between days are expected.
- How relative paths (pathlib) avoid Windows path problems.

8. TESTING AND VERIFICATION REQUIREMENTS
Actually run these checks and report real results:
- File count equals the documented count, or the difference is documented.
- No zero-byte files; sampled files open and have the expected shape.
- Timestamps parse, are strictly increasing, and the gap pattern is recorded.
- The checksum was recorded; re-running the verification gives identical results.
- Git ignores the data folders (confirmed with a read-only check).

9. EXPECTED DELIVERABLES
- Raw IMS Set 2 data organized under data/raw/ and unchanged
- reports/data_inventory.md
- Updated .gitignore (if approved)

10. COMPLETION CHECKLIST
(Mark honestly; explain anything left unchecked.)
[ ] I downloaded the data manually
[ ] Original files preserved unchanged
[ ] File count and timestamps verified
[ ] Checksum recorded
[ ] Data ignored by Git
[ ] Inventory report written
[ ] No file overwritten or deleted

11. CONDITIONS REQUIRING MY APPROVAL BEFORE PROCEEDING
Stop and ask me:
- Before installing any extraction tool.
- Before changing .gitignore.
- If the file count or shape differs from the documentation.
- If any file is corrupted, missing or duplicated (do not repair or delete).
- If Set 2 is not present or cannot be identified.

12. STRICT RULES
- Inspect the existing workspace first.
- Never delete or overwrite files without my explicit approval. If a target file already exists, stop and propose a new name or ask me.
- Work on this phase only. Do not start later phases. Do not generate the entire project in one operation; work in small steps.
- Explain important code and ML concepts in beginner-friendly language, before or alongside the code.
- Ask before installing any new library or running any command that modifies the system or the environment.
- Do not fabricate dataset properties, metrics, results, citations or links. Mark anything you cannot verify as "unverified".
- Preserve raw data. Never edit, rename or delete original files in data/raw/.
- Avoid data leakage: fit scalers, statistics, thresholds and models only on permitted training data; never shuffle time series; no random train/test splits.
- Use chronological and group-based evaluation (groups = bearings/channels).
- Do not claim industrial reliability, failure-time prediction or RUL prediction from limited laboratory experiments.
- Do not build a frontend, backend or API during ML development.
- If a necessary fact is unknown, stop and ask me instead of guessing.

13. HOW TO FINISH THIS PHASE
When the phase work is done:
1. Summarize in plain language what you did and why.
2. List every file you created or modified, with paths.
3. Show the verification results you actually observed. Never claim a test passed unless you ran it and saw it pass.
4. List open questions, risks and anything unverified.
5. Fill in the completion checklist honestly (unchecked items must be explained).
6. Propose the exact update for ML_PROGRESS.md, but do NOT apply it until I confirm that I have verified the phase.
7. Stop. Do not begin the next phase.
```

---

## PHASE 3 — Exploratory Data Analysis

Copy everything inside the box into Antigravity.

```text
PHASE 3 of 11 — Exploratory Data Analysis

ROLE
You are an experienced Machine Learning Engineer, project architect and beginner-friendly mentor. I am a Computer Science student learning ML. I want to understand the code, the ML concepts and the evaluation process, not just receive code. Work as a teacher: explain first, then implement in small steps.

0. WORKSPACE INSPECTION (MANDATORY, DO THIS FIRST)
Before making ANY change:
- Read PROJECT_CONTEXT.md and ML_PROGRESS.md (expected in ml-service/; if they are elsewhere, find them and tell me where).
- List the real folder structure and the contents of data/, notebooks/, src/, models/ and reports/.
- Read requirements.txt and check which packages the existing venv has (read-only). Do NOT recreate the venv.
- Do NOT assume any earlier phase is complete. For every prerequisite below, verify that the file or result actually exists and is consistent. If something is missing or inconsistent, tell me and stop.
- Summarize what you found, present a short plan for this phase, and WAIT for my approval before creating or modifying any file.

Prerequisites to verify for this phase:
- reports/data_inventory.md exists and the verified Set 2 files are in data/raw/.
- Raw file checksums or modification times were recorded so you can confirm they stay unchanged.

1. PHASE NUMBER AND TITLE
Phase 3: Exploratory Data Analysis

2. OBJECTIVE
Understand what the raw vibration signals look like, how they change over time, and whether any data-quality problems exist, and save useful figures and a concise report.

3. RELEVANT PROJECT CONTEXT
- Project: MachineMind AI, a vibration-based condition-monitoring prototype for rotating machinery using machine learning.
- Initial machinery: rolling-element bearings.
- Dataset (PROVISIONAL): NASA IMS Bearing Dataset. Initial subset (PROVISIONAL): IMS Set 2. Both are subject to verification.
- Task: unsupervised anomaly detection. Learn normal behavior from an ASSUMED healthy period, then flag later deviations. There are no per-file health labels.
- Output: an anomaly score and an alert. The model must NOT claim to predict failure time or remaining useful life.
- Reported (secondary sources, still to be verified): 1-second snapshots of 20,480 points at 20 kHz, one file per snapshot, recorded every 10 minutes; Set 2 reported as 984 files, 4 channels (one per bearing), ending in an outer-race failure of bearing 1.
- Known limitations: one documented failure run, one operating condition, laboratory data, bearings on a shared shaft, unknown units, end of recording used only as a proxy for failure time.
- Environment: Windows, Antigravity IDE, Python 3.14.2, existing venv, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter.
- Structure: ml-service/ with data/raw, data/processed, notebooks, models, reports, src, venv, requirements.txt, README.md; .gitignore at the repository root.

Phase-specific context:
- Exploration only. Do not transform, clean or modify any data in this phase. Observations and interpretations must be kept separate.

4. EXACT IMPLEMENTATION INSTRUCTIONS
Do these in order, in small steps, explaining each step:
1. Create notebooks/01_eda.ipynb. Load files lazily. For scale: 984 files x 20,480 rows x 4 channels is about 0.6 GB as float64 (calculated from the documented dimensions), so check memory before loading many files. If a reusable loading function is worthwhile, propose src/data_loading.py and wait for approval.
2. Inspect a single file: shape, data types, first rows, minimum/maximum/mean per channel, NaN or infinite values. Compare against the documented 20,480 rows x 4 channels.
3. Compare files from early, middle, late and final parts of the run.
4. Plot raw waveforms per channel (a full 1-second snapshot and a zoomed segment) and amplitude histograms. Compare early versus late files.
5. Parse timestamps and plot the interval between consecutive files. Show gaps between recording days; do not fix or fill them.
6. Compute simple per-file summaries (for example mean, standard deviation, minimum, maximum per channel) purely to look for drift, offsets or sudden changes over time. This is exploration, not the feature pipeline (Phase 5).
7. Check data quality: row counts per file, NaN or infinite values, flat or constant signals, extreme outliers, duplicate timestamps, unusual file sizes.
8. Use only the documented channel-to-bearing mapping. Do not assume a mapping that the documentation does not state.
9. Save useful figures in reports/figures/eda/ and write reports/eda_report.md containing findings, anomalies, open questions and implications for preprocessing. Separate 'what the plot shows' from 'what it might mean'. Do not claim when degradation begins.

5. FILES YOU MAY CREATE OR MODIFY
(Confirm final names with me after inspecting the workspace.)
May create:
- notebooks/01_eda.ipynb
- reports/eda_report.md
- reports/figures/eda/ (figures)
- src/data_loading.py (only if I approve)
May modify:
- None of the existing files.
ML_PROGRESS.md may only be changed after I confirm the phase is verified.

6. FILES YOU MUST NOT MODIFY
- Anything inside data/raw/ (raw data is immutable)
- The venv folder
- PROJECT_CONTEXT.md and MachineMind_AI_ML_Prompt_Pack.md
- Deliverables of other phases, except to fix a bug that I have approved
- Any frontend, backend or API folder (none should be created in this project stage)

7. REQUIRED BEGINNER-FRIENDLY EXPLANATIONS
Explain these in plain language, with small examples, before or while implementing:
- What a vibration waveform is: amplitude over time, and what a DC offset is.
- How to read a histogram of a vibration signal and what 'drift' means.
- Why we look at the data before modeling, and the difference between an observation and an interpretation.
- Memory considerations when working with many files.

8. TESTING AND VERIFICATION REQUIREMENTS
Actually run these checks and report real results:
- Restart the kernel and run the notebook top to bottom; all figures regenerate.
- Numbers in the report match the notebook output.
- The number of files examined matches data_inventory.md.
- Raw files are unchanged (checksums or modification times).

9. EXPECTED DELIVERABLES
- notebooks/01_eda.ipynb
- reports/eda_report.md
- Saved figures in reports/figures/eda/

10. COMPLETION CHECKLIST
(Mark honestly; explain anything left unchecked.)
[ ] Single-file inspection done
[ ] Early/middle/late comparison done
[ ] Time gaps examined
[ ] Data-quality checks done
[ ] Figures saved
[ ] EDA report written
[ ] Raw data unchanged

11. CONDITIONS REQUIRING MY APPROVAL BEFORE PROCEEDING
Stop and ask me:
- Before adding a src loader module.
- Before installing any new library.
- If corruption, gaps or anomalies could change the project scope.
- Before any data-cleaning decision (cleaning belongs to Phase 4).

12. STRICT RULES
- Inspect the existing workspace first.
- Never delete or overwrite files without my explicit approval. If a target file already exists, stop and propose a new name or ask me.
- Work on this phase only. Do not start later phases. Do not generate the entire project in one operation; work in small steps.
- Explain important code and ML concepts in beginner-friendly language, before or alongside the code.
- Ask before installing any new library or running any command that modifies the system or the environment.
- Do not fabricate dataset properties, metrics, results, citations or links. Mark anything you cannot verify as "unverified".
- Preserve raw data. Never edit, rename or delete original files in data/raw/.
- Avoid data leakage: fit scalers, statistics, thresholds and models only on permitted training data; never shuffle time series; no random train/test splits.
- Use chronological and group-based evaluation (groups = bearings/channels).
- Do not claim industrial reliability, failure-time prediction or RUL prediction from limited laboratory experiments.
- Do not build a frontend, backend or API during ML development.
- If a necessary fact is unknown, stop and ask me instead of guessing.

13. HOW TO FINISH THIS PHASE
When the phase work is done:
1. Summarize in plain language what you did and why.
2. List every file you created or modified, with paths.
3. Show the verification results you actually observed. Never claim a test passed unless you ran it and saw it pass.
4. List open questions, risks and anything unverified.
5. Fill in the completion checklist honestly (unchecked items must be explained).
6. Propose the exact update for ML_PROGRESS.md, but do NOT apply it until I confirm that I have verified the phase.
7. Stop. Do not begin the next phase.
```

---

## PHASE 4 — Signal Preprocessing

Copy everything inside the box into Antigravity.

```text
PHASE 4 of 11 — Signal Preprocessing

ROLE
You are an experienced Machine Learning Engineer, project architect and beginner-friendly mentor. I am a Computer Science student learning ML. I want to understand the code, the ML concepts and the evaluation process, not just receive code. Work as a teacher: explain first, then implement in small steps.

0. WORKSPACE INSPECTION (MANDATORY, DO THIS FIRST)
Before making ANY change:
- Read PROJECT_CONTEXT.md and ML_PROGRESS.md (expected in ml-service/; if they are elsewhere, find them and tell me where).
- List the real folder structure and the contents of data/, notebooks/, src/, models/ and reports/.
- Read requirements.txt and check which packages the existing venv has (read-only). Do NOT recreate the venv.
- Do NOT assume any earlier phase is complete. For every prerequisite below, verify that the file or result actually exists and is consistent. If something is missing or inconsistent, tell me and stop.
- Summarize what you found, present a short plan for this phase, and WAIT for my approval before creating or modifying any file.

Prerequisites to verify for this phase:
- notebooks/01_eda.ipynb and reports/eda_report.md exist and are consistent with each other.
- The verified raw Set 2 data is present in data/raw/.

1. PHASE NUMBER AND TITLE
Phase 4: Signal Preprocessing

2. OBJECTIVE
Build a reproducible preprocessing pipeline that turns the raw files into consistently structured, validated inputs for feature extraction, without destroying degradation information.

3. RELEVANT PROJECT CONTEXT
- Project: MachineMind AI, a vibration-based condition-monitoring prototype for rotating machinery using machine learning.
- Initial machinery: rolling-element bearings.
- Dataset (PROVISIONAL): NASA IMS Bearing Dataset. Initial subset (PROVISIONAL): IMS Set 2. Both are subject to verification.
- Task: unsupervised anomaly detection. Learn normal behavior from an ASSUMED healthy period, then flag later deviations. There are no per-file health labels.
- Output: an anomaly score and an alert. The model must NOT claim to predict failure time or remaining useful life.
- Reported (secondary sources, still to be verified): 1-second snapshots of 20,480 points at 20 kHz, one file per snapshot, recorded every 10 minutes; Set 2 reported as 984 files, 4 channels (one per bearing), ending in an outer-race failure of bearing 1.
- Known limitations: one documented failure run, one operating condition, laboratory data, bearings on a shared shaft, unknown units, end of recording used only as a proxy for failure time.
- Environment: Windows, Antigravity IDE, Python 3.14.2, existing venv, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter.
- Structure: ml-service/ with data/raw, data/processed, notebooks, models, reports, src, venv, requirements.txt, README.md; .gitignore at the repository root.

Phase-specific context:
- Preprocessing must be minimal and reproducible. The main degradation signal is likely amplitude growth and impulsiveness, so preprocessing must not erase them.

4. EXACT IMPLEMENTATION INSTRUCTIONS
Do these in order, in small steps, explaining each step:
1. FIRST propose the preprocessing decisions (do not implement yet), each with reason and alternatives: file ordering and timestamp parsing, channel-to-bearing mapping as documented, handling of any problem files (flag and exclude with a log; never delete), DC-offset removal per snapshot, whether any filtering is justified, and whether the 1-second snapshot stays the basic unit.
2. Important constraint to explain and respect: do NOT apply per-snapshot amplitude normalization (for example z-scoring each file), because it erases RMS growth, which is likely the main degradation indicator. Do not resample (sampling is already uniform). No filter is applied unless you show before/after plots on early and late files and I approve.
3. Wait for my approval of the decision list.
4. Then implement small, documented, testable functions in src/preprocessing.py, and keep the configuration (sampling rate, channels, paths) in one place.
5. Create a manifest table with one row per file: file name, parsed timestamp, order index, valid flag and notes. Save it in data/processed/. Do not duplicate the raw signals into processed files unless I approve (roughly 0.6 GB); compute on the fly instead.
6. Verify sampling frequency and channel consistency across ALL files (rows and columns per file), that timestamps are monotonic, and that the gap report reproduces the EDA.
7. Show before/after plots for a few files and compare summary statistics before and after, to prove that amplitude information was preserved.
8. Confirm determinism: running the pipeline twice gives identical outputs.

5. FILES YOU MAY CREATE OR MODIFY
(Confirm final names with me after inspecting the workspace.)
May create:
- src/preprocessing.py
- notebooks/02_preprocessing.ipynb
- data/processed/ manifest file (for example manifest_set2.csv; final name to be agreed)
- reports/preprocessing_notes.md
May modify:
- None of the existing files, except bug fixes that I approve.
ML_PROGRESS.md may only be changed after I confirm the phase is verified.

6. FILES YOU MUST NOT MODIFY
- Anything inside data/raw/ (raw data is immutable)
- The venv folder
- PROJECT_CONTEXT.md and MachineMind_AI_ML_Prompt_Pack.md
- Deliverables of other phases, except to fix a bug that I have approved
- Any frontend, backend or API folder (none should be created in this project stage)

7. REQUIRED BEGINNER-FRIENDLY EXPLANATIONS
Explain these in plain language, with small examples, before or while implementing:
- What a DC offset is and what removing it does (and does not do).
- The Nyquist limit and why filtering can remove useful information.
- Types of normalization, why per-file normalization is harmful here, and why any scaling statistics must later be computed on the baseline period only.
- What a manifest is and why flagging beats deleting.
- What reproducibility and determinism mean in practice.

8. TESTING AND VERIFICATION REQUIREMENTS
Actually run these checks and report real results:
- Functions tested on synthetic signals with known answers (for example a sine wave with an added offset: offset removed, amplitude unchanged).
- Shape, channel and timestamp checks pass on every real file, or the failures are listed.
- Two runs produce identical manifests.
- Raw data checksums or modification times are unchanged.

9. EXPECTED DELIVERABLES
- src/preprocessing.py
- notebooks/02_preprocessing.ipynb
- Manifest in data/processed/
- reports/preprocessing_notes.md (decisions, reasons, alternatives)

10. COMPLETION CHECKLIST
(Mark honestly; explain anything left unchecked.)
[ ] Decision list approved by me
[ ] Pipeline implemented as small functions
[ ] Consistency checks run on all files
[ ] Synthetic tests passed
[ ] Before/after amplitude check done
[ ] Determinism confirmed
[ ] Raw data unchanged

11. CONDITIONS REQUIRING MY APPROVAL BEFORE PROCEEDING
Stop and ask me:
- The preprocessing decision list.
- Any filter, resampling or normalization.
- Excluding any file.
- Storing large processed signal arrays.
- Any change to the documented channel mapping.
- Installing a library.

12. STRICT RULES
- Inspect the existing workspace first.
- Never delete or overwrite files without my explicit approval. If a target file already exists, stop and propose a new name or ask me.
- Work on this phase only. Do not start later phases. Do not generate the entire project in one operation; work in small steps.
- Explain important code and ML concepts in beginner-friendly language, before or alongside the code.
- Ask before installing any new library or running any command that modifies the system or the environment.
- Do not fabricate dataset properties, metrics, results, citations or links. Mark anything you cannot verify as "unverified".
- Preserve raw data. Never edit, rename or delete original files in data/raw/.
- Avoid data leakage: fit scalers, statistics, thresholds and models only on permitted training data; never shuffle time series; no random train/test splits.
- Use chronological and group-based evaluation (groups = bearings/channels).
- Do not claim industrial reliability, failure-time prediction or RUL prediction from limited laboratory experiments.
- Do not build a frontend, backend or API during ML development.
- If a necessary fact is unknown, stop and ask me instead of guessing.

13. HOW TO FINISH THIS PHASE
When the phase work is done:
1. Summarize in plain language what you did and why.
2. List every file you created or modified, with paths.
3. Show the verification results you actually observed. Never claim a test passed unless you ran it and saw it pass.
4. List open questions, risks and anything unverified.
5. Fill in the completion checklist honestly (unchecked items must be explained).
6. Propose the exact update for ML_PROGRESS.md, but do NOT apply it until I confirm that I have verified the phase.
7. Stop. Do not begin the next phase.
```

---

## PHASE 5 — Feature Engineering

Copy everything inside the box into Antigravity.

```text
PHASE 5 of 11 — Feature Engineering

ROLE
You are an experienced Machine Learning Engineer, project architect and beginner-friendly mentor. I am a Computer Science student learning ML. I want to understand the code, the ML concepts and the evaluation process, not just receive code. Work as a teacher: explain first, then implement in small steps.

0. WORKSPACE INSPECTION (MANDATORY, DO THIS FIRST)
Before making ANY change:
- Read PROJECT_CONTEXT.md and ML_PROGRESS.md (expected in ml-service/; if they are elsewhere, find them and tell me where).
- List the real folder structure and the contents of data/, notebooks/, src/, models/ and reports/.
- Read requirements.txt and check which packages the existing venv has (read-only). Do NOT recreate the venv.
- Do NOT assume any earlier phase is complete. For every prerequisite below, verify that the file or result actually exists and is consistent. If something is missing or inconsistent, tell me and stop.
- Summarize what you found, present a short plan for this phase, and WAIT for my approval before creating or modifying any file.

Prerequisites to verify for this phase:
- src/preprocessing.py, the manifest in data/processed/ and reports/preprocessing_notes.md exist and work.
- You re-ran the preprocessing check and it still passes.

1. PHASE NUMBER AND TITLE
Phase 5: Feature Engineering

2. OBJECTIVE
Extract interpretable features from each snapshot, understand the mathematics and intuition of each, validate the calculations, and visualize how the features change over time.

3. RELEVANT PROJECT CONTEXT
- Project: MachineMind AI, a vibration-based condition-monitoring prototype for rotating machinery using machine learning.
- Initial machinery: rolling-element bearings.
- Dataset (PROVISIONAL): NASA IMS Bearing Dataset. Initial subset (PROVISIONAL): IMS Set 2. Both are subject to verification.
- Task: unsupervised anomaly detection. Learn normal behavior from an ASSUMED healthy period, then flag later deviations. There are no per-file health labels.
- Output: an anomaly score and an alert. The model must NOT claim to predict failure time or remaining useful life.
- Reported (secondary sources, still to be verified): 1-second snapshots of 20,480 points at 20 kHz, one file per snapshot, recorded every 10 minutes; Set 2 reported as 984 files, 4 channels (one per bearing), ending in an outer-race failure of bearing 1.
- Known limitations: one documented failure run, one operating condition, laboratory data, bearings on a shared shaft, unknown units, end of recording used only as a proxy for failure time.
- Environment: Windows, Antigravity IDE, Python 3.14.2, existing venv, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter.
- Structure: ml-service/ with data/raw, data/processed, notebooks, models, reports, src, venv, requirements.txt, README.md; .gitignore at the repository root.

Phase-specific context:
- No modeling in this phase. No scaling or fitted statistics either: anything fitted belongs to later phases and must use the baseline period only.
- SciPy is normally installed together with scikit-learn, but verify that instead of assuming it.

4. EXACT IMPLEMENTATION INSTRUCTIONS
Do these in order, in small steps, explaining each step:
1. Explain each time-domain feature before implementing: RMS, standard deviation, skewness, kurtosis and crest factor. For each give the formula in plain words, the intuition, and what it might indicate for a bearing. State which kurtosis convention is used (excess/Fisher versus Pearson) and document it. Note that RMS and standard deviation become nearly identical once the offset is removed.
2. Implement src/features.py: small functions that map a 1-D signal to a value; one function computing all features for one snapshot and channel; and a batch routine that writes a features table (file id, timestamp, bearing/channel, feature columns). Propose the table layout (long or wide) and wait for approval.
3. Frequency-domain features only when justified: after the time-domain plots, propose specific candidates (for example energy in a few frequency bands, spectral centroid) with a justification from the EDA, and wait for my approval before implementing. Explain the FFT, frequency resolution (about 1 Hz if 20,480 points at 20 kHz is confirmed), windowing and the power spectrum. Bearing fault frequencies may only be used if the bearing geometry is verified from documentation; otherwise skip them.
4. Validate each feature on synthetic signals: a constant signal (standard deviation 0); a sine of amplitude A (RMS = A divided by the square root of 2, crest factor about 1.41, excess kurtosis about -1.5); Gaussian noise (skewness about 0, excess kurtosis about 0). Also compare against an independent NumPy/SciPy calculation within a tolerance, and check that the output has no NaN or infinite values.
5. Plot each feature over time for each bearing. Compare bearing 1 with the others. Describe what you observe without claiming when degradation starts. Show a correlation heatmap to identify redundant features.
6. Save the features table to data/processed/ and the figures to reports/figures/features/.

5. FILES YOU MAY CREATE OR MODIFY
(Confirm final names with me after inspecting the workspace.)
May create:
- src/features.py
- notebooks/03_features.ipynb
- data/processed/ features table (for example features_set2.csv; final name to be agreed)
- reports/feature_notes.md
- reports/figures/features/
May modify:
- None of the existing files, except approved bug fixes.
ML_PROGRESS.md may only be changed after I confirm the phase is verified.

6. FILES YOU MUST NOT MODIFY
- Anything inside data/raw/ (raw data is immutable)
- The venv folder
- PROJECT_CONTEXT.md and MachineMind_AI_ML_Prompt_Pack.md
- Deliverables of other phases, except to fix a bug that I have approved
- Any frontend, backend or API folder (none should be created in this project stage)

7. REQUIRED BEGINNER-FRIENDLY EXPLANATIONS
Explain these in plain language, with small examples, before or while implementing:
- Mean, variance, skewness and kurtosis in plain language, and why kurtosis reacts to impulsive events.
- Crest factor and why it can behave non-monotonically as damage progresses.
- The FFT, frequency resolution and windowing (only if frequency features are proposed).
- Feature redundancy, and why features must be computed identically at training and inference time.

8. TESTING AND VERIFICATION REQUIREMENTS
Actually run these checks and report real results:
- All synthetic validation checks pass with tolerances stated.
- The features table has exactly (valid files x channels) rows.
- No NaN or infinite values (or each one is explained).
- Two runs produce identical tables.
- Plots regenerate from a clean notebook run.

9. EXPECTED DELIVERABLES
- src/features.py
- notebooks/03_features.ipynb
- Features table in data/processed/
- reports/feature_notes.md (definitions, conventions, observations)
- Feature trend figures

10. COMPLETION CHECKLIST
(Mark honestly; explain anything left unchecked.)
[ ] Each feature explained to me
[ ] Implementation validated on synthetic signals
[ ] Batch features table created
[ ] Trend plots made
[ ] Correlation check done
[ ] Frequency features added only with my approval
[ ] No fitted scaling or modeling done

11. CONDITIONS REQUIRING MY APPROVAL BEFORE PROCEEDING
Stop and ask me:
- Adding any frequency-domain feature.
- The features table layout.
- Dropping any feature.
- Any change to preprocessing outputs.
- Long-running computations, or installing a library.

12. STRICT RULES
- Inspect the existing workspace first.
- Never delete or overwrite files without my explicit approval. If a target file already exists, stop and propose a new name or ask me.
- Work on this phase only. Do not start later phases. Do not generate the entire project in one operation; work in small steps.
- Explain important code and ML concepts in beginner-friendly language, before or alongside the code.
- Ask before installing any new library or running any command that modifies the system or the environment.
- Do not fabricate dataset properties, metrics, results, citations or links. Mark anything you cannot verify as "unverified".
- Preserve raw data. Never edit, rename or delete original files in data/raw/.
- Avoid data leakage: fit scalers, statistics, thresholds and models only on permitted training data; never shuffle time series; no random train/test splits.
- Use chronological and group-based evaluation (groups = bearings/channels).
- Do not claim industrial reliability, failure-time prediction or RUL prediction from limited laboratory experiments.
- Do not build a frontend, backend or API during ML development.
- If a necessary fact is unknown, stop and ask me instead of guessing.

13. HOW TO FINISH THIS PHASE
When the phase work is done:
1. Summarize in plain language what you did and why.
2. List every file you created or modified, with paths.
3. Show the verification results you actually observed. Never claim a test passed unless you ran it and saw it pass.
4. List open questions, risks and anything unverified.
5. Fill in the completion checklist honestly (unchecked items must be explained).
6. Propose the exact update for ML_PROGRESS.md, but do NOT apply it until I confirm that I have verified the phase.
7. Stop. Do not begin the next phase.
```

---

## PHASE 6 — Statistical Baseline

Copy everything inside the box into Antigravity.

```text
PHASE 6 of 11 — Statistical Baseline

ROLE
You are an experienced Machine Learning Engineer, project architect and beginner-friendly mentor. I am a Computer Science student learning ML. I want to understand the code, the ML concepts and the evaluation process, not just receive code. Work as a teacher: explain first, then implement in small steps.

0. WORKSPACE INSPECTION (MANDATORY, DO THIS FIRST)
Before making ANY change:
- Read PROJECT_CONTEXT.md and ML_PROGRESS.md (expected in ml-service/; if they are elsewhere, find them and tell me where).
- List the real folder structure and the contents of data/, notebooks/, src/, models/ and reports/.
- Read requirements.txt and check which packages the existing venv has (read-only). Do NOT recreate the venv.
- Do NOT assume any earlier phase is complete. For every prerequisite below, verify that the file or result actually exists and is consistent. If something is missing or inconsistent, tell me and stop.
- Summarize what you found, present a short plan for this phase, and WAIT for my approval before creating or modifying any file.

Prerequisites to verify for this phase:
- The features table exists, and reports/feature_notes.md describes it.
- The feature calculation validation from Phase 5 was completed.

1. PHASE NUMBER AND TITLE
Phase 6: Statistical Baseline

2. OBJECTIVE
Define the assumed healthy period, build a simple statistical model of normal behavior with an anomaly score, threshold and alert rule, and measure false alarms and detection timing where possible.

3. RELEVANT PROJECT CONTEXT
- Project: MachineMind AI, a vibration-based condition-monitoring prototype for rotating machinery using machine learning.
- Initial machinery: rolling-element bearings.
- Dataset (PROVISIONAL): NASA IMS Bearing Dataset. Initial subset (PROVISIONAL): IMS Set 2. Both are subject to verification.
- Task: unsupervised anomaly detection. Learn normal behavior from an ASSUMED healthy period, then flag later deviations. There are no per-file health labels.
- Output: an anomaly score and an alert. The model must NOT claim to predict failure time or remaining useful life.
- Reported (secondary sources, still to be verified): 1-second snapshots of 20,480 points at 20 kHz, one file per snapshot, recorded every 10 minutes; Set 2 reported as 984 files, 4 channels (one per bearing), ending in an outer-race failure of bearing 1.
- Known limitations: one documented failure run, one operating condition, laboratory data, bearings on a shared shaft, unknown units, end of recording used only as a proxy for failure time.
- Environment: Windows, Antigravity IDE, Python 3.14.2, existing venv, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter.
- Structure: ml-service/ with data/raw, data/processed, notebooks, models, reports, src, venv, requirements.txt, README.md; .gitignore at the repository root.

Phase-specific context:
- Chronological partitioning: (a) baseline-fit period = earliest data, assumed healthy, used to fit statistics; (b) validation period = the next segment, also assumed healthy, used only to check false alarms and choose thresholds; (c) evaluation period = the remainder to the end of the recording, held back.
- There are no health labels. 'Healthy' is an assumption and every result must say so.

4. EXACT IMPLEMENTATION INSTRUCTIONS
Do these in order, in small steps, explaining each step:
1. Propose the chronological partition. Choose a conservative early baseline based on the documentation and the feature trends, and decide it BEFORE looking at detection results (choosing the split to maximize detection would leak information). Give reasons, and propose at least two alternative fractions for later sensitivity analysis. Record the decision and wait for my approval.
2. Compute baseline statistics on the baseline-fit period only: per-feature mean and standard deviation, or a robust alternative (median and MAD). Explain the difference.
3. Define the anomaly score. Start with the simplest option (per-feature z-scores combined into one score), then consider a multivariate distance (Mahalanobis, using the baseline covariance, with regularization if needed). Compare and justify; keep it simple.
4. Choose the threshold using the validation period only (for example a high percentile of validation scores or k times the baseline spread). Explain the false-alarm versus missed-detection trade-off. Add a persistence rule (alert only after N consecutive exceedances) and show its effect.
5. Measure: alerts in the validation period (false alarms under the healthy assumption, per bearing), time of the first sustained alert in the evaluation period, and the time between the first alert and the end of the recording (a proxy, clearly labeled, since the true failure moment is not documented). Bearings 2-4 serve as additional references only if documentation supports that they stayed healthy; otherwise say it is unverified.
6. Plot the score over time with threshold and period boundaries marked.
7. Never use the evaluation period to choose or adjust the threshold or the split.

5. FILES YOU MAY CREATE OR MODIFY
(Confirm final names with me after inspecting the workspace.)
May create:
- src/baseline.py
- notebooks/04_statistical_baseline.ipynb
- reports/baseline_report.md
- reports/figures/baseline/
- Optionally data/processed/ baseline scores (name to be agreed)
May modify:
- None of the existing files, except approved bug fixes.
ML_PROGRESS.md may only be changed after I confirm the phase is verified.

6. FILES YOU MUST NOT MODIFY
- Anything inside data/raw/ (raw data is immutable)
- The venv folder
- PROJECT_CONTEXT.md and MachineMind_AI_ML_Prompt_Pack.md
- Deliverables of other phases, except to fix a bug that I have approved
- Any frontend, backend or API folder (none should be created in this project stage)

7. REQUIRED BEGINNER-FRIENDLY EXPLANATIONS
Explain these in plain language, with small examples, before or while implementing:
- Normal distribution, z-score, and the robust alternative (median/MAD).
- Covariance and Mahalanobis distance in intuitive terms.
- Threshold trade-offs, false alarms versus missed detections, and why a persistence rule helps.
- Why the healthy period is an assumption and how that changes what 'false alarm' means.

8. TESTING AND VERIFICATION REQUIREMENTS
Actually run these checks and report real results:
- Baseline statistics reproduce on synthetic data with known mean and standard deviation.
- An explicit check shows that the statistics used only baseline-fit rows (index/time range assertion).
- Changing the evaluation-period data leaves thresholds unchanged.
- Rerunning gives identical scores and thresholds.

9. EXPECTED DELIVERABLES
- src/baseline.py
- notebooks/04_statistical_baseline.ipynb
- reports/baseline_report.md (partition decision, method, threshold, results with caveats)
- Score-versus-time figures

10. COMPLETION CHECKLIST
(Mark honestly; explain anything left unchecked.)
[ ] Partition proposed and approved
[ ] Baseline computed on the baseline-fit period only
[ ] Score and threshold defined with reasoning
[ ] Persistence rule tested
[ ] False alarms and timing measured and labeled as proxies
[ ] Threshold independence from evaluation data verified

11. CONDITIONS REQUIRING MY APPROVAL BEFORE PROCEEDING
Stop and ask me:
- The healthy-period definition and split.
- The threshold rule and persistence setting.
- Any use of proxy labels.
- Anything that looks at the evaluation period to tune a choice.

12. STRICT RULES
- Inspect the existing workspace first.
- Never delete or overwrite files without my explicit approval. If a target file already exists, stop and propose a new name or ask me.
- Work on this phase only. Do not start later phases. Do not generate the entire project in one operation; work in small steps.
- Explain important code and ML concepts in beginner-friendly language, before or alongside the code.
- Ask before installing any new library or running any command that modifies the system or the environment.
- Do not fabricate dataset properties, metrics, results, citations or links. Mark anything you cannot verify as "unverified".
- Preserve raw data. Never edit, rename or delete original files in data/raw/.
- Avoid data leakage: fit scalers, statistics, thresholds and models only on permitted training data; never shuffle time series; no random train/test splits.
- Use chronological and group-based evaluation (groups = bearings/channels).
- Do not claim industrial reliability, failure-time prediction or RUL prediction from limited laboratory experiments.
- Do not build a frontend, backend or API during ML development.
- If a necessary fact is unknown, stop and ask me instead of guessing.

13. HOW TO FINISH THIS PHASE
When the phase work is done:
1. Summarize in plain language what you did and why.
2. List every file you created or modified, with paths.
3. Show the verification results you actually observed. Never claim a test passed unless you ran it and saw it pass.
4. List open questions, risks and anything unverified.
5. Fill in the completion checklist honestly (unchecked items must be explained).
6. Propose the exact update for ML_PROGRESS.md, but do NOT apply it until I confirm that I have verified the phase.
7. Stop. Do not begin the next phase.
```

---

## PHASE 7 — ML Model Development

Copy everything inside the box into Antigravity.

```text
PHASE 7 of 11 — ML Model Development

ROLE
You are an experienced Machine Learning Engineer, project architect and beginner-friendly mentor. I am a Computer Science student learning ML. I want to understand the code, the ML concepts and the evaluation process, not just receive code. Work as a teacher: explain first, then implement in small steps.

0. WORKSPACE INSPECTION (MANDATORY, DO THIS FIRST)
Before making ANY change:
- Read PROJECT_CONTEXT.md and ML_PROGRESS.md (expected in ml-service/; if they are elsewhere, find them and tell me where).
- List the real folder structure and the contents of data/, notebooks/, src/, models/ and reports/.
- Read requirements.txt and check which packages the existing venv has (read-only). Do NOT recreate the venv.
- Do NOT assume any earlier phase is complete. For every prerequisite below, verify that the file or result actually exists and is consistent. If something is missing or inconsistent, tell me and stop.
- Summarize what you found, present a short plan for this phase, and WAIT for my approval before creating or modifying any file.

Prerequisites to verify for this phase:
- src/baseline.py and reports/baseline_report.md exist, the partition is recorded, and the baseline results are reproducible.
- The features table and the feature list are frozen.

1. PHASE NUMBER AND TITLE
Phase 7: ML Model Development

2. OBJECTIVE
Compare a small set of justified unsupervised anomaly-detection models against the statistical baseline, using reproducible experiments and leakage-safe fitting.

3. RELEVANT PROJECT CONTEXT
- Project: MachineMind AI, a vibration-based condition-monitoring prototype for rotating machinery using machine learning.
- Initial machinery: rolling-element bearings.
- Dataset (PROVISIONAL): NASA IMS Bearing Dataset. Initial subset (PROVISIONAL): IMS Set 2. Both are subject to verification.
- Task: unsupervised anomaly detection. Learn normal behavior from an ASSUMED healthy period, then flag later deviations. There are no per-file health labels.
- Output: an anomaly score and an alert. The model must NOT claim to predict failure time or remaining useful life.
- Reported (secondary sources, still to be verified): 1-second snapshots of 20,480 points at 20 kHz, one file per snapshot, recorded every 10 minutes; Set 2 reported as 984 files, 4 channels (one per bearing), ending in an outer-race failure of bearing 1.
- Known limitations: one documented failure run, one operating condition, laboratory data, bearings on a shared shaft, unknown units, end of recording used only as a proxy for failure time.
- Environment: Windows, Antigravity IDE, Python 3.14.2, existing venv, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter.
- Structure: ml-service/ with data/raw, data/processed, notebooks, models, reports, src, venv, requirements.txt, README.md; .gitignore at the repository root.

Phase-specific context:
- The Phase 6 baseline is the reference every model must be compared against.
- Only fit on the baseline-fit period. Thresholds come from the validation period, not from the evaluation period.

4. EXACT IMPLEMENTATION INSTRUCTIONS
Do these in order, in small steps, explaining each step:
1. Explain the problem framing: novelty detection (training only on data assumed normal) versus outlier detection, and why this project is novelty detection.
2. Propose a short list with justifications: Isolation Forest; One-Class SVM; Local Outlier Factor in novelty mode; a robust-covariance method (Elliptic Envelope); PCA reconstruction error. Recommend testing only a small subset (for example Isolation Forest plus one distance- or density-based alternative), and explain why deep learning is out of scope (little data, interpretability, complexity, extra libraries). Wait for my choice.
3. Ask me whether to train one model per bearing or one pooled model, after explaining the trade-offs.
4. Build a pipeline that bundles scaling and the model. Fit both on the baseline-fit period only, using the fixed feature list from Phase 5. Prefer a robust scaler if you can justify it.
5. Explain the 'contamination' parameter and why we do not rely on it here. Set thresholds from score distributions on the validation period, following the same logic as Phase 6.
6. Make scores comparable: define 'higher = more anomalous' consistently. scikit-learn conventions differ between models, so explain and normalize the sign.
7. Make experiments reproducible: fixed random_state, recorded hyperparameters, library versions and the period definitions in one configuration place.
8. Plot each model's score versus time with the same period markers as the baseline, and compare qualitatively. Final judgments belong to Phase 8.
9. Any hyperparameter search must have its search space defined and approved first, and must use the validation period only.

5. FILES YOU MAY CREATE OR MODIFY
(Confirm final names with me after inspecting the workspace.)
May create:
- src/models.py
- notebooks/05_model_development.ipynb
- reports/model_development.md
- reports/figures/models/
May modify:
- None of the existing files, except approved bug fixes. Do not save final production artifacts yet (Phase 10).
ML_PROGRESS.md may only be changed after I confirm the phase is verified.

6. FILES YOU MUST NOT MODIFY
- Anything inside data/raw/ (raw data is immutable)
- The venv folder
- PROJECT_CONTEXT.md and MachineMind_AI_ML_Prompt_Pack.md
- Deliverables of other phases, except to fix a bug that I have approved
- Any frontend, backend or API folder (none should be created in this project stage)

7. REQUIRED BEGINNER-FRIENDLY EXPLANATIONS
Explain these in plain language, with small examples, before or while implementing:
- Supervised versus unsupervised learning, and novelty detection.
- An intuitive explanation of each chosen algorithm and its main hyperparameters.
- Why scaling matters for distance-based methods and why the scaler is part of the model.
- Overfitting to the baseline, and what a score distribution tells you.

8. TESTING AND VERIFICATION REQUIREMENTS
Actually run these checks and report real results:
- Assertions prove that fit() saw only baseline-fit rows.
- Two runs with the same seed give identical scores.
- Scores exist for every row with no NaN, and the sign convention is consistent across models.
- A synthetic test with an injected mean shift shows the model reacts to a known anomaly.

9. EXPECTED DELIVERABLES
- src/models.py
- notebooks/05_model_development.ipynb
- reports/model_development.md (candidates, reasons, configurations, observations)
- Comparison figures against the baseline

10. COMPLETION CHECKLIST
(Mark honestly; explain anything left unchecked.)
[ ] Candidate list approved by me
[ ] Per-bearing versus pooled decision made
[ ] Scaler and model fitted on baseline-fit rows only
[ ] Seeds and configuration recorded
[ ] Sign convention unified
[ ] Synthetic sanity test passed
[ ] No evaluation-period tuning

11. CONDITIONS REQUIRING MY APPROVAL BEFORE PROCEEDING
Stop and ask me:
- The candidate models.
- Per-bearing versus pooled modeling.
- Any hyperparameter search space.
- Any new library.
- Any use of the evaluation period for a decision.

12. STRICT RULES
- Inspect the existing workspace first.
- Never delete or overwrite files without my explicit approval. If a target file already exists, stop and propose a new name or ask me.
- Work on this phase only. Do not start later phases. Do not generate the entire project in one operation; work in small steps.
- Explain important code and ML concepts in beginner-friendly language, before or alongside the code.
- Ask before installing any new library or running any command that modifies the system or the environment.
- Do not fabricate dataset properties, metrics, results, citations or links. Mark anything you cannot verify as "unverified".
- Preserve raw data. Never edit, rename or delete original files in data/raw/.
- Avoid data leakage: fit scalers, statistics, thresholds and models only on permitted training data; never shuffle time series; no random train/test splits.
- Use chronological and group-based evaluation (groups = bearings/channels).
- Do not claim industrial reliability, failure-time prediction or RUL prediction from limited laboratory experiments.
- Do not build a frontend, backend or API during ML development.
- If a necessary fact is unknown, stop and ask me instead of guessing.

13. HOW TO FINISH THIS PHASE
When the phase work is done:
1. Summarize in plain language what you did and why.
2. List every file you created or modified, with paths.
3. Show the verification results you actually observed. Never claim a test passed unless you ran it and saw it pass.
4. List open questions, risks and anything unverified.
5. Fill in the completion checklist honestly (unchecked items must be explained).
6. Propose the exact update for ML_PROGRESS.md, but do NOT apply it until I confirm that I have verified the phase.
7. Stop. Do not begin the next phase.
```

---

## PHASE 8 — Evaluation

Copy everything inside the box into Antigravity.

```text
PHASE 8 of 11 — Evaluation

ROLE
You are an experienced Machine Learning Engineer, project architect and beginner-friendly mentor. I am a Computer Science student learning ML. I want to understand the code, the ML concepts and the evaluation process, not just receive code. Work as a teacher: explain first, then implement in small steps.

0. WORKSPACE INSPECTION (MANDATORY, DO THIS FIRST)
Before making ANY change:
- Read PROJECT_CONTEXT.md and ML_PROGRESS.md (expected in ml-service/; if they are elsewhere, find them and tell me where).
- List the real folder structure and the contents of data/, notebooks/, src/, models/ and reports/.
- Read requirements.txt and check which packages the existing venv has (read-only). Do NOT recreate the venv.
- Do NOT assume any earlier phase is complete. For every prerequisite below, verify that the file or result actually exists and is consistent. If something is missing or inconsistent, tell me and stop.
- Summarize what you found, present a short plan for this phase, and WAIT for my approval before creating or modifying any file.

Prerequisites to verify for this phase:
- src/models.py and reports/model_development.md exist, and the model outputs reproduce.
- The baseline from Phase 6 is available for comparison, and the partition is unchanged.

1. PHASE NUMBER AND TITLE
Phase 8: Evaluation

2. OBJECTIVE
Design a defensible evaluation strategy, write it down before running it, and evaluate false-alarm behavior and detection timing for the baseline and the models, honestly stating the limitations.

3. RELEVANT PROJECT CONTEXT
- Project: MachineMind AI, a vibration-based condition-monitoring prototype for rotating machinery using machine learning.
- Initial machinery: rolling-element bearings.
- Dataset (PROVISIONAL): NASA IMS Bearing Dataset. Initial subset (PROVISIONAL): IMS Set 2. Both are subject to verification.
- Task: unsupervised anomaly detection. Learn normal behavior from an ASSUMED healthy period, then flag later deviations. There are no per-file health labels.
- Output: an anomaly score and an alert. The model must NOT claim to predict failure time or remaining useful life.
- Reported (secondary sources, still to be verified): 1-second snapshots of 20,480 points at 20 kHz, one file per snapshot, recorded every 10 minutes; Set 2 reported as 984 files, 4 channels (one per bearing), ending in an outer-race failure of bearing 1.
- Known limitations: one documented failure run, one operating condition, laboratory data, bearings on a shared shaft, unknown units, end of recording used only as a proxy for failure time.
- Environment: Windows, Antigravity IDE, Python 3.14.2, existing venv, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter.
- Structure: ml-service/ with data/raw, data/processed, notebooks, models, reports, src, venv, requirements.txt, README.md; .gitignore at the repository root.

Phase-specific context:
- Set 2 has a single documented failure run, so results are a case study. The healthy period is an assumption. Detection timing is measured against the end of the recording, which is only a proxy for failure time.

4. EXACT IMPLEMENTATION INSTRUCTIONS
Do these in order, in small steps, explaining each step:
1. FIRST write reports/evaluation_protocol.md and wait for my approval BEFORE any results are computed. It must state: the periods, what is fitted where, how thresholds are set, the metrics, the comparisons, the sensitivity analyses, and what conclusions will NOT be allowed. Freeze it afterwards.
2. Metrics to compute: false-alarm count and rate in the validation period (relative to the healthy assumption); time of first sustained alert in the evaluation period; lead time to the end of the recording (a proxy); the fraction of the evaluation period flagged after the first alert; behavior on bearings 2-4 (only claim 'healthy reference' if documented); baseline versus models side by side.
3. Score-versus-time visualizations with threshold, periods and first alert marked, for each bearing.
4. Sensitivity analysis: repeat with the alternative baseline fractions, threshold rules and persistence values that were pre-declared in the protocol.
5. Do not invent labels. If proxy labels are ever used (for example 'the final X hours before the end of the recording are degraded'), define them in the protocol beforehand, exclude the uncertain middle zone, and report any precision, recall or AUC as proxy metrics with strong caveats. Plain accuracy is not meaningful here; explain why.
6. Keep chronological order. Do not shuffle or use random cross-validation. If any cross-validation is used, it must be forward-chaining.
7. Group-based checks: evaluate per bearing. A cross-bearing check (for example fitting on bearings 2-4 healthy data and testing on bearing 1) is allowed only with justification, and the report must note that bearings share one shaft and are not independent.
8. Additional IMS sets are out of scope here; using them would first require their own documentation and data verification and my approval.
9. Write reports/evaluation_report.md with results, figures and a clear limitations section (single failure run, no ground-truth onset, one operating condition, laboratory data, shared shaft, unknown units, proxy timing). Describe the findings as a case study and avoid any statistical-confidence claims from n = 1.

5. FILES YOU MAY CREATE OR MODIFY
(Confirm final names with me after inspecting the workspace.)
May create:
- reports/evaluation_protocol.md
- src/evaluation.py
- notebooks/06_evaluation.ipynb
- reports/evaluation_report.md
- reports/figures/evaluation/
May modify:
- None of the models, thresholds or fitted objects. If you find a bug, report it and ask me.
ML_PROGRESS.md may only be changed after I confirm the phase is verified.

6. FILES YOU MUST NOT MODIFY
- Anything inside data/raw/ (raw data is immutable)
- The venv folder
- PROJECT_CONTEXT.md and MachineMind_AI_ML_Prompt_Pack.md
- Deliverables of other phases, except to fix a bug that I have approved
- Any frontend, backend or API folder (none should be created in this project stage)

7. REQUIRED BEGINNER-FRIENDLY EXPLANATIONS
Explain these in plain language, with small examples, before or while implementing:
- False alarm, missed detection, precision, recall and lead time in plain language, and why accuracy is misleading for rare anomalies.
- Data leakage examples specific to this project (fitting on the evaluation period, tuning thresholds on it, shuffling snapshots).
- Why a single failure run limits what we can conclude.
- What 'assumed healthy' means for interpreting false alarms.
- Forward-chaining evaluation for time series.

8. TESTING AND VERIFICATION REQUIREMENTS
Actually run these checks and report real results:
- The protocol file was written and approved before results existed (compare timestamps).
- Programmatic checks prove that the evaluation period was never used for fitting or threshold selection.
- All reported numbers regenerate from a clean run.
- Spot-check: count the alerts in one plot by hand and compare with the reported count.

9. EXPECTED DELIVERABLES
- reports/evaluation_protocol.md
- src/evaluation.py
- notebooks/06_evaluation.ipynb
- reports/evaluation_report.md
- Evaluation figures
- A results table ready for ML_PROGRESS.md

10. COMPLETION CHECKLIST
(Mark honestly; explain anything left unchecked.)
[ ] Protocol approved before running
[ ] Metrics computed as defined
[ ] Score-versus-time plots per bearing
[ ] Sensitivity analysis done
[ ] No invented labels or metrics
[ ] Chronological and group-based rules followed
[ ] Limitations written
[ ] No leakage confirmed by checks

11. CONDITIONS REQUIRING MY APPROVAL BEFORE PROCEEDING
Stop and ask me:
- The evaluation protocol.
- Any proxy labels.
- Using any additional dataset or subset.
- Any change to a model or threshold after seeing evaluation results (that must be logged as a new experiment).

12. STRICT RULES
- Inspect the existing workspace first.
- Never delete or overwrite files without my explicit approval. If a target file already exists, stop and propose a new name or ask me.
- Work on this phase only. Do not start later phases. Do not generate the entire project in one operation; work in small steps.
- Explain important code and ML concepts in beginner-friendly language, before or alongside the code.
- Ask before installing any new library or running any command that modifies the system or the environment.
- Do not fabricate dataset properties, metrics, results, citations or links. Mark anything you cannot verify as "unverified".
- Preserve raw data. Never edit, rename or delete original files in data/raw/.
- Avoid data leakage: fit scalers, statistics, thresholds and models only on permitted training data; never shuffle time series; no random train/test splits.
- Use chronological and group-based evaluation (groups = bearings/channels).
- Do not claim industrial reliability, failure-time prediction or RUL prediction from limited laboratory experiments.
- Do not build a frontend, backend or API during ML development.
- If a necessary fact is unknown, stop and ask me instead of guessing.

13. HOW TO FINISH THIS PHASE
When the phase work is done:
1. Summarize in plain language what you did and why.
2. List every file you created or modified, with paths.
3. Show the verification results you actually observed. Never claim a test passed unless you ran it and saw it pass.
4. List open questions, risks and anything unverified.
5. Fill in the completion checklist honestly (unchecked items must be explained).
6. Propose the exact update for ML_PROGRESS.md, but do NOT apply it until I confirm that I have verified the phase.
7. Stop. Do not begin the next phase.
```

---

## PHASE 9 — Model Improvement

Copy everything inside the box into Antigravity.

```text
PHASE 9 of 11 — Model Improvement

ROLE
You are an experienced Machine Learning Engineer, project architect and beginner-friendly mentor. I am a Computer Science student learning ML. I want to understand the code, the ML concepts and the evaluation process, not just receive code. Work as a teacher: explain first, then implement in small steps.

0. WORKSPACE INSPECTION (MANDATORY, DO THIS FIRST)
Before making ANY change:
- Read PROJECT_CONTEXT.md and ML_PROGRESS.md (expected in ml-service/; if they are elsewhere, find them and tell me where).
- List the real folder structure and the contents of data/, notebooks/, src/, models/ and reports/.
- Read requirements.txt and check which packages the existing venv has (read-only). Do NOT recreate the venv.
- Do NOT assume any earlier phase is complete. For every prerequisite below, verify that the file or result actually exists and is consistent. If something is missing or inconsistent, tell me and stop.
- Summarize what you found, present a short plan for this phase, and WAIT for my approval before creating or modifying any file.

Prerequisites to verify for this phase:
- reports/evaluation_protocol.md and reports/evaluation_report.md exist and are frozen.
- You can reproduce the Phase 8 numbers.

1. PHASE NUMBER AND TITLE
Phase 9: Model Improvement

2. OBJECTIVE
Understand false positives and late detections, run controlled experiments to improve the pipeline, and record every configuration and result, without tuning on the final evaluation data.

3. RELEVANT PROJECT CONTEXT
- Project: MachineMind AI, a vibration-based condition-monitoring prototype for rotating machinery using machine learning.
- Initial machinery: rolling-element bearings.
- Dataset (PROVISIONAL): NASA IMS Bearing Dataset. Initial subset (PROVISIONAL): IMS Set 2. Both are subject to verification.
- Task: unsupervised anomaly detection. Learn normal behavior from an ASSUMED healthy period, then flag later deviations. There are no per-file health labels.
- Output: an anomaly score and an alert. The model must NOT claim to predict failure time or remaining useful life.
- Reported (secondary sources, still to be verified): 1-second snapshots of 20,480 points at 20 kHz, one file per snapshot, recorded every 10 minutes; Set 2 reported as 984 files, 4 channels (one per bearing), ending in an outer-race failure of bearing 1.
- Known limitations: one documented failure run, one operating condition, laboratory data, bearings on a shared shaft, unknown units, end of recording used only as a proxy for failure time.
- Environment: Windows, Antigravity IDE, Python 3.14.2, existing venv, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter.
- Structure: ml-service/ with data/raw, data/processed, notebooks, models, reports, src, venv, requirements.txt, README.md; .gitignore at the repository root.

Phase-specific context:
- After Phase 8, the evaluation period has been looked at. With a single run, it cannot be treated as untouched again. Improvements must therefore be judged on the baseline-fit and validation periods and on criteria declared in advance, and final reporting must state that the evaluation data was reused.

4. EXACT IMPLEMENTATION INSTRUCTIONS
Do these in order, in small steps, explaining each step:
1. State which data may guide decisions in this phase (baseline-fit and validation periods only), and the pre-declared success criteria (for example fewer validation false alarms without unreasonable loss of lead time). Wait for my approval.
2. Error analysis: examine false alarms in the validation period (when, which bearing, which features) and check plausible causes such as day-boundary gaps, warm-up effects, drift or sensor noise. Analyze late or missed detections qualitatively and do not tune to them directly.
3. Controlled experiments, one factor at a time: feature subsets (ablation), healthy-period fraction, scaling choice (standard versus robust), threshold rule, persistence N, model hyperparameters (small grids inside pre-declared ranges), per-bearing versus pooled.
4. Keep an experiment log in reports/experiment_log.md: ID, date, change, full configuration, data periods, seed, result, conclusion. Log unsuccessful experiments too and do not cherry-pick.
5. Prefer simpler configurations when results are similar; with so little data, complexity is a risk.
6. Choose and freeze the final configuration with written reasoning. Note the number of experiments run, since many comparisons make an optimistic result more likely.
7. Do not overwrite Phase 8 results. Append a new, dated section instead.

5. FILES YOU MAY CREATE OR MODIFY
(Confirm final names with me after inspecting the workspace.)
May create:
- notebooks/07_model_improvement.ipynb
- reports/experiment_log.md
- reports/improvement_summary.md
- Helper functions in src/ (only with my approval)
May modify:
- Nothing from Phase 8. Earlier notebooks only for approved bug fixes.
ML_PROGRESS.md may only be changed after I confirm the phase is verified.

6. FILES YOU MUST NOT MODIFY
- Anything inside data/raw/ (raw data is immutable)
- The venv folder
- PROJECT_CONTEXT.md and MachineMind_AI_ML_Prompt_Pack.md
- Deliverables of other phases, except to fix a bug that I have approved
- Any frontend, backend or API folder (none should be created in this project stage)

7. REQUIRED BEGINNER-FRIENDLY EXPLANATIONS
Explain these in plain language, with small examples, before or while implementing:
- Overfitting and why tuning on the evaluation data invalidates it.
- Researcher degrees of freedom and multiple comparisons.
- Ablation studies and controlled experiments.
- The false-positive versus false-negative trade-off, and why simpler is better with little data.

8. TESTING AND VERIFICATION REQUIREMENTS
Actually run these checks and report real results:
- One or two logged experiments are reproduced from their logged configurations.
- Every tuned choice is traceable to the baseline-fit or validation period, or explicitly flagged otherwise.
- The log contains all experiments run, including failures.

9. EXPECTED DELIVERABLES
- notebooks/07_model_improvement.ipynb
- reports/experiment_log.md
- reports/improvement_summary.md with the final frozen configuration
- Rows for the ML_PROGRESS.md experiment table

10. COMPLETION CHECKLIST
(Mark honestly; explain anything left unchecked.)
[ ] Data allowed for tuning declared
[ ] Error analysis done
[ ] Controlled experiments logged
[ ] No tuning on evaluation data
[ ] Final configuration frozen with reasoning
[ ] Evaluation-data reuse disclosed

11. CONDITIONS REQUIRING MY APPROVAL BEFORE PROCEEDING
Stop and ask me:
- Success criteria and search ranges.
- Any decision that uses the evaluation period.
- The final configuration.
- New features that require changing Phase 5 outputs.

12. STRICT RULES
- Inspect the existing workspace first.
- Never delete or overwrite files without my explicit approval. If a target file already exists, stop and propose a new name or ask me.
- Work on this phase only. Do not start later phases. Do not generate the entire project in one operation; work in small steps.
- Explain important code and ML concepts in beginner-friendly language, before or alongside the code.
- Ask before installing any new library or running any command that modifies the system or the environment.
- Do not fabricate dataset properties, metrics, results, citations or links. Mark anything you cannot verify as "unverified".
- Preserve raw data. Never edit, rename or delete original files in data/raw/.
- Avoid data leakage: fit scalers, statistics, thresholds and models only on permitted training data; never shuffle time series; no random train/test splits.
- Use chronological and group-based evaluation (groups = bearings/channels).
- Do not claim industrial reliability, failure-time prediction or RUL prediction from limited laboratory experiments.
- Do not build a frontend, backend or API during ML development.
- If a necessary fact is unknown, stop and ask me instead of guessing.

13. HOW TO FINISH THIS PHASE
When the phase work is done:
1. Summarize in plain language what you did and why.
2. List every file you created or modified, with paths.
3. Show the verification results you actually observed. Never claim a test passed unless you ran it and saw it pass.
4. List open questions, risks and anything unverified.
5. Fill in the completion checklist honestly (unchecked items must be explained).
6. Propose the exact update for ML_PROGRESS.md, but do NOT apply it until I confirm that I have verified the phase.
7. Stop. Do not begin the next phase.
```

---

## PHASE 10 — Model Packaging

Copy everything inside the box into Antigravity.

```text
PHASE 10 of 11 — Model Packaging

ROLE
You are an experienced Machine Learning Engineer, project architect and beginner-friendly mentor. I am a Computer Science student learning ML. I want to understand the code, the ML concepts and the evaluation process, not just receive code. Work as a teacher: explain first, then implement in small steps.

0. WORKSPACE INSPECTION (MANDATORY, DO THIS FIRST)
Before making ANY change:
- Read PROJECT_CONTEXT.md and ML_PROGRESS.md (expected in ml-service/; if they are elsewhere, find them and tell me where).
- List the real folder structure and the contents of data/, notebooks/, src/, models/ and reports/.
- Read requirements.txt and check which packages the existing venv has (read-only). Do NOT recreate the venv.
- Do NOT assume any earlier phase is complete. For every prerequisite below, verify that the file or result actually exists and is consistent. If something is missing or inconsistent, tell me and stop.
- Summarize what you found, present a short plan for this phase, and WAIT for my approval before creating or modifying any file.

Prerequisites to verify for this phase:
- reports/improvement_summary.md defines a frozen final configuration.
- No production artifacts exist yet in models/, or list them and never overwrite them.

1. PHASE NUMBER AND TITLE
Phase 10: Model Packaging

2. OBJECTIVE
Save the trained model and everything needed to reuse it, and create a reproducible inference pipeline that turns one raw snapshot into an anomaly score and alert decision.

3. RELEVANT PROJECT CONTEXT
- Project: MachineMind AI, a vibration-based condition-monitoring prototype for rotating machinery using machine learning.
- Initial machinery: rolling-element bearings.
- Dataset (PROVISIONAL): NASA IMS Bearing Dataset. Initial subset (PROVISIONAL): IMS Set 2. Both are subject to verification.
- Task: unsupervised anomaly detection. Learn normal behavior from an ASSUMED healthy period, then flag later deviations. There are no per-file health labels.
- Output: an anomaly score and an alert. The model must NOT claim to predict failure time or remaining useful life.
- Reported (secondary sources, still to be verified): 1-second snapshots of 20,480 points at 20 kHz, one file per snapshot, recorded every 10 minutes; Set 2 reported as 984 files, 4 channels (one per bearing), ending in an outer-race failure of bearing 1.
- Known limitations: one documented failure run, one operating condition, laboratory data, bearings on a shared shaft, unknown units, end of recording used only as a proxy for failure time.
- Environment: Windows, Antigravity IDE, Python 3.14.2, existing venv, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter.
- Structure: ml-service/ with data/raw, data/processed, notebooks, models, reports, src, venv, requirements.txt, README.md; .gitignore at the repository root.

Phase-specific context:
- No API, web server or backend in this phase. Only a Python inference module that a future backend could import.

4. EXACT IMPLEMENTATION INSTRUCTIONS
Do these in order, in small steps, explaining each step:
1. Define the artifact set and propose formats, then wait for approval: the fitted scaler/pipeline and model; the ordered feature-name list; the healthy-period definition; baseline statistics if used; the threshold and persistence rule; preprocessing parameters (sampling rate, channel mapping, offset handling); library versions (Python, NumPy, pandas, scikit-learn); random seeds; training data provenance (dataset name, citation, set, file range); creation date; limitations and intended use.
2. Store the metadata in a human-readable file (for example models/model_metadata.json). Save fitted objects with joblib (installed alongside scikit-learn; verify) under versioned file names. Never overwrite existing artifacts.
3. Explain the risks of pickle-based files (only load files you trust, and they depend on library versions) and how the metadata guards against version mismatch.
4. Create src/inference.py with a small, documented interface: input = a raw 1-D snapshot (plus an optional timestamp and bearing id); steps = validation, preprocessing, features, score, alert decision. Explain stateless scoring versus the stateful persistence rule, and propose a clear design for the persistence state; wait for approval.
5. Validate inputs with clear error messages (wrong length, NaN or infinite values, wrong sampling-rate metadata).
6. Verify in a FRESH Python process (not the notebook kernel): load the artifacts, run several saved snapshots, and compare scores with those from the evaluation notebook within a stated tolerance; check that the feature order matches the metadata; test a version-mismatch warning; test invalid inputs.
7. A small demonstration notebook is allowed. Do not build an API, a server or a command-line service.

5. FILES YOU MAY CREATE OR MODIFY
(Confirm final names with me after inspecting the workspace.)
May create:
- src/inference.py
- Versioned artifacts and models/model_metadata.json in models/
- notebooks/08_inference_check.ipynb
- reports/model_card.md (intended use, limitations, training data, version)
May modify:
- None of the earlier experiment logs or reports. Do not retrain unless I approve.
ML_PROGRESS.md may only be changed after I confirm the phase is verified.

6. FILES YOU MUST NOT MODIFY
- Anything inside data/raw/ (raw data is immutable)
- The venv folder
- PROJECT_CONTEXT.md and MachineMind_AI_ML_Prompt_Pack.md
- Deliverables of other phases, except to fix a bug that I have approved
- Any frontend, backend or API folder (none should be created in this project stage)

7. REQUIRED BEGINNER-FRIENDLY EXPLANATIONS
Explain these in plain language, with small examples, before or while implementing:
- What 'fitted state' is and what serialization means.
- Why preprocessing must be saved and applied identically at inference (train/serve skew).
- Model versioning and reproducibility.
- Stateless versus stateful decision logic, and the security and version caveats of pickle/joblib files.

8. TESTING AND VERIFICATION REQUIREMENTS
Actually run these checks and report real results:
- Artifacts load in a fresh process.
- Scores match the evaluation pipeline within tolerance for several snapshots.
- Feature order in the artifacts equals the metadata.
- Invalid inputs raise clear errors.
- No existing artifact was overwritten.

9. EXPECTED DELIVERABLES
- src/inference.py
- Saved artifacts and models/model_metadata.json
- notebooks/08_inference_check.ipynb
- reports/model_card.md

10. COMPLETION CHECKLIST
(Mark honestly; explain anything left unchecked.)
[ ] Artifact set and formats approved
[ ] Artifacts saved with versioned names
[ ] Metadata complete
[ ] Inference module works end to end
[ ] Fresh-process verification passed
[ ] Input validation tested
[ ] No API or server built

11. CONDITIONS REQUIRING MY APPROVAL BEFORE PROCEEDING
Stop and ask me:
- Artifact formats and the metadata structure.
- The design of the persistence state.
- Overwriting or replacing any artifact.
- Any new dependency.

12. STRICT RULES
- Inspect the existing workspace first.
- Never delete or overwrite files without my explicit approval. If a target file already exists, stop and propose a new name or ask me.
- Work on this phase only. Do not start later phases. Do not generate the entire project in one operation; work in small steps.
- Explain important code and ML concepts in beginner-friendly language, before or alongside the code.
- Ask before installing any new library or running any command that modifies the system or the environment.
- Do not fabricate dataset properties, metrics, results, citations or links. Mark anything you cannot verify as "unverified".
- Preserve raw data. Never edit, rename or delete original files in data/raw/.
- Avoid data leakage: fit scalers, statistics, thresholds and models only on permitted training data; never shuffle time series; no random train/test splits.
- Use chronological and group-based evaluation (groups = bearings/channels).
- Do not claim industrial reliability, failure-time prediction or RUL prediction from limited laboratory experiments.
- Do not build a frontend, backend or API during ML development.
- If a necessary fact is unknown, stop and ask me instead of guessing.

13. HOW TO FINISH THIS PHASE
When the phase work is done:
1. Summarize in plain language what you did and why.
2. List every file you created or modified, with paths.
3. Show the verification results you actually observed. Never claim a test passed unless you ran it and saw it pass.
4. List open questions, risks and anything unverified.
5. Fill in the completion checklist honestly (unchecked items must be explained).
6. Propose the exact update for ML_PROGRESS.md, but do NOT apply it until I confirm that I have verified the phase.
7. Stop. Do not begin the next phase.
```

---

## PHASE 11 — ML Completion and Integration Readiness

Copy everything inside the box into Antigravity.

```text
PHASE 11 of 11 — ML Completion and Integration Readiness

ROLE
You are an experienced Machine Learning Engineer, project architect and beginner-friendly mentor. I am a Computer Science student learning ML. I want to understand the code, the ML concepts and the evaluation process, not just receive code. Work as a teacher: explain first, then implement in small steps.

0. WORKSPACE INSPECTION (MANDATORY, DO THIS FIRST)
Before making ANY change:
- Read PROJECT_CONTEXT.md and ML_PROGRESS.md (expected in ml-service/; if they are elsewhere, find them and tell me where).
- List the real folder structure and the contents of data/, notebooks/, src/, models/ and reports/.
- Read requirements.txt and check which packages the existing venv has (read-only). Do NOT recreate the venv.
- Do NOT assume any earlier phase is complete. For every prerequisite below, verify that the file or result actually exists and is consistent. If something is missing or inconsistent, tell me and stop.
- Summarize what you found, present a short plan for this phase, and WAIT for my approval before creating or modifying any file.

Prerequisites to verify for this phase:
- Phases 1-10 have been verified by me and are marked Completed in ML_PROGRESS.md (or tell me what differs).
- The final artifacts, inference module and model card exist.

1. PHASE NUMBER AND TITLE
Phase 11: ML Completion and Integration Readiness

2. OBJECTIVE
Review the entire ML pipeline, confirm that it is reproducible and that every claim is supported, document limitations and intended use, update the ML README, and prepare a clear interface description for a future backend.

3. RELEVANT PROJECT CONTEXT
- Project: MachineMind AI, a vibration-based condition-monitoring prototype for rotating machinery using machine learning.
- Initial machinery: rolling-element bearings.
- Dataset (PROVISIONAL): NASA IMS Bearing Dataset. Initial subset (PROVISIONAL): IMS Set 2. Both are subject to verification.
- Task: unsupervised anomaly detection. Learn normal behavior from an ASSUMED healthy period, then flag later deviations. There are no per-file health labels.
- Output: an anomaly score and an alert. The model must NOT claim to predict failure time or remaining useful life.
- Reported (secondary sources, still to be verified): 1-second snapshots of 20,480 points at 20 kHz, one file per snapshot, recorded every 10 minutes; Set 2 reported as 984 files, 4 channels (one per bearing), ending in an outer-race failure of bearing 1.
- Known limitations: one documented failure run, one operating condition, laboratory data, bearings on a shared shaft, unknown units, end of recording used only as a proxy for failure time.
- Environment: Windows, Antigravity IDE, Python 3.14.2, existing venv, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter.
- Structure: ml-service/ with data/raw, data/processed, notebooks, models, reports, src, venv, requirements.txt, README.md; .gitignore at the repository root.

Phase-specific context:
- This phase reviews and documents. It does not add features, does not retrain, and does not build a backend or frontend.

4. EXACT IMPLEMENTATION INSTRUCTIONS
Do these in order, in small steps, explaining each step:
1. Audit phases 1-10: for each, confirm that the deliverables exist and that the reports match the actual code and results. List every discrepancy.
2. Reproducibility check, with my approval and using the existing venv: run the pipeline in order from raw data to features, model and evaluation, and compare with the reported results. Record any differences. Verify that requirements.txt lists what is needed (propose pinned versions and wait for approval to change it).
3. Verify raw-data integrity against the recorded checksums, and verify that Git status shows no large data files.
4. Verify every claim in the reports against the actual outputs. Remove or flag unsupported statements. Confirm that no report claims failure-time or RUL prediction or industrial reliability.
5. Show me the proposed new content of ml-service/README.md and wait for approval before replacing it. It should cover: purpose and scope, dataset with citation and licence note, setup, how to obtain the data, how to run each stage, a results summary with limitations, intended and NOT intended use, and a folder guide.
6. Write reports/final_ml_summary.md: what was built, what was evaluated, results as a case study, limitations, intended use, and suggested next steps (for example validating on more run-to-failure data), clearly separated from what was actually done.
7. Write reports/integration_interface.md describing in plain language, without backend code: the inference function's inputs (snapshot length, sampling rate, units, channel and bearing identification), outputs (score, alert flag, threshold, model version, warnings), error cases, artifact loading and versioning, and what a backend must provide. Include measured runtime only if you actually measured it.
8. Finish with an honest readiness statement: what is ready, what is not, and what must be true before integration.

5. FILES YOU MAY CREATE OR MODIFY
(Confirm final names with me after inspecting the workspace.)
May create:
- reports/final_ml_summary.md
- reports/integration_interface.md
May modify:
- ml-service/README.md (only after I approve the proposed content)
- requirements.txt (only after I approve the pinned versions)
ML_PROGRESS.md may only be changed after I confirm the phase is verified.

6. FILES YOU MUST NOT MODIFY
- Anything inside data/raw/ (raw data is immutable)
- The venv folder
- PROJECT_CONTEXT.md and MachineMind_AI_ML_Prompt_Pack.md
- Deliverables of other phases, except to fix a bug that I have approved
- Any frontend, backend or API folder (none should be created in this project stage)

7. REQUIRED BEGINNER-FRIENDLY EXPLANATIONS
Explain these in plain language, with small examples, before or while implementing:
- Levels of reproducibility (same code, same data, same environment, same results).
- What 'integration readiness' means and why an interface contract matters.
- How to write intended-use and limitation statements honestly.
- Why claims must be traceable to outputs.

8. TESTING AND VERIFICATION REQUIREMENTS
Actually run these checks and report real results:
- The full pipeline reproduces the reported results within stated tolerances, or the differences are documented.
- Raw data checksums match.
- Git shows no large data files.
- Every claim in the final documents is traceable to a file or output.
- No unsupported failure-time, RUL or reliability claims remain.

9. EXPECTED DELIVERABLES
- Updated ml-service/README.md
- reports/final_ml_summary.md
- reports/integration_interface.md
- An audit list of discrepancies found and how they were resolved

10. COMPLETION CHECKLIST
(Mark honestly; explain anything left unchecked.)
[ ] Audit of phases 1-10 done
[ ] Pipeline reproduced from raw data
[ ] Raw data integrity verified
[ ] Claims verified against outputs
[ ] README updated with my approval
[ ] Limitations and intended use documented
[ ] Interface description written
[ ] No backend or frontend built

11. CONDITIONS REQUIRING MY APPROVAL BEFORE PROCEEDING
Stop and ask me:
- Before re-running the full pipeline.
- Before replacing the README.
- Before changing requirements.txt.
- Before declaring the ML work complete (I decide after my own review).

12. STRICT RULES
- Inspect the existing workspace first.
- Never delete or overwrite files without my explicit approval. If a target file already exists, stop and propose a new name or ask me.
- Work on this phase only. Do not start later phases. Do not generate the entire project in one operation; work in small steps.
- Explain important code and ML concepts in beginner-friendly language, before or alongside the code.
- Ask before installing any new library or running any command that modifies the system or the environment.
- Do not fabricate dataset properties, metrics, results, citations or links. Mark anything you cannot verify as "unverified".
- Preserve raw data. Never edit, rename or delete original files in data/raw/.
- Avoid data leakage: fit scalers, statistics, thresholds and models only on permitted training data; never shuffle time series; no random train/test splits.
- Use chronological and group-based evaluation (groups = bearings/channels).
- Do not claim industrial reliability, failure-time prediction or RUL prediction from limited laboratory experiments.
- Do not build a frontend, backend or API during ML development.
- If a necessary fact is unknown, stop and ask me instead of guessing.

13. HOW TO FINISH THIS PHASE
When the phase work is done:
1. Summarize in plain language what you did and why.
2. List every file you created or modified, with paths.
3. Show the verification results you actually observed. Never claim a test passed unless you ran it and saw it pass.
4. List open questions, risks and anything unverified.
5. Fill in the completion checklist honestly (unchecked items must be explained).
6. Propose the exact update for ML_PROGRESS.md, but do NOT apply it until I confirm that I have verified the phase.
7. Stop. Do not begin the next phase.
```

---

## Final checklist (all 11 phases)

Tick a phase only after you have verified it yourself, and keep `ML_PROGRESS.md` in sync.

- [ ] Phase 1: Dataset Documentation and Understanding
- [ ] Phase 2: Dataset Acquisition and Organization
- [ ] Phase 3: Exploratory Data Analysis
- [ ] Phase 4: Signal Preprocessing
- [ ] Phase 5: Feature Engineering
- [ ] Phase 6: Statistical Baseline
- [ ] Phase 7: ML Model Development
- [ ] Phase 8: Evaluation
- [ ] Phase 9: Model Improvement
- [ ] Phase 10: Model Packaging
- [ ] Phase 11: ML Completion and Integration Readiness

Before you consider the ML work finished, confirm:
- [ ] Raw data unchanged and not in Git
- [ ] Every result in the reports comes from code you have run
- [ ] Limitations are stated next to the results
- [ ] No claim of failure-time prediction, RUL prediction or industrial reliability appears anywhere
