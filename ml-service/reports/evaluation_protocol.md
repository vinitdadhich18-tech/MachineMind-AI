# Phase 8: Evaluation Protocol — MachineMind AI

**Project:** MachineMind AI  
**Dataset:** NASA IMS Bearing Dataset (Subset: Set 2, 984 snapshots)  
**Task:** Unsupervised Vibration Anomaly Detection  
**Phase:** Phase 8 — Model Evaluation & Comparison  
**Document Type:** Frozen Pre-Execution Evaluation Protocol (Final Revised)  
**Author:** MachineMind AI Team & Mentor  
**Date:** September 29, 2026  

---

## A. Project Objective

The primary objective of Phase 8 is to perform a rigorous, leakage-free evaluation and comparative analysis across three unsupervised anomaly detection strategies:
1. **Statistical Baseline:** Non-ML composite Z-score metrics (`score_rms_z` as primary; `score_rms_m`, `score_max_z`, `score_max_m` as secondary comparisons).
2. **Isolation Forest (iForest):** Tree-partitioning novelty detection pipelines (`score_iforest_ch1` to `score_iforest_ch4`).
3. **PCA Reconstruction Error:** Subspace reconstruction error pipelines (`score_pca_ch1` to `score_pca_ch4`).

### Operational Task Framing
The models operate in **unsupervised novelty detection mode**. The goal is to learn the boundary of normal operational behavior from an initial fitting period assumed to be healthy (`0..159`), calibrate alert thresholds on a validation period (`160..199`), and flag subsequent statistical deviations during sequential evaluation (`200..983`).

### Strict Scope Boundaries
- This protocol evaluates statistical anomaly scores and alert behaviors on laboratory test data.
- The models **do not** predict remaining useful life (RUL) or exact failure timestamps.
- No claim of operationally verified maintenance lead time or industrial reliability is made.

---

## B. Dataset and Limitations

- **Dataset Identifier:** NASA IMS Bearing Dataset, Set 2 (acquired and verified in Phase 2).
- **Temporal Scope:** 984 chronological snapshots recorded at 10-minute intervals (spanning `2004-02-12 10:32:39` to `2004-02-19 06:22:39`).
- **Sensor Architecture:** 4 accelerometer channels (`ch1`, `ch2`, `ch3`, `ch4`) monitoring 4 bearings mounted on a single shared rotating shaft (~2000 RPM, 6000 lb radial load).
- **Lack of Per-Snapshot Health Labels:** The dataset contains no ground-truth health annotations or physical inspection logs for individual snapshots.
- **Unconfirmed Failure Onset:** The dataset documents an outer-race defect on Bearing 1 at experiment termination, but the exact physical onset timestamp of micro-structural degradation is unknown.
- **Single Run-to-Failure Case Study:** Results represent a single-case study ($n=1$ failure trajectory). Results cannot establish statistical generalizability across different machines, loads, speeds, or bearing geometries.

---

## C. Chronological Data Partitions

To prevent temporal data leakage, evaluation strictly enforces the three pre-established chronological partitions:

| Partition Name | Snapshot Index Range | File Count ($N$) | Approx. Time Duration | Primary Purpose & Scoping Rules |
|---|---|---|---|---|
| **Fit Period** | `0` to `159` | 160 | ~26.7 hours | Model fitting, baseline parameter estimation ($\mu, \sigma, \text{IQR}$), and scaler transformation (`RobustScaler`). Assumed healthy baseline reference. |
| **Validation Period** | `160` to `199` | 40 | ~6.7 hours | Threshold calibration and percentile estimation ($P_{99}, \mu + 3\sigma$). Used strictly for reference line selection. |
| **Evaluation Period** | `200` to `983` | 784 | ~130.7 hours | Unseen sequential evaluation. Strictly held back until final scoring. |

### Chronological Preservation Rule
- **No Random Shuffling:** Time-series snapshots must remain strictly ordered by file timestamp.
- **No K-Fold Cross-Validation:** Standard random K-Fold CV is strictly prohibited as it destroys temporal autocorrelation and causes bidirectional data leakage.

---

## D. Evaluation Methods & Score Definitions

Evaluation compares three distinct algorithmic families. All score functions obey the verified sign convention: **HIGHER SCORE = MORE ANOMALOUS**.

```
                           +-----------------------------------+
                           |  Feature Extractor (36 Features)  |
                           +-----------------------------------+
                                             |
            +--------------------------------+--------------------------------+
            |                                |                                |
            v                                v                                v
   [ Statistical Baseline ]        [ Isolation Forest ]            [ PCA Reconstruction ]
   - RMS Z-Score Composite         - Tree-partitioning             - Subspace MSE Error
   - S_RMS = sqrt(mean(Z_i^2))     - S_iForest = 0.5 - dec_func    - S_PCA = (1/d)||X - X_rec||^2
```

### 1. Statistical Baseline (`src/baseline.py`) Scope Categorization
Phase 6 produces four composite anomaly metrics. Their roles in Phase 8 evaluation are explicitly categorized:

- **PRIMARY Baseline Benchmark (`score_rms_z`):**  
  RMS Composite Z-Score across all 36 feature dimensions:
  $$S_{\text{RMS\_Z}} = \sqrt{\frac{1}{K} \sum_{i=1}^K \left(\frac{x_i - \mu_{\text{fit}, i}}{\sigma_{\text{guarded}, i}}\right)^2}$$
  *Role:* Primary non-ML benchmark metric evaluated against threshold $T_{\text{baseline}} = 3.0$. Measures total standardized system energy deviation.
- **SECONDARY Robust Scale Benchmark (`score_rms_m`):**  
  RMS Composite Modified Z-Score using median/MAD scaling.  
  *Role:* Secondary benchmark comparison to verify robustness against baseline outliers ($r > 0.999$ correlation with `score_rms_z`).
- **SECONDARY Single-Feature Indicators (`score_max_z`, `score_max_m`):**  
  Maximum absolute Z-score or Modified Z-score across features.  
  *Role:* Secondary diagnostic indicators for detecting isolated feature spikes. Excluded from primary baseline metric comparison due to sensitivity to narrow-IQR noise spikes.

### 2. Isolation Forest Pipeline (`src/models.py`)
- **Model:** `IsolationForest(n_estimators=100, random_state=42)` bundled per channel with `RobustScaler`.
- **Score Inversion:** Standardized anomaly score:
  $$S_{\text{iForest}} = 0.5 - \text{decision\_function}(X_{\text{scaled}})$$
- **Score Scale:** Normal baseline scores average $\sim 0.40 - 0.45$; anomalous scores increase toward $> 0.55 - 0.75$.

### 3. PCA Reconstruction Error Pipeline (`src/models.py`)
- **Model:** `PCA(n_components=2, random_state=42)` bundled per channel with `RobustScaler`.
- **Score Calculation:** Mean Squared Error between scaled input $X_{\text{scaled}}$ and reconstructed projection $\hat{X}_{\text{scaled}}$:
  $$S_{\text{PCA}} = \frac{1}{d} \sum_{j=1}^d (x_{j, \text{scaled}} - \hat{x}_{j, \text{scaled}})^2$$
- **Score Scale:** Strictly non-negative. Normal baseline scores average $\sim 0.20 - 0.40$; anomalous scores elevate to $> 1.0 - 100.0+$.

---

## E. Primary Threshold Selection Rules

All anomaly detection thresholds must be **calibrated on validation data** (snapshots `160..199`) or Phase 6 baseline specifications and **frozen** prior to scoring the evaluation period (snapshots `200..983`).

### 1. Primary Threshold Configurations

| Method / Model | Role in Evaluation | Primary Threshold Rule | Calculation Source & Formula | Frozen Threshold Value |
|---|---|---|---|---|
| **Baseline (`score_rms_z`)** | Primary Non-ML Baseline | Standard Reference Line $T=3.0$ | Fixed Phase 6 benchmark ($T_{\text{baseline}} = 3.0$). Represents 3x unit RMS energy deviation. | $T = 3.0000$ |
| **Baseline (`score_rms_m`)** | Secondary Baseline | Standard Reference Line $T=3.0$ | Fixed Phase 6 robust benchmark ($T_{\text{baseline}} = 3.0$). | $T = 3.0000$ |
| **Baseline (`score_max_z`)** | Secondary Indicator | Fixed Reference Line $T=3.0$ | Fixed Phase 6 max benchmark ($T_{\text{baseline}} = 3.0$). | $T = 3.0000$ |
| **Isolation Forest (`ch1`)** | Primary Model (`ch1`) | Validation $P_{99}$ Percentile | 99th percentile of validation scores (`160..199`, $N=40$) | $T = 0.5310$ |
| **Isolation Forest (`ch2`)** | Primary Model (`ch2`) | Validation $P_{99}$ Percentile | 99th percentile of validation scores (`160..199`, $N=40$) | $T = 0.5184$ |
| **Isolation Forest (`ch3`)** | Primary Model (`ch3`) | Validation $P_{99}$ Percentile | 99th percentile of validation scores (`160..199`, $N=40$) | $T = 0.5990$ |
| **Isolation Forest (`ch4`)** | Primary Model (`ch4`) | Validation $P_{99}$ Percentile | 99th percentile of validation scores (`160..199`, $N=40$) | $T = 0.6247$ |
| **PCA Error (`ch1`)** | Primary Model (`ch1`) | Validation $P_{99}$ Percentile | 99th percentile of validation scores (`160..199`, $N=40$) | $T = 1.0721$ |
| **PCA Error (`ch2`)** | Primary Model (`ch2`) | Validation $P_{99}$ Percentile | 99th percentile of validation scores (`160..199`, $N=40$) | $T = 0.8657$ |
| **PCA Error (`ch3`)** | Primary Model (`ch3`) | Validation $P_{99}$ Percentile | 99th percentile of validation scores (`160..199`, $N=40$) | $T = 0.6793$ |
| **PCA Error (`ch4`)** | Primary Model (`ch4`) | Validation $P_{99}$ Percentile | 99th percentile of validation scores (`160..199`, $N=40$) | $T = 1.3281$ |

### 2. Alert Generation Rule
An instantaneous threshold exceedance occurs at snapshot $t$ if the score exceeds the frozen threshold:
$$\text{Exceedance}_t = \begin{cases} 1 & \text{if } S_t > T \\ 0 & \text{otherwise} \end{cases}$$

---

## F. Persistence Filtering & Precise Alert Timing Definitions

To prevent isolated single-snapshot noise spikes from triggering false alarms, evaluation enforces **persistence filtering** ($k$-consecutive exceedances).

### 1. Precise Timestamp & Index Definitions
- **First Exceedance Index ($t_{\text{exceed}}$):**  
  The snapshot index $t \ge 200$ where a single score first exceeds threshold ($S_t > T$, instantaneous $k=1$).
- **Start Index of Sustained Alert ($t_{\text{start}}$):**  
  The snapshot index $t \ge 200$ where a continuous run of $\ge k$ consecutive snapshots ($S_{\tau} > T$ for $\tau \in [t, t+k-1]$) begins.
- **Alert Confirmation Index ($t_{\text{confirm}}$):**  
  The exact snapshot index at which a causal online monitoring system can **confirm** that $k$ consecutive exceedances have occurred:
  $$t_{\text{confirm}} = t_{\text{start}} + k - 1$$
- **Alert Confirmation Timestamp ($timestamp_{\text{confirm}}$):**  
  The actual recorded timestamp corresponding to snapshot index $t_{\text{confirm}}$.
- **Proxy Duration to Recording End ($\Delta t_{\text{proxy}}$):**  
  The time difference from $timestamp_{\text{confirm}}$ to the final dataset snapshot timestamp ($t_{\text{final}} = \text{2004-02-19 06:22:39}$):
  $$\Delta t_{\text{proxy}} = t_{\text{final}} - timestamp_{\text{confirm}}$$

### 2. Concrete Timeline Examples for $k = 1, 3, 5$

Consider a timeline of threshold exceedance flags $\mathbb{I}(S_t > T)$:

```
Snapshot Index:   100  101  102  103  104  105  106  107  108  109  110
Exceedance Flag:    0    1    1    1    1    1    0    0    1    0    0
                         ^
                  t_exceed = 101 (First single exceedance)
```

- **Case 1: $k = 1$ (Instantaneous Exceedance, No Filter)**
  - First Exceedance Index ($t_{\text{exceed}}$): `101`
  - Start Index ($t_{\text{start}}$): `101`
  - Confirmation Index ($t_{\text{confirm}}$): $101 + 1 - 1 = \mathbf{101}$ (Confirmed instantly at snapshot 101).
  
- **Case 2: $k = 3$ (Short Persistence Filter)**
  - First Exceedance Index ($t_{\text{exceed}}$): `101`
  - Start Index ($t_{\text{start}}$): `101` (Run of 5 exceedances `101..105` satisfies $k \ge 3$)
  - Confirmation Index ($t_{\text{confirm}}$): $101 + 3 - 1 = \mathbf{103}$ (System confirms alert at snapshot 103, 20 minutes after $t_{\text{start}}$).

- **Case 3: $k = 5$ (Robust Persistence Filter)**
  - First Exceedance Index ($t_{\text{exceed}}$): `101`
  - Start Index ($t_{\text{start}}$): `101` (Run of 5 exceedances `101..105` satisfies $k \ge 5$)
  - Confirmation Index ($t_{\text{confirm}}$): $101 + 5 - 1 = \mathbf{105}$ (System confirms alert at snapshot 105, 40 minutes after $t_{\text{start}}$).

---

## G. Sustained-Alert Event Counting Logic

It is essential to distinguish between **event counts** (episodes) and **flagged snapshot counts** (individual sample points).

```
Snapshot Exceedances: [0,  1, 1, 1, 1, 1,  0, 0,  1, 1, 1,  0]
                      |<- Event 1 (len 5)->|      |<-Event 2->|
```

### 1. Sustained-Alert Event Counting Rules
- **Definition of an Event:** A continuous contiguous sequence of snapshots where every snapshot exceeds threshold ($S_{\tau} > T$) for a length $L \ge k$ constitutes **ONE distinct sustained-alert event** (episode).
- **Run Interruption:** If the score drops below threshold ($S_{\tau} \le T$) for one or more snapshots, the current event terminates.
- **Subsequent Events:** If the score later exceeds threshold again for $\ge k$ consecutive snapshots, a **new distinct sustained-alert event** is logged.

### 2. Distinction Between Metrics
1. **Sustained-Alert Event Count ($N_{\text{sustained\_events}}$):**  
   Total number of distinct continuous alert episodes of length $L \ge k$ in a specified partition.
2. **Flagged Snapshot Count ($N_{\text{flagged\_snapshots}}$):**  
   Total count of individual snapshot points that belong to any sustained alert episode.
3. **Effect of Persistence $k$:**
   - Increasing $k$ **reduces or maintains** $N_{\text{sustained\_events}}$ by filtering out isolated short runs ($L < k$).
   - Short runs of length $L < k$ contribute zero to $N_{\text{flagged\_snapshots}}$ under persistence $k$.

---

## H. Quantitative Evaluation Metrics

Metrics are categorized and explicitly mapped to their persistence applicability:

### 1. Threshold Exceedance & Alert Metrics
- **Raw Threshold-Exceedance Count ($N_{\text{exceed}}$):** Total count of individual snapshots in a partition where $S_t > T$. *(Persistence $k=1$ only)*.
- **Sustained-Alert Event Count ($N_{\text{sustained\_events}}$):** Total number of distinct continuous alert episodes of length $L \ge k$. *(Persistence $k \ge 1$ applies)*.
- **Evaluation Flagged Fraction ($F_{\text{eval\_flagged}}$):** Fraction of snapshots in the evaluation period (`200..983`, $N=784$) with $S_t > T$:
  $$F_{\text{eval\_flagged}} = \frac{N_{\text{exceed, eval}}}{784}$$
- **Post-Confirmation Flagged Fraction ($F_{\text{post\_confirm}}$):** Fraction of snapshots from $t_{\text{confirm}}$ to snapshot 983 that maintain $S_t > T$:
  $$F_{\text{post\_confirm}} = \frac{\sum_{\tau=t_{\text{confirm}}}^{983} \mathbb{I}(S_{\tau} > T)}{984 - t_{\text{confirm}}}$$

### 2. Assumed Healthy Period & False-Alarm Metrics
The initial partition (`0..199`, $N=200$ snapshots spanning fit and validation) is treated as an **assumed healthy reference period**.

- **Raw False-Alarm Count ($N_{\text{FA, raw}}$):** Total individual snapshots in `0..199` with $S_t > T$. *(Persistence $k=1$)*.
- **Sustained False-Alarm Count ($N_{\text{FA, sustained}}$):** Total snapshots in `0..199` that belong to a sustained run of length $L \ge k$. *(Persistence $k \ge 1$ applies)*.
- **False-Alarm Rate ($R_{\text{FA}}$):** Fraction of assumed healthy snapshots flagged:
  $$R_{\text{FA, raw}} = \frac{N_{\text{FA, raw}}}{200}, \quad R_{\text{FA, sustained}} = \frac{N_{\text{FA, sustained}}}{200}$$

> **Interpretation Note:** Alerts in snapshots `0..199` are classified as false alarms *strictly relative to the baseline health assumption*. If early physical micro-wear occurred, these may not be physical false alarms, but under our protocol they are formally counted as baseline false alarms.

---

## I. Pre-Declared Sensitivity Analysis Grid

Sensitivity analysis tests the stability of model results across pre-declared threshold and persistence settings. Sensitivity experiments **must not** be used post-hoc to retroactively select a favorable result.

### 1. Sensitivity Threshold Variants (Validation Partition `160..199`)
- **Variant 1 ($\mu + 3\sigma$):** $\mu_{\text{val}} + 3 \cdot \sigma_{\text{val}}$
- **Variant 2 ($\mu + 4\sigma$):** $\mu_{\text{val}} + 4 \cdot \sigma_{\text{val}}$
- **Variant 3 ($\mu + 6\sigma$):** $\mu_{\text{val}} + 6 \cdot \sigma_{\text{val}}$
- **Variant 4 ($P_{99}$):** 99th Percentile of validation scores (Primary for iForest/PCA)
- **Variant 5 ($P_{99.5}$):** 99.5th Percentile of validation scores

### 2. Persistence Values ($k$)
- **$k = 1$:** Instantaneous exceedance (no filter).
- **$k = 3$:** Short persistence filter (20 minutes continuous exceedance).
- **$k = 5$:** Robust persistence filter (40 minutes continuous exceedance).

---

## J. Multi-Channel & Aggregate Alert Definitions

Primary evaluation reporting is conducted **per channel** (`ch1` to `ch4`).

### Logical OR Shaft Aggregate Alert Logic
If a multi-channel shaft-level alert is evaluated, it uses strict **Logical OR Aggregation**:
1. **Channel Alert Condition:** Channel $c$ triggers a sustained alert at snapshot $t$ if $k$ consecutive snapshots on channel $c$ satisfy $S_{c, \tau} > T_c$.
2. **Shaft Aggregate Status:** The shaft aggregate alert is active at snapshot $t$ if **at least one channel** has satisfied its sustained alert condition.
3. **Shaft Confirmation Index ($t_{\text{confirm, shaft}}$):**
   $$t_{\text{confirm, shaft}} = \min_{c \in \{\text{ch1..ch4}\}} t_{\text{confirm, } c}$$

> **Strict Rule:** No synthetic channel score averaging (e.g. $\text{mean}(S_{\text{ch1}} \dots S_{\text{ch4}})$) is permitted, as spatial averaging dilutes single-bearing anomalies.

---

## K. Strict Data Leakage Prevention Rules

The evaluation implementation (`src/evaluation.py` and `06_evaluation.ipynb`) must programmatically enforce the following strict boundaries:

1. **No Evaluation Fitting:** Scalers (`RobustScaler`), statistical parameters ($\mu, \sigma$), PCA projections, and Isolation Forest trees must NEVER fit or transform using snapshots `200..983`.
2. **No Evaluation Threshold Tuning:** Thresholds must be calculated using validation snapshots `160..199` ONLY (or Phase 6 baseline specification).
3. **No Chronological Shuffling:** Time series sequence ordering must be strictly preserved.
4. **No Alignment Optimization:** Thresholds or persistence parameters must NOT be selected by observing where alerts land relative to the final failure snapshot.
5. **No Look-Ahead Filtering:** Sustained alert determination at snapshot $t$ must only use information available up to $t+k-1$ (causal online filtering).

---

## L. Scientific Limitations & Interpretation Safeguards

1. **$n=1$ Case Study Limit:** Results describe a single run-to-failure trajectory under constant laboratory speed and load.
2. **No Ground-Truth Onset:** Precision, Recall, Receiver Operating Characteristic (ROC), and Area Under Curve (AUC) metrics cannot be reliably calculated because there are no true binary health labels per snapshot.
3. **Shared Shaft Mechanical Coupling:** All 4 bearings share one shaft. Anomaly score elevations in channels 2, 3, and 4 in the second half of the experiment represent plausible structural vibration transfer across the shaft, not confirmed independent bearing defects.
4. **Statistical Anomaly vs Physical Damage:** An anomaly score elevation indicates statistical deviation from baseline vibration patterns, not a confirmed physical defect mechanism.

---

## M. Explicitly Disallowed Conclusions

The final evaluation report (`reports/evaluation_report.md`) must **NEVER** state or imply any of the following:

- ❌ **"The model predicts remaining useful life (RUL) or exact time-to-failure."**
- ❌ **"Snapshot X is the true physical failure onset of Bearing 1."**
- ❌ **"The model achieves X% precision / recall / AUC on true failure labels."**
- ❌ **"Isolation Forest / PCA is proven superior for industrial bearing monitoring in general."**
- ❌ **"The model provides X hours of verified maintenance lead time."**
- ❌ **"Channels 2, 3, and 4 failed independently at snapshot Y."**

---

## N. Protocol Status & Next Steps

This revised protocol is **frozen**. No evaluation metrics will be computed and no code will be modified until explicit approval of this document is granted.

**Next Immediate Step Upon Approval:** Implement modular metric calculation functions in `src/evaluation.py` and execute unit tests in `src/test_evaluation.py`.
