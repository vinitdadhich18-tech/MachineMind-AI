# NASA IMS Bearing Dataset — RUL Target Definition & Analysis Report

**Phase 5 Deliverable**  
**Date:** 2026-10-01 06:51 UTC  
**Git Commit:** `8759bb2a2af50453b02789566f1d9aec0ae561e8`

---

## 1. RUL Label Derivation Context

The NASA IMS Bearing Set 2 dataset is a **run-to-failure historical research dataset** comprising 984 snapshots recorded at ~10-minute intervals. The dataset contains **no explicit per-snapshot RUL labels**.

Therefore, remaining useful life (RUL) targets must be mathematically derived from the chronological snapshot index $t \in [0, 983]$.

---

## 2. Target Formulations Evaluated

### Option A: Un-capped Linear RUL
$$\text{RUL}_{\text{linear}}(t) = (N - 1) - t = 983 - t$$
- **Unit:** Snapshots remaining until final failure.
- **Physical Assumption:** Assumes steady, uniform wear from snapshot 0 to failure at snapshot 983.
- **Limitation:** Physically unrealistic for early life (snapshots 0–500) where bearing vibration signatures remain baseline healthy and virtually indistinguishable.

### Option B: Capped Piecewise-Linear RUL (Selected)
$$\text{RUL}_{\text{capped}}(t) = \min((N - 1) - t, \text{cap}) = \min(983 - t, 400)$$
- **Unit:** Snapshots remaining (capped at 400 snapshots $\approx 66.6$ hours).
- **Physical Rationale:** Standard PHM (Prognostics and Health Management) best practice. Reflects that health degradation is only observable once initial wear begins. Prevents the model from attempting to distinguish healthy snapshot 10 from healthy snapshot 300.

---

## 3. Target-Range Extrapolation Analysis (Critical ML Limitation)

Tree-based ensemble models (including XGBoost) split feature space using axis-aligned decision boundaries. They **cannot extrapolate target predictions outside the range of $Y$ values seen during training**.

### Empirical Target Ranges Across Chronological Split

| Split Region | Snapshot Indices | Linear RUL Target Range | Capped RUL (400) Target Range |
|---|---|---|---|
| **Train Set** | 0 – 600 | **[383.0, 983.0]** | **[383.0, 400.0]** |
| **Validation Set** | 601 – 750 | [233.0, 382.0] | [233.0, 382.0] |
| **Test Set** | 751 – 983 | **[0.0, 232.0]** | **[0.0, 232.0]** |

### Critical Finding:
When splitting chronologically:
1. Under **Linear RUL**, the training set target range is $[383, 983]$, while the test set target range is $[0, 232]$.
2. The test set targets $[0, 232]$ lie **entirely below** the minimum target seen in training ($\min(Y_{train}) = 383$).
3. Consequently, an un-capped model trained only on early snapshots $0–600$ cannot predict RUL values near 0 at failure, producing a structural prediction floor around $\approx 383$.
4. **Capped RUL (400)** bounds the early health target to 400, reducing the range discrepancy and aligning with standard PHM practice.

---

## 4. Final Target Decision

- **Selected Target:** `capped_rul_400`
- **Formula:** $\text{RUL}(t) = \min(983 - t, 400)$
- **Unit:** Snapshots ($\approx 10$ min / snapshot)
- **Selection Rationale:** Chosen based strictly on physical plausibility and training/validation target distribution analysis, without inspecting test set predictions.
