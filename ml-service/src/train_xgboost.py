"""
MachineMind AI — XGBoost Supervised RUL Training & Evaluation Module
Phase 5: Supervised Dataset + XGBoost Training Pipeline

Supervised predictive maintenance pipeline:
- Loads canonical 36-feature matrix from raw NASA IMS Set 2 dataset
- Evaluates linear RUL vs capped RUL target definitions and target-range extrapolation limitations
- Enforces chronological train (0-600), val (601-750), test (751-983) splits
- Runs automated leakage audit
- Fits scaler on training set only
- Evaluates Mean and LinearRegression baselines
- Trains XGBoost Regressor with early stopping on validation set
- Evaluates metrics (RMSE, MAE, R²) on Train, Validation, and Test (Test evaluated ONCE after freeze)
- Persists versioned artifact bundle (xgboost_rul_model.json, xgboost_scaler.joblib, xgboost_model_metadata.json)
- Generates comprehensive markdown documentation reports
"""

import os
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from src.canonical_feature_pipeline import CANONICAL_FEATURE_NAMES, FeatureScalerAdapter
from src.build_feature_matrix import load_or_build_feature_matrix
from src.leakage_audit import run_full_leakage_audit

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def get_git_commit_hash() -> str:
    """Returns current git commit hash or 'unknown' if git fails."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_matrix_hash(df_matrix: pd.DataFrame) -> str:
    """Computes SHA256 fingerprint of the feature values array."""
    feature_bytes = df_matrix[CANONICAL_FEATURE_NAMES].values.tobytes()
    return hashlib.sha256(feature_bytes).hexdigest()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes RMSE, MAE, R2 evaluation metrics."""
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4)
    }


def train_and_evaluate_xgboost(
    target_column: str = "capped_rul_400",
    train_range: Tuple[int, int] = (0, 600),
    val_range: Tuple[int, int] = (601, 750),
    test_range: Tuple[int, int] = (751, 983),
    random_seed: int = 42,
    scaler_type: str = "robust",
    xgb_params: Optional[Dict[str, Any]] = None,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR
) -> Dict[str, Any]:
    """
    Executes full Phase 5 training, evaluation, artifact export, and documentation pipeline.
    """
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load feature matrix (984 rows)
    df_matrix = load_or_build_feature_matrix()
    matrix_hash = compute_matrix_hash(df_matrix)

    # 2. Run automated leakage audit
    leakage_report = run_full_leakage_audit(df_matrix, CANONICAL_FEATURE_NAMES)
    if not leakage_report["all_passed"]:
        raise RuntimeError(f"Temporal leakage audit failed: {leakage_report}")

    # 3. Prepare train / val / test arrays
    train_slice = range(train_range[0], train_range[1] + 1)
    val_slice = range(val_range[0], val_range[1] + 1)
    test_slice = range(test_range[0], test_range[1] + 1)

    X_train_raw = df_matrix.loc[train_slice, CANONICAL_FEATURE_NAMES].values
    y_train = df_matrix.loc[train_slice, target_column].values

    X_val_raw = df_matrix.loc[val_slice, CANONICAL_FEATURE_NAMES].values
    y_val = df_matrix.loc[val_slice, target_column].values

    X_test_raw = df_matrix.loc[test_slice, CANONICAL_FEATURE_NAMES].values
    y_test = df_matrix.loc[test_slice, target_column].values

    # Target range extrapolation diagnostics
    target_ranges = {
        "linear_rul": {
            "train": [float(df_matrix.loc[train_slice, "linear_rul"].min()), float(df_matrix.loc[train_slice, "linear_rul"].max())],
            "val": [float(df_matrix.loc[val_slice, "linear_rul"].min()), float(df_matrix.loc[val_slice, "linear_rul"].max())],
            "test": [float(df_matrix.loc[test_slice, "linear_rul"].min()), float(df_matrix.loc[test_slice, "linear_rul"].max())]
        },
        target_column: {
            "train": [float(df_matrix.loc[train_slice, target_column].min()), float(df_matrix.loc[train_slice, target_column].max())],
            "val": [float(df_matrix.loc[val_slice, target_column].min()), float(df_matrix.loc[val_slice, target_column].max())],
            "test": [float(df_matrix.loc[test_slice, target_column].min()), float(df_matrix.loc[test_slice, target_column].max())]
        }
    }

    # 4. Fit Scaler strictly on X_train
    scaler = FeatureScalerAdapter(scaler_type=scaler_type)
    scaler.fit(X_train_raw)

    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # 5. Baseline Models Evaluation
    # 5a. Mean Predictor Baseline
    y_train_mean = float(np.mean(y_train))
    mean_baseline_train = calculate_metrics(y_train, np.full_like(y_train, y_train_mean))
    mean_baseline_val = calculate_metrics(y_val, np.full_like(y_val, y_train_mean))
    mean_baseline_test = calculate_metrics(y_test, np.full_like(y_test, y_train_mean))

    # 5b. Linear Regression Baseline
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_train_pred = lin_reg.predict(X_train)
    lin_val_pred = lin_reg.predict(X_val)
    lin_test_pred = lin_reg.predict(X_test)
    lin_baseline_train = calculate_metrics(y_train, lin_train_pred)
    lin_baseline_val = calculate_metrics(y_val, lin_val_pred)
    lin_baseline_test = calculate_metrics(y_test, lin_test_pred)

    # 6. XGBoost Model Training
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": random_seed,
            "n_jobs": 1,
            "early_stopping_rounds": 20
        }

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    xgb_train_pred = model.predict(X_train)
    xgb_val_pred = model.predict(X_val)
    # Test set evaluated ONCE after freezing hyperparameters
    xgb_test_pred = model.predict(X_test)

    xgb_train_metrics = calculate_metrics(y_train, xgb_train_pred)
    xgb_val_metrics = calculate_metrics(y_val, xgb_val_pred)
    xgb_test_metrics = calculate_metrics(y_test, xgb_test_pred)

    # Feature Importance (Gain & Weight)
    importance_gain = model.get_booster().get_score(importance_type="gain")
    # Map f0, f1... to feature names
    feature_importance = {}
    for idx, f_name in enumerate(CANONICAL_FEATURE_NAMES):
        f_key = f"f{idx}"
        feature_importance[f_name] = round(float(importance_gain.get(f_key, 0.0)), 4)

    # Sort feature importance descending
    sorted_feature_importance = dict(
        sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    )

    # 7. Persist Versioned Artifact Bundle
    model_json_path = models_dir / "xgboost_rul_model.json"
    model_joblib_path = models_dir / "xgboost_rul_model.joblib"
    scaler_path = models_dir / "xgboost_scaler.joblib"
    metadata_path = models_dir / "xgboost_model_metadata.json"

    # Save XGBoost model native JSON & Joblib formats
    model.save_model(model_json_path)
    joblib.dump(model, model_joblib_path)

    # Save Scaler
    joblib.dump(scaler, scaler_path)

    # Build Metadata Object
    metadata = {
        "model_name": "MachineMind_XGBoost_RUL_Regressor",
        "model_version": "1.0.0",
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "data_fingerprint": matrix_hash,
        "raw_snapshot_count": len(df_matrix),
        "split_definition": {
            "train": list(train_range),
            "val": list(val_range),
            "test": list(test_range)
        },
        "target_definition": {
            "target_column": target_column,
            "formula": "min(983 - t, 400)",
            "cap_value": 400,
            "unit": "snapshots (~10 min per snapshot)"
        },
        "target_ranges": target_ranges,
        "feature_schema": CANONICAL_FEATURE_NAMES,
        "feature_count": len(CANONICAL_FEATURE_NAMES),
        "hyperparameters": xgb_params,
        "best_iteration": int(getattr(model, "best_iteration", xgb_params["n_estimators"])),
        "library_versions": {
            "python": os.sys.version.split()[0],
            "xgboost": xgb.__version__,
            "scikit-learn": joblib.__import__("sklearn").__version__ if hasattr(joblib, "__import__") else __import__("sklearn").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__
        },
        "metrics": {
            "mean_baseline": {
                "train": mean_baseline_train,
                "val": mean_baseline_val,
                "test": mean_baseline_test
            },
            "linear_regression_baseline": {
                "train": lin_baseline_train,
                "val": lin_baseline_val,
                "test": lin_baseline_test
            },
            "xgboost": {
                "train": xgb_train_metrics,
                "val": xgb_val_metrics,
                "test": xgb_test_metrics
            }
        },
        "leakage_audit_summary": leakage_report,
        "top_features_by_gain": dict(list(sorted_feature_importance.items())[:10])
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 8. Generate Documentation Reports
    generate_target_definition_report(reports_dir / "rul_target_definition.md", metadata)
    generate_training_report(reports_dir / "xgboost_training_report.md", metadata, sorted_feature_importance)

    return metadata


def generate_target_definition_report(output_file: Path, metadata: Dict[str, Any]):
    """Generates `ml-service/reports/rul_target_definition.md`."""
    target_info = metadata["target_definition"]
    ranges = metadata["target_ranges"]

    doc = f"""# NASA IMS Bearing Dataset — RUL Target Definition & Analysis Report

**Phase 5 Deliverable**  
**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Git Commit:** `{metadata['git_commit']}`

---

## 1. RUL Label Derivation Context

The NASA IMS Bearing Set 2 dataset is a **run-to-failure historical research dataset** comprising 984 snapshots recorded at ~10-minute intervals. The dataset contains **no explicit per-snapshot RUL labels**.

Therefore, remaining useful life (RUL) targets must be mathematically derived from the chronological snapshot index $t \\in [0, 983]$.

---

## 2. Target Formulations Evaluated

### Option A: Un-capped Linear RUL
$$\\text{{RUL}}_{{\\text{{linear}}}}(t) = (N - 1) - t = 983 - t$$
- **Unit:** Snapshots remaining until final failure.
- **Physical Assumption:** Assumes steady, uniform wear from snapshot 0 to failure at snapshot 983.
- **Limitation:** Physically unrealistic for early life (snapshots 0–500) where bearing vibration signatures remain baseline healthy and virtually indistinguishable.

### Option B: Capped Piecewise-Linear RUL (Selected)
$$\\text{{RUL}}_{{\\text{{capped}}}}(t) = \\min((N - 1) - t, \\text{{cap}}) = \\min(983 - t, 400)$$
- **Unit:** Snapshots remaining (capped at 400 snapshots $\\approx 66.6$ hours).
- **Physical Rationale:** Standard PHM (Prognostics and Health Management) best practice. Reflects that health degradation is only observable once initial wear begins. Prevents the model from attempting to distinguish healthy snapshot 10 from healthy snapshot 300.

---

## 3. Target-Range Extrapolation Analysis (Critical ML Limitation)

Tree-based ensemble models (including XGBoost) split feature space using axis-aligned decision boundaries. They **cannot extrapolate target predictions outside the range of $Y$ values seen during training**.

### Empirical Target Ranges Across Chronological Split

| Split Region | Snapshot Indices | Linear RUL Target Range | Capped RUL (400) Target Range |
|---|---|---|---|
| **Train Set** | 0 – 600 | **[{ranges['linear_rul']['train'][0]}, {ranges['linear_rul']['train'][1]}]** | **[{ranges[target_info['target_column']]['train'][0]}, {ranges[target_info['target_column']]['train'][1]}]** |
| **Validation Set** | 601 – 750 | [{ranges['linear_rul']['val'][0]}, {ranges['linear_rul']['val'][1]}] | [{ranges[target_info['target_column']]['val'][0]}, {ranges[target_info['target_column']]['val'][1]}] |
| **Test Set** | 751 – 983 | **[{ranges['linear_rul']['test'][0]}, {ranges['linear_rul']['test'][1]}]** | **[{ranges[target_info['target_column']]['test'][0]}, {ranges[target_info['target_column']]['test'][1]}]** |

### Critical Finding:
When splitting chronologically:
1. Under **Linear RUL**, the training set target range is $[383, 983]$, while the test set target range is $[0, 232]$.
2. The test set targets $[0, 232]$ lie **entirely below** the minimum target seen in training ($\min(Y_{{train}}) = 383$).
3. Consequently, an un-capped model trained only on early snapshots $0–600$ cannot predict RUL values near 0 at failure, producing a structural prediction floor around $\\approx 383$.
4. **Capped RUL (400)** bounds the early health target to 400, reducing the range discrepancy and aligning with standard PHM practice.

---

## 4. Final Target Decision

- **Selected Target:** `capped_rul_400`
- **Formula:** $\\text{{RUL}}(t) = \\min(983 - t, 400)$
- **Unit:** Snapshots ($\\approx 10$ min / snapshot)
- **Selection Rationale:** Chosen based strictly on physical plausibility and training/validation target distribution analysis, without inspecting test set predictions.
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(doc)


def generate_training_report(output_file: Path, metadata: Dict[str, Any], feature_importance: Dict[str, float]):
    """Generates `ml-service/reports/xgboost_training_report.md`."""
    m = metadata["metrics"]
    audit = metadata["leakage_audit_summary"]

    top_10_features = list(feature_importance.items())[:10]
    top_features_table = "\n".join(
        [f"| `{f}` | {gain:.4f} |" for f, gain in top_10_features]
    )

    doc = f"""# XGBoost RUL Supervised Model Training & Evaluation Report

**Phase 5 Deliverable**  
**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Model Name:** `{metadata['model_name']}` (v{metadata['model_version']})  
**Git Commit:** `{metadata['git_commit']}`  
**Data Fingerprint:** `{metadata['data_fingerprint'][:16]}...`

---

## 1. Dataset & Chronological Split Overview

- **Source:** Historical NASA IMS Bearing Set 2 replayed vibration telemetry.
- **Snapshot Count:** 984 snapshots ($20,480 \times 4$ vibration samples per snapshot).
- **Feature Vector:** 36 canonical features (9 time/frequency features $\times$ 4 channels).
- **Target:** `capped_rul_400` ($\\min(983 - t, 400)$ snapshots).

### Split Configuration

| Split | Index Range | Snapshot Count | Percentage |
|---|---|---|---|
| **Training** | 0 – 600 | 601 | 61.1% |
| **Validation** | 601 – 750 | 150 | 15.2% |
| **Test (Frozen)** | 751 – 983 | 233 | 23.7% |

---

## 2. Temporal Leakage Audit Results

An automated leakage audit was executed prior to training:

- **Chronological Split Contiguity & Ordering:** {'PASSED' if audit['splits_audit']['passed'] else 'FAILED'}
- **Feature Leakage Check (No target/sequence in features):** {'PASSED' if audit['feature_leakage_audit']['passed'] else 'FAILED'}
- **Preprocessor Isolation (Fit strictly on train set):** {'PASSED' if audit['preprocessor_isolation_audit']['passed'] else 'FAILED'}
- **Causality Verification (Snapshot $t$ independent of $t > t$):** {'PASSED' if audit['causality_audit']['passed'] else 'FAILED'}

---

## 3. Measured Model Evaluation & Baselines

All metrics are measured from actual execution. Test set was evaluated **ONCE** after configuration freeze.

### Metric Comparison Table (Target Unit: Snapshots)

| Model | Train RMSE | Train MAE | Val RMSE | Val MAE | Test RMSE | Test MAE | Test $R^2$ |
|---|---|---|---|---|---|---|---|
| **Mean Predictor Baseline** | {m['mean_baseline']['train']['rmse']} | {m['mean_baseline']['train']['mae']} | {m['mean_baseline']['val']['rmse']} | {m['mean_baseline']['val']['mae']} | {m['mean_baseline']['test']['rmse']} | {m['mean_baseline']['test']['mae']} | {m['mean_baseline']['test']['r2']} |
| **Linear Regression Baseline** | {m['linear_regression_baseline']['train']['rmse']} | {m['linear_regression_baseline']['train']['mae']} | {m['linear_regression_baseline']['val']['rmse']} | {m['linear_regression_baseline']['val']['mae']} | {m['linear_regression_baseline']['test']['rmse']} | {m['linear_regression_baseline']['test']['mae']} | {m['linear_regression_baseline']['test']['r2']} |
| **XGBoost Regressor (Selected)** | **{m['xgboost']['train']['rmse']}** | **{m['xgboost']['train']['mae']}** | **{m['xgboost']['val']['rmse']}** | **{m['xgboost']['val']['mae']}** | **{m['xgboost']['test']['rmse']}** | **{m['xgboost']['test']['mae']}** | **{m['xgboost']['test']['r2']}** |

### Approx Time Error Conversion (Test Set)
- **XGBoost Test MAE:** {m['xgboost']['test']['mae']} snapshots $\approx$ {round(m['xgboost']['test']['mae'] * 10 / 60, 2)} hours.
- **XGBoost Test RMSE:** {m['xgboost']['test']['rmse']} snapshots $\approx$ {round(m['xgboost']['test']['rmse'] * 10 / 60, 2)} hours.

---

## 4. Top Feature Importance (Gain Metric)

> **Disclaimer:** Feature importance reflects decision-tree splitting gain and does **not** constitute physical/causal proof of bearing mechanics.

| Feature | Gain Score |
|---|---|
{top_features_table}

---

## 5. Artifact Bundle & Reproducibility Command

The following artifacts have been persisted:
1. `ml-service/models/xgboost_rul_model.json` (Native XGBoost format)
2. `ml-service/models/xgboost_scaler.joblib` (Fitted RobustScaler)
3. `ml-service/models/xgboost_model_metadata.json` (Versioned metadata)

### Regeneration Command
To reproduce the feature matrix, leakage audit, training, and report generation from raw data:
```bash
python -m src.train_xgboost
```
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(doc)


if __name__ == "__main__":
    print("Executing Phase 5 XGBoost Supervised RUL Training & Evaluation Pipeline...")
    meta = train_and_evaluate_xgboost()
    print("Training Complete!")
    print(f"Metrics (XGBoost Test): RMSE={meta['metrics']['xgboost']['test']['rmse']}, MAE={meta['metrics']['xgboost']['test']['mae']}, R2={meta['metrics']['xgboost']['test']['r2']}")
