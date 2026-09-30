"""
export_models.py - Model Artifact Exporter for MachineMind AI.

Phase 10: Model Packaging - Step 1
Target Dataset: NASA IMS Bearing Dataset (Set 2)

This module trains per-channel anomaly detection pipelines (Isolation Forest and PCA)
on the baseline fit partition (snapshots 0..159), computes exact calibration P99
thresholds on the calibration partition (snapshots 160..179), and exports versioned
joblib artifacts and model_metadata.json into ml-service/models/.

Strict Rules:
- Fit ONLY on snapshots 0..159 (N=160).
- Calibration P99 thresholds calculated strictly on snapshots 160..179 (N=20).
- No evaluation snapshots (200..983) used for fitting or threshold calculation.
- Random seed fixed to 42 for complete reproducibility.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.models import AnomalyDetectionPipeline

# Constants aligned with Phase 9 final configuration
DEFAULT_FIT_END_IDX = 160   # Snapshots 0..159 (N=160)
DEFAULT_CAL_END_IDX = 180   # Snapshots 160..179 (N=20)
DEFAULT_RANDOM_STATE = 42

CHANNELS = ["ch1", "ch2", "ch3", "ch4"]
TIME_STATS_PER_CHANNEL = [
    "mean",
    "std",
    "rms",
    "p2p",
    "skewness",
    "kurtosis",
    "crest_factor",
]

ORDERED_28_FEATURES = [
    f"{ch}_{stat}" for ch in CHANNELS for stat in TIME_STATS_PER_CHANNEL
]

# File path defaults
DEFAULT_FEATURES_PATH = Path("data/processed/features_set2.csv")
DEFAULT_MODELS_DIR = Path("models")


def load_and_validate_features(
    features_csv_path: Union[str, Path] = DEFAULT_FEATURES_PATH
) -> pd.DataFrame:
    """
    Loads extracted features dataset and verifies the existence of all 28 required time-domain features.
    """
    csv_path = Path(features_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Features dataset CSV not found at: {csv_path}")

    df_features = pd.read_csv(csv_path)

    missing_cols = [col for col in ORDERED_28_FEATURES if col not in df_features.columns]
    if missing_cols:
        raise KeyError(
            f"Input features CSV is missing {len(missing_cols)} required time-domain features: {missing_cols}"
        )

    return df_features


def train_and_package_channel_pipelines(
    df_features: pd.DataFrame,
    model_type: str,
    model_params: Dict[str, Any],
    fit_end_idx: int = DEFAULT_FIT_END_IDX,
    cal_end_idx: int = DEFAULT_CAL_END_IDX,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Fits per-channel scalers and estimators strictly on fit_end_idx (0..159),
    computes exact calibration P99 thresholds on (fit_end_idx..cal_end_idx-1),
    and returns artifact dict and thresholds dict.
    """
    channels_dict = {}
    thresholds_p99 = {}

    for ch in CHANNELS:
        ch_cols = [f"{ch}_{stat}" for stat in TIME_STATS_PER_CHANNEL]
        X_all = df_features[ch_cols].to_numpy(dtype=np.float64)

        X_fit = X_all[:fit_end_idx]
        X_cal = X_all[fit_end_idx:cal_end_idx]

        # Instantiate pipeline (RobustScaler + Estimator)
        pipeline = AnomalyDetectionPipeline(
            model_type=model_type,
            scaler_type="robust",
            model_params=model_params,
            random_state=random_state,
        )

        # Fit strictly on fit partition (0..159)
        pipeline.fit(X_fit)

        # Compute calibration scores on calibration partition (160..179)
        scores_cal = pipeline.compute_anomaly_scores(X_cal)

        if np.isnan(scores_cal).any() or np.isinf(scores_cal).any():
            raise ValueError(f"NaN or Inf detected in calibration scores for {model_type} channel {ch}.")

        # Compute exact P99 threshold from calibration scores
        thresh_val = float(np.percentile(scores_cal, 99.0))
        thresholds_p99[ch] = thresh_val

        channels_dict[ch] = {
            "scaler": pipeline.scaler,
            "model": pipeline.model,
            "pipeline": pipeline,
            "feature_names": ch_cols,
            "threshold_p99": thresh_val,
        }

    artifact_dict = {
        "model_type": model_type,
        "version": "1.0.0",
        "fit_window": [0, fit_end_idx],
        "cal_window": [fit_end_idx, cal_end_idx],
        "ordered_feature_names": ORDERED_28_FEATURES,
        "channels": channels_dict,
    }

    return artifact_dict, thresholds_p99


def build_metadata(
    iforest_thresholds: Dict[str, float],
    pca_thresholds: Dict[str, float],
    fit_end_idx: int = DEFAULT_FIT_END_IDX,
    cal_end_idx: int = DEFAULT_CAL_END_IDX,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Constructs comprehensive human-readable model metadata.
    """
    metadata = {
        "model_name": "MachineMind AI Vibration Anomaly Detector",
        "version": "1.0.0",
        "creation_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "task": "Unsupervised Vibration Anomaly Detection",
        "provenance": {
            "dataset": "NASA IMS Bearing Dataset",
            "subset": "Set 2",
            "citation": "J. Lee et al. (2007). IMS, University of Cincinnati. NASA Prognostics Data Repository.",
            "file_range": "2004.02.12.01.05.39 to 2004.02.19.06.22.39 (984 snapshots)",
            "healthy_period_definition": f"Snapshots 0 to {fit_end_idx-1} (first ~26.7 hours, assumed healthy)",
            "calibration_period_definition": f"Snapshots {fit_end_idx} to {cal_end_idx-1} (~3.3 hours)",
        },
        "preprocessing_parameters": {
            "sampling_rate_hz": 20000,
            "snapshot_points": 20480,
            "duration_seconds": 1.024,
            "dc_offset_removal": True,
            "channels": CHANNELS,
        },
        "feature_extraction": {
            "feature_set_type": "time_domain_28",
            "features_per_channel": TIME_STATS_PER_CHANNEL,
            "total_features": len(ORDERED_28_FEATURES),
            "ordered_feature_names": ORDERED_28_FEATURES,
        },
        "scaler": {
            "type": "RobustScaler",
            "fit_window": f"snapshots 0..{fit_end_idx-1}",
        },
        "models": {
            "iforest": {
                "type": "IsolationForest",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_samples": "auto",
                    "contamination": "auto",
                    "random_state": random_state,
                },
                "channel_thresholds_P99": iforest_thresholds,
            },
            "pca": {
                "type": "PCA",
                "hyperparameters": {
                    "n_components": 3,
                    "random_state": random_state,
                },
                "channel_thresholds_P99": pca_thresholds,
            },
        },
        "decision_logic": {
            "persistence_k": 3,
            "system_alert_policy": "Logical OR across channels",
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        },
        "limitations_and_intended_use": [
            "Research/learning prototype based on NASA IMS laboratory dataset.",
            "Single run-to-failure trajectory under constant speed (~2000 RPM) and radial load (6000 lbs).",
            "Does NOT predict remaining useful life (RUL) or exact failure time.",
            "Anomaly alert indicates statistical deviation from baseline, not specific physical defect classification.",
        ],
    }
    return metadata


def export_all_artifacts(
    features_csv_path: Union[str, Path] = DEFAULT_FEATURES_PATH,
    models_dir: Union[str, Path] = DEFAULT_MODELS_DIR,
    fit_end_idx: int = DEFAULT_FIT_END_IDX,
    cal_end_idx: int = DEFAULT_CAL_END_IDX,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Executes complete artifact export pipeline.
    """
    target_models_dir = Path(models_dir)
    target_models_dir.mkdir(parents=True, exist_ok=True)

    df_features = load_and_validate_features(features_csv_path)

    # 1. Isolation Forest Pipeline
    iforest_params = {
        "n_estimators": 100,
        "max_samples": "auto",
        "contamination": "auto",
        "random_state": random_state,
    }
    iforest_artifact, iforest_thresholds = train_and_package_channel_pipelines(
        df_features=df_features,
        model_type="iforest",
        model_params=iforest_params,
        fit_end_idx=fit_end_idx,
        cal_end_idx=cal_end_idx,
        random_state=random_state,
    )

    # 2. PCA Pipeline
    pca_params = {
        "n_components": 3,
        "random_state": random_state,
    }
    pca_artifact, pca_thresholds = train_and_package_channel_pipelines(
        df_features=df_features,
        model_type="pca",
        model_params=pca_params,
        fit_end_idx=fit_end_idx,
        cal_end_idx=cal_end_idx,
        random_state=random_state,
    )

    # 3. Metadata
    metadata = build_metadata(
        iforest_thresholds=iforest_thresholds,
        pca_thresholds=pca_thresholds,
        fit_end_idx=fit_end_idx,
        cal_end_idx=cal_end_idx,
        random_state=random_state,
    )

    # File paths
    iforest_path = target_models_dir / "iforest_pipeline_v1.joblib"
    pca_path = target_models_dir / "pca_pipeline_v1.joblib"
    metadata_path = target_models_dir / "model_metadata.json"

    # Save joblib artifacts
    joblib.dump(iforest_artifact, iforest_path)
    joblib.dump(pca_artifact, pca_path)

    # Save metadata JSON
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    export_summary = {
        "iforest_path": str(iforest_path),
        "pca_path": str(pca_path),
        "metadata_path": str(metadata_path),
        "iforest_thresholds": iforest_thresholds,
        "pca_thresholds": pca_thresholds,
        "metadata": metadata,
    }

    return export_summary


if __name__ == "__main__":
    print("Starting MachineMind AI Model Packaging Exporter...")
    summary = export_all_artifacts()
    print("\n--- Model Packaging Summary ---")
    print(f"Isolation Forest Artifact: {summary['iforest_path']}")
    print(f"PCA Artifact:               {summary['pca_path']}")
    print(f"Metadata JSON:             {summary['metadata_path']}")
    print("\nCalculated Isolation Forest P99 Thresholds:")
    for ch, val in summary['iforest_thresholds'].items():
        print(f"  {ch}: {val:.6f}")
    print("\nCalculated PCA Reconstruction Error P99 Thresholds:")
    for ch, val in summary['pca_thresholds'].items():
        print(f"  {ch}: {val:.6f}")
    print("\nModel Exporter completed successfully.")
