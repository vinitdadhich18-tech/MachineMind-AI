"""
inference.py - Reproducible Anomaly Inference Engine for MachineMind AI.

Phase 10: Model Packaging - Step 2
Target Dataset: NASA IMS Bearing Dataset (Set 2)

SECURITY WARNING:
Never load joblib or pickle model artifacts from untrusted or unverified sources.
Joblib artifacts can execute arbitrary code during deserialization.

This module provides the production-ready AnomalyInferenceEngine class which turns a single
raw 1-second vibration snapshot (20,480 samples x 4 channels) into continuous anomaly scores,
per-channel exceedance flags, persistence-filtered sustained flags (k=3 consecutive exceedances),
and a system-level alert decision using a Logical OR aggregation policy across channels.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.feature_extraction import extract_time_features

DEFAULT_MODELS_DIR = Path("models")
CHANNELS = ["ch1", "ch2", "ch3", "ch4"]


class AnomalyInferenceEngine:
    """
    Production Anomaly Inference Engine for MachineMind AI.

    Loads packaged model artifacts (Isolation Forest or PCA Reconstruction Error)
    and metadata from models/ to evaluate raw 4-channel vibration snapshots.
    """

    def __init__(
        self,
        artifacts_dir: Union[str, Path] = DEFAULT_MODELS_DIR,
        model_type: str = "iforest",
        stateful: bool = True,
    ):
        """
        Initializes the AnomalyInferenceEngine.

        Parameters
        ----------
        artifacts_dir : str or Path, default "models"
            Path to directory containing packaged joblib artifacts and model_metadata.json.
        model_type : str, default "iforest"
            Anomaly detection model type: "iforest" or "pca".
        stateful : bool, default True
            If True, maintains stateful persistence counters across sequential snapshot calls.
            If False, operates statelessly (scoring and raw flags only, sustained flags stay False).
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.model_type = model_type.lower().strip()
        self.stateful = stateful

        if self.model_type not in ["iforest", "pca"]:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Choose 'iforest' or 'pca'."
            )

        # 1. Load metadata JSON
        self.metadata_path = self.artifacts_dir / "model_metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata JSON not found at: {self.metadata_path}"
            )

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # 2. Version compatibility check
        self._check_environment_compatibility()

        # 3. Load joblib artifact
        artifact_filename = (
            "iforest_pipeline_v1.joblib"
            if self.model_type == "iforest"
            else "pca_pipeline_v1.joblib"
        )
        self.artifact_path = self.artifacts_dir / artifact_filename
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Packaged model artifact not found at: {self.artifact_path}"
            )

        self.artifact = joblib.load(self.artifact_path)

        # 4. Validate artifact structure and channels
        if "channels" not in self.artifact:
            raise KeyError(
                f"Artifact at {self.artifact_path} is missing 'channels' dictionary."
            )

        for ch in CHANNELS:
            if ch not in self.artifact["channels"]:
                raise KeyError(
                    f"Artifact at {self.artifact_path} is missing channel '{ch}'."
                )

        # 5. Extract feature ordering, thresholds, and persistence parameter k from metadata
        self.ordered_feature_names = self.metadata["feature_extraction"][
            "ordered_feature_names"
        ]
        if len(self.ordered_feature_names) != 28:
            raise ValueError(
                f"Expected 28 ordered feature names in metadata, got {len(self.ordered_feature_names)}."
            )

        self.thresholds = self.metadata["models"][self.model_type][
            "channel_thresholds_P99"
        ]
        self.k = int(self.metadata["decision_logic"]["persistence_k"])

        # 6. Initialize persistence state
        self.consecutive_exceedances = {ch: 0 for ch in CHANNELS}

    def _check_environment_compatibility(self) -> None:
        """
        Compares runtime dependency versions against metadata versions and issues warnings if mismatched.
        """
        env_meta = self.metadata.get("environment", {})
        current_versions = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        }

        for pkg, curr_ver in current_versions.items():
            meta_key = f"{pkg}_version"
            meta_ver = env_meta.get(meta_key)
            if meta_ver and curr_ver != meta_ver:
                warnings.warn(
                    f"Environment version mismatch for '{pkg}': "
                    f"Runtime is {curr_ver}, artifact metadata recorded {meta_ver}. "
                    "Artifact loading will proceed, but inspect results carefully.",
                    UserWarning,
                )

    def reset_state(self) -> None:
        """
        Resets all channel persistence exceedance counters to zero.
        """
        for ch in CHANNELS:
            self.consecutive_exceedances[ch] = 0

    def validate_raw_snapshot(
        self, raw_snapshot: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Validates raw 4-channel snapshot array shape and numerical values.

        Parameters
        ----------
        raw_snapshot : np.ndarray or pd.DataFrame
            Input raw vibration snapshot array of shape (20480, 4).

        Returns
        -------
        np.ndarray
            Validated 2D float64 numpy array of shape (20480, 4).
        """
        if isinstance(raw_snapshot, pd.DataFrame):
            data = raw_snapshot.to_numpy()
        elif isinstance(raw_snapshot, np.ndarray):
            data = raw_snapshot.copy()
        else:
            raise TypeError(
                f"Expected raw_snapshot to be np.ndarray or pd.DataFrame, got {type(raw_snapshot)}."
            )

        if data.size == 0:
            raise ValueError("Input raw_snapshot is empty.")

        if not np.issubdtype(data.dtype, np.number):
            raise TypeError(
                f"Input raw_snapshot contains non-numeric data type '{data.dtype}'."
            )

        if data.ndim != 2:
            raise ValueError(
                f"Expected 2D raw_snapshot of shape (20480, 4), got {data.ndim}D array with shape {data.shape}."
            )

        if data.shape != (20480, 4):
            raise ValueError(
                f"Expected raw_snapshot shape (20480, 4), got {data.shape}."
            )

        if np.isnan(data).any():
            raise ValueError("Input raw_snapshot contains NaN values.")

        if np.isinf(data).any():
            raise ValueError("Input raw_snapshot contains Infinite values.")

        return data.astype(np.float64, copy=False)

    def predict_snapshot(
        self,
        raw_snapshot: Union[np.ndarray, pd.DataFrame],
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference pipeline on a single raw snapshot:
        validation -> preprocessing -> feature extraction -> scaling -> scoring -> thresholding -> persistence -> alert.

        Parameters
        ----------
        raw_snapshot : np.ndarray or pd.DataFrame
            Input raw vibration snapshot of shape (20480, 4).
        timestamp : str, optional
            Optional timestamp string for output record tracking.

        Returns
        -------
        dict
            Structured inference dictionary containing native Python types.
        """
        # 1. Input Validation
        arr_valid = self.validate_raw_snapshot(raw_snapshot)

        # 2. Preprocessing & Feature Extraction
        # Convert validated array to DataFrame for feature extraction
        df_snapshot = pd.DataFrame(
            arr_valid,
            columns=["Channel_1", "Channel_2", "Channel_3", "Channel_4"],
        )

        time_features = extract_time_features(df_snapshot)

        # 3. Verify feature alignment against metadata
        extracted_keys = list(time_features.keys())
        if len(extracted_keys) != 28:
            raise ValueError(
                f"Extracted {len(extracted_keys)} features, expected exactly 28."
            )

        missing_feats = [
            col for col in self.ordered_feature_names if col not in time_features
        ]
        if missing_feats:
            raise KeyError(
                f"Feature extraction missing required metadata feature names: {missing_feats}"
            )

        # 4. Per-channel scoring and thresholding
        channel_scores = {}
        channel_thresholds = {}
        channel_raw_flags = {}
        channel_sustained_flags = {}

        for ch in CHANNELS:
            ch_stats = [
                "mean",
                "std",
                "rms",
                "p2p",
                "skewness",
                "kurtosis",
                "crest_factor",
            ]
            ch_feat_cols = [f"{ch}_{s}" for s in ch_stats]

            x_ch = np.array([[time_features[col] for col in ch_feat_cols]], dtype=np.float64)

            # Retrieve pipeline object from loaded channel dictionary
            pipeline = self.artifact["channels"][ch]["pipeline"]

            # Compute score without calling fit()
            score_val = float(pipeline.compute_anomaly_scores(x_ch)[0])
            thresh_val = float(self.thresholds[ch])
            raw_flag = bool(score_val > thresh_val)

            channel_scores[ch] = score_val
            channel_thresholds[ch] = thresh_val
            channel_raw_flags[ch] = raw_flag

            # 5. Stateful Persistence vs Stateless Mode
            if self.stateful:
                if raw_flag:
                    self.consecutive_exceedances[ch] += 1
                else:
                    self.consecutive_exceedances[ch] = 0

                sustained_flag = bool(self.consecutive_exceedances[ch] >= self.k)
            else:
                sustained_flag = False

            channel_sustained_flags[ch] = sustained_flag

        # 6. Logical OR System Alert Policy
        if self.stateful:
            system_alert = bool(any(channel_sustained_flags.values()))
        else:
            system_alert = False

        # Output record with native Python types
        result = {
            "timestamp": str(timestamp) if timestamp is not None else None,
            "model_type": self.model_type,
            "channel_scores": channel_scores,
            "channel_thresholds": channel_thresholds,
            "channel_raw_flags": channel_raw_flags,
            "channel_sustained_flags": channel_sustained_flags,
            "system_alert": system_alert,
            "state_consecutive_counts": {
                ch: int(self.consecutive_exceedances[ch]) for ch in CHANNELS
            },
        }

        return result
