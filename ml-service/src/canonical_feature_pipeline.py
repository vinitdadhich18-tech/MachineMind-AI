"""
canonical_feature_pipeline.py - Canonical Vibration Feature Engineering Pipeline.

Phase 4: Signal Processing & Feature Engineering Pipeline
Target Dataset: NASA IMS Bearing Dataset (Set 2)

Architecture Role:
ONE authoritative feature extraction pipeline shared identically for:
1. Offline dataset feature extraction (XGBoost model training)
2. Streaming MQTT inference (live TelemetryRecord processing)
3. Optional InfluxDB feature persistence (Phase 3 compatibility)

Feature Schema (36 total canonical features):
----------------------------------------------
4 Channels (ch1, ch2, ch3, ch4) x 9 Features = 36 Features per snapshot.

Time-Domain Features (7 per channel):
1. mean: Arithmetic mean (DC offset).
2. std: Sample standard deviation (ddof=1).
3. rms: Root Mean Square amplitude sqrt(mean(x^2)).
4. p2p: Peak-to-peak amplitude max(x) - min(x).
5. skewness: Third standardized moment (distribution asymmetry).
6. kurtosis: Fisher's excess kurtosis (4th standardized moment - 3.0; Gaussian = 0.0).
7. crest_factor: Peak-to-RMS ratio max(|x|) / (rms + epsilon).

Frequency-Domain Features (2 per channel):
8. spectral_energy: Hann-windowed positive frequency FFT energy sum.
9. spectral_centroid: Center of mass frequency (Hz).

Parity Guarantee:
The exact same signal processing calculations are used for offline DataFrames
and online streaming TelemetryRecords. No dual code paths. Zero future data leakage.
"""

import sys
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

# Ensure ml-service root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_SERVICE_DIR = PROJECT_ROOT / "ml-service"
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

from src.ingestion_consumer import TelemetryRecord
from src.feature_extraction import (
    extract_time_features,
    extract_frequency_features,
    EPSILON
)

logger = logging.getLogger("MachineMind.CanonicalFeaturePipeline")

CHANNELS = ["ch1", "ch2", "ch3", "ch4"]
FEATURE_SUFFIXES = [
    "mean", "std", "rms", "p2p", "skewness", "kurtosis", "crest_factor",
    "spectral_energy", "spectral_centroid"
]

# Standard deterministic 36 feature names in exact column order
CANONICAL_FEATURE_NAMES: List[str] = [
    f"{ch}_{suffix}" for ch in CHANNELS for suffix in FEATURE_SUFFIXES
]


@dataclass
class CanonicalFeatureVector:
    """
    Structured, immutable output representation of canonical vibration features for a single snapshot.
    """
    machine_id: str
    snapshot_sequence: int
    original_timestamp: str
    ingest_timestamp: str
    sampling_rate_hz: float
    feature_names: List[str]
    feature_values: np.ndarray  # 1D float64 NumPy array of shape (36,)
    feature_dict: Dict[str, float]

    def to_dataframe(self) -> pd.DataFrame:
        """Converts feature vector to 1-row pandas DataFrame with metadata columns."""
        meta = {
            "machine_id": self.machine_id,
            "snapshot_sequence": self.snapshot_sequence,
            "original_timestamp": self.original_timestamp,
            "ingest_timestamp": self.ingest_timestamp,
            "sampling_rate_hz": self.sampling_rate_hz,
        }
        full_dict = {**meta, **self.feature_dict}
        return pd.DataFrame([full_dict])

    def validate_finite(self) -> bool:
        """Asserts that all 36 feature values are finite numbers (no NaNs or infinities)."""
        if self.feature_values.shape != (36,):
            raise ValueError(f"Expected 36 feature values, got shape {self.feature_values.shape}.")
        if np.isnan(self.feature_values).any():
            raise ValueError("Feature vector contains NaN values.")
        if np.isinf(self.feature_values).any():
            raise ValueError("Feature vector contains Infinite values.")
        return True


class CanonicalFeaturePipeline:
    """
    Authoritative Signal Processing and Feature Engineering Pipeline.
    Guarantees 100% offline/online mathematical parity.
    """

    def __init__(self, sampling_rate_hz: float = 20480.0):
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.feature_names = CANONICAL_FEATURE_NAMES.copy()

    def process_snapshot_dataframe(
        self,
        df_snapshot: pd.DataFrame,
        machine_id: str = "ims_set2_rig",
        snapshot_sequence: int = 0,
        original_timestamp: str = "2004-02-12T10:32:39Z",
        ingest_timestamp: Optional[str] = None
    ) -> CanonicalFeatureVector:
        """
        Extracts 36 canonical features from a 20480x4 raw snapshot DataFrame (offline path).

        Parameters
        ----------
        df_snapshot : pd.DataFrame
            Raw 4-channel vibration DataFrame (shape 20480, 4).
        machine_id : str
            Machinery equipment ID.
        snapshot_sequence : int
            Monotonic sequence index.
        original_timestamp : str
            Original NASA snapshot timestamp.
        ingest_timestamp : str, optional
            Processing timestamp. Defaults to current UTC.

        Returns
        -------
        CanonicalFeatureVector
        """
        if not isinstance(df_snapshot, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df_snapshot)}.")

        if df_snapshot.shape != (20480, 4):
            raise ValueError(f"Expected snapshot shape (20480, 4), got {df_snapshot.shape}.")

        # standard column names
        df_norm = df_snapshot.copy()
        df_norm.columns = ["Channel_1", "Channel_2", "Channel_3", "Channel_4"]

        # Extract features using shared mathematical utility functions
        time_feats = extract_time_features(df_norm)
        freq_feats = extract_frequency_features(df_norm, sampling_rate=self.sampling_rate_hz)
        merged_dict = {**time_feats, **freq_feats}

        # Build ordered feature array (36 elements)
        feat_vals = np.array([float(merged_dict[col]) for col in self.feature_names], dtype=np.float64)

        if ingest_timestamp is None:
            ingest_ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            ingest_ts_str = str(ingest_timestamp)

        vector = CanonicalFeatureVector(
            machine_id=str(machine_id),
            snapshot_sequence=int(snapshot_sequence),
            original_timestamp=str(original_timestamp),
            ingest_timestamp=ingest_ts_str,
            sampling_rate_hz=self.sampling_rate_hz,
            feature_names=self.feature_names.copy(),
            feature_values=feat_vals,
            feature_dict={col: float(merged_dict[col]) for col in self.feature_names}
        )

        vector.validate_finite()
        return vector

    def process_telemetry_record(self, record: TelemetryRecord) -> CanonicalFeatureVector:
        """
        Extracts 36 canonical features from an MQTT TelemetryRecord (streaming online path).

        Parameters
        ----------
        record : TelemetryRecord
            Validated TelemetryRecord from Phase 2 ingestion consumer.

        Returns
        -------
        CanonicalFeatureVector
        """
        if not isinstance(record, TelemetryRecord):
            raise TypeError(f"Expected TelemetryRecord, got {type(record)}.")

        # Reconstruct DataFrame from 1D float64 NumPy arrays in TelemetryRecord
        df_snapshot = pd.DataFrame({
            "Channel_1": record.channels["ch1"],
            "Channel_2": record.channels["ch2"],
            "Channel_3": record.channels["ch3"],
            "Channel_4": record.channels["ch4"],
        })

        return self.process_snapshot_dataframe(
            df_snapshot=df_snapshot,
            machine_id=record.machine_id,
            snapshot_sequence=record.snapshot_sequence,
            original_timestamp=record.original_timestamp,
            ingest_timestamp=record.ingest_timestamp
        )

    def process_dataset(
        self,
        snapshot_files: List[Path],
        machine_id: str = "ims_set2_rig"
    ) -> pd.DataFrame:
        """
        Processes a list of raw snapshot files chronologically to generate a full feature dataset.
        Used for offline XGBoost training pipeline.

        Returns
        -------
        pd.DataFrame
            DataFrame of shape (N_snapshots, 41) containing metadata + 36 feature columns.
        """
        rows = []
        for i, fpath in enumerate(snapshot_files):
            df_raw = pd.read_csv(fpath, sep=r"\s+", header=None)
            ts = fpath.name  # file timestamp name e.g. 2004.02.12.10.32.39
            vector = self.process_snapshot_dataframe(
                df_snapshot=df_raw,
                machine_id=machine_id,
                snapshot_sequence=i,
                original_timestamp=ts
            )
            rows.append(vector.to_dataframe())

        return pd.concat(rows, ignore_index=True)


class FeatureScalerAdapter:
    """
    Preprocessing scaler wrapper for RobustScaler/StandardScaler.
    Enforces fit-on-training-only rules to prevent data leakage.
    """

    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type.lower()
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'robust' or 'standard'.")
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "FeatureScalerAdapter":
        """Fits scaler strictly on training feature matrix."""
        if X_train.ndim != 2 or X_train.shape[1] != 36:
            raise ValueError(f"Expected 2D training matrix of shape (N, 36), got {X_train.shape}.")
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scales feature matrix using fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted on training data before calling transform().")
        return self.scaler.transform(X)
