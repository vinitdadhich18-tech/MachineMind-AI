"""
models.py - Unsupervised Anomaly Detection Module for MachineMind AI.

Phase 7: ML Model Development
Target Dataset: NASA IMS Bearing Dataset (Set 2)

This module implements reproducible unsupervised anomaly detection pipelines
(Isolation Forest and PCA Reconstruction Error) for per-bearing condition monitoring.

Key Design Rules:
1. Baseline Scoping: Scalers and models are fit strictly on baseline-fit snapshots (0 to ref_end_idx-1).
2. Data Leakage Prevention: Validation period (ref_end_idx to val_end_idx-1) is used strictly
   for score calibration and threshold estimation.
3. Score Standardization: All model score functions are normalized such that
   HIGHER SCORE = MORE ANOMALOUS.
4. Pipeline Integration: RobustScaler is bundled with estimators to handle feature scale differences
   and micro-outliers safely.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA


# Default partition constants aligned with Phase 6
DEFAULT_REF_END_IDX = 160   # Snapshots 0..159 (Fit period, N=160)
DEFAULT_VAL_END_IDX = 200   # Snapshots 160..199 (Validation period, N=40)
DEFAULT_RANDOM_STATE = 42

METADATA_COLUMNS = [
    "file_index",
    "filename",
    "timestamp",
    "is_valid",
    "window_index",
    "window_start_sample",
    "window_end_sample",
]


class PCAReconstructionModel:
    """
    PCA Reconstruction Error Anomaly Detector.
    
    Fits PCA on normal baseline feature matrix and computes anomaly score
    as the Mean Squared Reconstruction Error across feature dimensions:
        score = (1 / d) * || X - X_reconstructed ||_2^2
    """

    def __init__(self, n_components: Union[int, float] = 2, random_state: int = DEFAULT_RANDOM_STATE):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "PCAReconstructionModel":
        """Fit PCA strictly on normal baseline feature vectors."""
        self.pca.fit(X)
        self.is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute mean squared reconstruction error for each sample.
        Higher score = higher reconstruction error = more anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring samples.")
        X_trans = self.pca.transform(X)
        X_rec = self.pca.inverse_transform(X_trans)
        # Mean Squared Error per row across feature dimensions
        mse = np.mean((X - X_rec) ** 2, axis=1)
        return mse


class AnomalyDetectionPipeline:
    """
    Unified Pipeline bundling RobustScaler with an anomaly detection model
    (IsolationForest or PCAReconstructionModel).
    
    Guarantees strict baseline-fit scoping and standardized 'higher = more anomalous' scoring.
    """

    def __init__(
        self,
        model_type: str = "iforest",
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model_params = model_params or {}
        
        self.scaler = RobustScaler()
        self.is_fitted = False
        
        if self.model_type == "iforest":
            params = {
                "n_estimators": 100,
                "max_samples": "auto",
                "contamination": "auto",
                "random_state": self.random_state,
            }
            params.update(self.model_params)
            self.model = IsolationForest(**params)
        elif self.model_type == "pca":
            params = {
                "n_components": 2,
                "random_state": self.random_state,
            }
            params.update(self.model_params)
            self.model = PCAReconstructionModel(**params)
        else:
            raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'iforest' or 'pca'.")

    def fit(self, X: np.ndarray) -> "AnomalyDetectionPipeline":
        """
        Fit scaler and anomaly detection model strictly on training data.
        """
        X_scaled = self.scaler.fit_transform(X)
        if self.model_type == "iforest":
            self.model.fit(X_scaled)
        elif self.model_type == "pca":
            self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute normalized anomaly scores on target matrix X.
        Guaranteed convention: HIGHER = MORE ANOMALOUS (strictly non-negative).
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before computing anomaly scores.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == "iforest":
            # decision_function: positive (~0.1..0.4) for inliers, negative (-0.4..0) for outliers.
            # score = 0.5 - decision_function(X) maps inliers to ~0.1..0.4 (low) and outliers to > 0.5 (high).
            raw_dec = self.model.decision_function(X_scaled)
            scores = 0.5 - raw_dec
        elif self.model_type == "pca":
            scores = self.model.score_samples(X_scaled)
            
        return scores


def calculate_validation_thresholds(
    val_scores: np.ndarray,
    sigmas: List[float] = [3.0, 4.0, 5.0],
    percentiles: List[float] = [95.0, 99.0, 99.9]
) -> Dict[str, float]:
    """
    Calculate anomaly score thresholds based strictly on validation period scores.
    """
    mean_val = np.mean(val_scores)
    std_val = np.std(val_scores, ddof=1) if len(val_scores) > 1 else 0.0
    
    thresholds = {}
    for k in sigmas:
        thresholds[f"mean_{int(k)}sigma"] = float(mean_val + k * std_val)
    for p in percentiles:
        p_name = f"p{str(p).replace('.', '_')}"
        thresholds[p_name] = float(np.percentile(val_scores, p))
        
    thresholds["val_max"] = float(np.max(val_scores))
    thresholds["val_mean"] = float(mean_val)
    thresholds["val_std"] = float(std_val)
    
    return thresholds


def train_and_evaluate_bearing_models(
    df_features: pd.DataFrame,
    ref_end_idx: int = DEFAULT_REF_END_IDX,
    val_end_idx: int = DEFAULT_VAL_END_IDX,
    model_types: List[str] = ["iforest", "pca"],
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains per-bearing anomaly detection pipelines for each channel (ch1..ch4),
    computes scores across all snapshots, and derives validation thresholds.
    """
    if not isinstance(df_features, pd.DataFrame):
        raise TypeError("df_features must be a pandas DataFrame.")
    if len(df_features) < val_end_idx:
        raise ValueError(f"df_features must have at least {val_end_idx} rows.")
        
    meta_cols = [c for c in df_features.columns if c in METADATA_COLUMNS]
    df_scores = df_features[meta_cols].copy() if meta_cols else pd.DataFrame(index=df_features.index)
    
    channels = ["ch1", "ch2", "ch3", "ch4"]
    results_meta = {
        "ref_end_idx": ref_end_idx,
        "val_end_idx": val_end_idx,
        "random_state": random_state,
        "models": {},
        "thresholds": {}
    }
    
    for m_type in model_types:
        results_meta["models"][m_type] = {}
        results_meta["thresholds"][m_type] = {}
        
        for ch in channels:
            ch_cols = [c for c in df_features.columns if c.startswith(f"{ch}_") and c not in METADATA_COLUMNS]
            if not ch_cols:
                continue
                
            X_all = df_features[ch_cols].to_numpy(dtype=np.float64)
            X_train = X_all[:ref_end_idx]
            
            pipeline = AnomalyDetectionPipeline(
                model_type=m_type,
                random_state=random_state
            )
            pipeline.fit(X_train)
            
            scores_all = pipeline.compute_anomaly_scores(X_all)
            scores_val = scores_all[ref_end_idx:val_end_idx]
            
            val_thresh = calculate_validation_thresholds(scores_val)
            
            score_col_name = f"score_{m_type}_{ch}"
            df_scores[score_col_name] = scores_all
            
            results_meta["models"][m_type][ch] = pipeline
            results_meta["thresholds"][m_type][ch] = val_thresh
            
    return df_scores, results_meta
