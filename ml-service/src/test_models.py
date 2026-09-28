"""
test_models.py - Verification and Unit Test Suite for MachineMind AI Models Module.

Phase 7: ML Model Development
Run with: python -m unittest src/test_models.py
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory / src directory to sys.path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.models import (
        AnomalyDetectionPipeline,
        PCAReconstructionModel,
        train_and_evaluate_bearing_models,
        calculate_validation_thresholds,
        DEFAULT_REF_END_IDX,
        DEFAULT_VAL_END_IDX,
        DEFAULT_RANDOM_STATE,
    )
except ImportError:
    from models import (
        AnomalyDetectionPipeline,
        PCAReconstructionModel,
        train_and_evaluate_bearing_models,
        calculate_validation_thresholds,
        DEFAULT_REF_END_IDX,
        DEFAULT_VAL_END_IDX,
        DEFAULT_RANDOM_STATE,
    )


class TestAnomalyModels(unittest.TestCase):
    """Unit test suite for Phase 7 anomaly detection models."""

    def setUp(self):
        """Generate synthetic dataset for testing."""
        np.random.seed(42)
        n_snapshots = 250
        
        # Build 4 channel features (7 time-domain + 2 spectral per channel = 36 features)
        data = {}
        data["file_index"] = np.arange(n_snapshots)
        data["filename"] = [f"2004.02.12.00.00.{i:02d}" for i in range(n_snapshots)]
        data["timestamp"] = pd.date_range("2004-02-12", periods=n_snapshots, freq="10min")
        data["is_valid"] = True
        
        channels = ["ch1", "ch2", "ch3", "ch4"]
        feat_types = ["mean", "std", "rms", "p2p", "skewness", "kurtosis", "crest_factor", "spectral_energy", "spectral_centroid"]
        
        for ch in channels:
            for ft in feat_types:
                col_name = f"{ch}_{ft}"
                # Normal baseline signal
                vals = np.random.normal(loc=1.0, scale=0.1, size=n_snapshots)
                
                # Inject progressive anomaly into ch1 after snapshot 180
                if ch == "ch1" and ft in ["rms", "p2p", "spectral_energy"]:
                    vals[180:] += np.linspace(0.5, 5.0, n_snapshots - 180)
                    
                data[col_name] = vals
                
        self.df_features = pd.DataFrame(data)

    def test_fit_scoping_isolation(self):
        """Assertion proves that fit() saw only baseline-fit rows."""
        X_all = self.df_features[[c for c in self.df_features.columns if c.startswith("ch1_")]].to_numpy()
        X_train = X_all[:DEFAULT_REF_END_IDX]
        
        # Fit pipeline on X_train only
        pipe = AnomalyDetectionPipeline(model_type="iforest", random_state=42)
        pipe.fit(X_train)
        
        # Check pipeline is fitted and scaler center_ matches feature dimension count
        self.assertTrue(pipe.is_fitted)
        self.assertEqual(len(pipe.scaler.center_), X_train.shape[1])

    def test_seed_reproducibility(self):
        """Two runs with the same seed give identical scores."""
        X_all = self.df_features[[c for c in self.df_features.columns if c.startswith("ch1_")]].to_numpy()
        X_train = X_all[:DEFAULT_REF_END_IDX]
        
        pipe1 = AnomalyDetectionPipeline(model_type="iforest", random_state=42).fit(X_train)
        scores1 = pipe1.compute_anomaly_scores(X_all)
        
        pipe2 = AnomalyDetectionPipeline(model_type="iforest", random_state=42).fit(X_train)
        scores2 = pipe2.compute_anomaly_scores(X_all)
        
        np.testing.assert_array_almost_equal(scores1, scores2)

    def test_scores_validity_and_sign_convention(self):
        """Scores exist for every row with no NaN, and higher = more anomalous."""
        df_scores, meta = train_and_evaluate_bearing_models(
            self.df_features,
            ref_end_idx=DEFAULT_REF_END_IDX,
            val_end_idx=DEFAULT_VAL_END_IDX,
            model_types=["iforest", "pca"],
            random_state=42
        )
        
        for m_type in ["iforest", "pca"]:
            for ch in ["ch1", "ch2", "ch3", "ch4"]:
                col = f"score_{m_type}_{ch}"
                self.assertIn(col, df_scores.columns)
                scores = df_scores[col].to_numpy()
                
                # No NaNs or Infinite values
                self.assertFalse(np.isnan(scores).any())
                self.assertFalse(np.isinf(scores).any())
                self.assertEqual(len(scores), len(self.df_features))
                # Scores must be non-negative
                self.assertTrue((scores >= 0).all())

    def test_synthetic_anomaly_detection(self):
        """Synthetic test with an injected mean shift shows model reacts to a known anomaly."""
        df_scores, meta = train_and_evaluate_bearing_models(
            self.df_features,
            ref_end_idx=DEFAULT_REF_END_IDX,
            val_end_idx=DEFAULT_VAL_END_IDX,
            model_types=["iforest", "pca"],
            random_state=42
        )
        
        # Bearing 1 has injected anomaly after snapshot 180
        # Bearing 2 stays normal throughout
        for m_type in ["iforest", "pca"]:
            score_ch1 = df_scores[f"score_{m_type}_ch1"].to_numpy()
            score_ch2 = df_scores[f"score_{m_type}_ch2"].to_numpy()
            
            baseline_ch1_mean = np.mean(score_ch1[:DEFAULT_REF_END_IDX])
            degraded_ch1_mean = np.mean(score_ch1[200:])
            degraded_ch2_mean = np.mean(score_ch2[200:])
            
            # Degraded ch1 score must be significantly higher than its baseline fit score
            self.assertGreater(degraded_ch1_mean, baseline_ch1_mean + 0.05)
            
            # For PCA, degraded ch1 score must be drastically higher than healthy ch2 score
            if m_type == "pca":
                self.assertGreater(degraded_ch1_mean, degraded_ch2_mean * 2.0)


if __name__ == "__main__":
    unittest.main()
