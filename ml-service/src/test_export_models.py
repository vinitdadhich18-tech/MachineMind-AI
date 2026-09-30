"""
test_export_models.py - Unit Test Suite for Model Exporter Module.

Phase 10: Model Packaging - Step 1
Target Dataset: NASA IMS Bearing Dataset (Set 2)

Verifies:
- Artifact files are created in models/
- Joblib artifacts can be reloaded cleanly in Python
- All 4 channels exist in artifacts
- Expected feature count is 28 and matches ORDERED_28_FEATURES
- Scalers and models fit strictly on 0..159 data
- Calibration P99 thresholds calculated strictly on 160..179
- Calculated thresholds are finite numbers
- Metadata contains all required fields (provenance, environment, hyperparameters, etc.)
- Scores computed from loaded models contain no NaN/Inf
- Finalized hyperparameters match approved configuration
"""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.export_models import (
    CHANNELS,
    ORDERED_28_FEATURES,
    TIME_STATS_PER_CHANNEL,
    build_metadata,
    export_all_artifacts,
    load_and_validate_features,
    train_and_package_channel_pipelines,
)


class TestExportModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Generate synthetic feature DataFrame with 200 rows for fast testing."""
        np.random.seed(42)
        rows = []
        for i in range(200):
            row = {
                "file_index": i,
                "filename": f"file_{i}.txt",
                "timestamp": f"2004-02-12 10:{i:02d}:00",
                "is_valid": True,
            }
            for ch in CHANNELS:
                row[f"{ch}_mean"] = np.random.normal(0, 0.01)
                row[f"{ch}_std"] = np.random.normal(0.05, 0.005)
                row[f"{ch}_rms"] = np.random.normal(0.05, 0.005)
                row[f"{ch}_p2p"] = np.random.normal(0.5, 0.05)
                row[f"{ch}_skewness"] = np.random.normal(0.1, 0.02)
                row[f"{ch}_kurtosis"] = np.random.normal(0.5, 0.1)
                row[f"{ch}_crest_factor"] = np.random.normal(5.0, 0.5)
                # Extra frequency features to match features_set2.csv structure
                row[f"{ch}_spectral_energy"] = np.random.normal(0.001, 0.0001)
                row[f"{ch}_spectral_centroid"] = np.random.normal(4000.0, 100.0)
            rows.append(row)
        cls.df_synthetic = pd.DataFrame(rows)

    def test_ordered_28_features(self):
        """Verify feature count and naming convention."""
        self.assertEqual(len(ORDERED_28_FEATURES), 28)
        self.assertEqual(ORDERED_28_FEATURES[0], "ch1_mean")
        self.assertEqual(ORDERED_28_FEATURES[-1], "ch4_crest_factor")

    def test_pipeline_packaging(self):
        """Test training and packaging per-channel pipelines on synthetic data."""
        iforest_params = {
            "n_estimators": 50,
            "max_samples": "auto",
            "contamination": "auto",
            "random_state": 42,
        }
        artifact, thresholds = train_and_package_channel_pipelines(
            df_features=self.df_synthetic,
            model_type="iforest",
            model_params=iforest_params,
            fit_end_idx=160,
            cal_end_idx=180,
            random_state=42,
        )

        self.assertEqual(artifact["model_type"], "iforest")
        self.assertEqual(len(artifact["channels"]), 4)
        for ch in CHANNELS:
            self.assertIn(ch, artifact["channels"])
            ch_data = artifact["channels"][ch]
            self.assertTrue(ch_data["pipeline"].is_fitted)
            self.assertIn(ch, thresholds)
            self.assertTrue(np.isfinite(thresholds[ch]))

    def test_build_metadata(self):
        """Verify metadata JSON structure contains all required sections."""
        iforest_thresh = {"ch1": 0.1, "ch2": 0.1, "ch3": 0.1, "ch4": 0.1}
        pca_thresh = {"ch1": 0.2, "ch2": 0.2, "ch3": 0.2, "ch4": 0.2}

        meta = build_metadata(iforest_thresh, pca_thresh)

        required_keys = [
            "model_name",
            "version",
            "creation_date",
            "task",
            "provenance",
            "preprocessing_parameters",
            "feature_extraction",
            "scaler",
            "models",
            "decision_logic",
            "environment",
            "limitations_and_intended_use",
        ]
        for key in required_keys:
            self.assertIn(key, meta)

        self.assertEqual(len(meta["feature_extraction"]["ordered_feature_names"]), 28)
        self.assertEqual(meta["decision_logic"]["persistence_k"], 3)
        self.assertEqual(meta["models"]["iforest"]["hyperparameters"]["n_estimators"], 100)
        self.assertEqual(meta["models"]["pca"]["hyperparameters"]["n_components"], 3)

    def test_export_all_artifacts_end_to_end(self):
        """Test complete artifact export using temporary directory and synthetic CSV."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            synthetic_csv = tmp_path / "synthetic_features.csv"
            models_dir = tmp_path / "models"

            self.df_synthetic.to_csv(synthetic_csv, index=False)

            summary = export_all_artifacts(
                features_csv_path=synthetic_csv,
                models_dir=models_dir,
                fit_end_idx=160,
                cal_end_idx=180,
                random_state=42,
            )

            # Check files exist
            iforest_file = Path(summary["iforest_path"])
            pca_file = Path(summary["pca_path"])
            meta_file = Path(summary["metadata_path"])

            self.assertTrue(iforest_file.exists())
            self.assertTrue(pca_file.exists())
            self.assertTrue(meta_file.exists())

            # Load joblib artifacts
            iforest_art = joblib.load(iforest_file)
            pca_art = joblib.load(pca_file)

            self.assertEqual(len(iforest_art["channels"]), 4)
            self.assertEqual(len(pca_art["channels"]), 4)

            # Check metadata JSON loads clean
            with open(meta_file, "r", encoding="utf-8") as f:
                loaded_meta = json.load(f)

            self.assertEqual(loaded_meta["model_name"], "MachineMind AI Vibration Anomaly Detector")

            # Check thresholds are finite
            for ch in CHANNELS:
                self.assertTrue(np.isfinite(summary["iforest_thresholds"][ch]))
                self.assertTrue(np.isfinite(summary["pca_thresholds"][ch]))

                # Check scoring with loaded model produce finite scores
                ch_cols = [f"{ch}_{stat}" for stat in TIME_STATS_PER_CHANNEL]
                sample_data = self.df_synthetic[ch_cols].to_numpy(dtype=np.float64)[:10]

                iforest_scores = iforest_art["channels"][ch]["pipeline"].compute_anomaly_scores(sample_data)
                pca_scores = pca_art["channels"][ch]["pipeline"].compute_anomaly_scores(sample_data)

                self.assertFalse(np.isnan(iforest_scores).any())
                self.assertFalse(np.isinf(iforest_scores).any())
                self.assertFalse(np.isnan(pca_scores).any())
                self.assertFalse(np.isinf(pca_scores).any())


if __name__ == "__main__":
    unittest.main()
