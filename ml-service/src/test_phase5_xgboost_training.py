"""
MachineMind AI — Phase 5 Test Suite
Supervised Dataset, RUL Target, Leakage Audit & XGBoost Model Verification

Tests:
1. Feature matrix shape, column count, and finiteness.
2. Causal feature extraction integrity.
3. Target derivation and capped RUL bounds.
4. Automated temporal leakage audit.
5. Preprocessor train-only fitting isolation.
6. Artifact bundle loadability & prediction schema validation.
7. Deterministic retraining reproducibility.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib
import xgboost as xgb

from src.canonical_feature_pipeline import CANONICAL_FEATURE_NAMES, CanonicalFeaturePipeline, FeatureScalerAdapter
from src.build_feature_matrix import load_or_build_feature_matrix
from src.leakage_audit import run_full_leakage_audit
from src.train_xgboost import train_and_evaluate_xgboost, MODELS_DIR


class TestPhase5SupervisedXGBoost:
    """Phase 5 verification suite for supervised dataset, leakage, and XGBoost model artifacts."""

    @classmethod
    def setup_class(cls):
        """Loads or builds feature matrix and runs training pipeline if artifacts missing."""
        cls.df_matrix = load_or_build_feature_matrix()
        cls.metadata_path = MODELS_DIR / "xgboost_model_metadata.json"
        cls.model_json_path = MODELS_DIR / "xgboost_rul_model.json"
        cls.scaler_path = MODELS_DIR / "xgboost_scaler.joblib"

        if not cls.metadata_path.exists() or not cls.model_json_path.exists():
            train_and_evaluate_xgboost()

    def test_feature_matrix_shape_and_columns(self):
        """Verifies 984 rows, 36 canonical features, metadata, and finite values."""
        assert len(self.df_matrix) == 984
        assert "snapshot_sequence" in self.df_matrix.columns
        assert "original_timestamp" in self.df_matrix.columns
        assert "linear_rul" in self.df_matrix.columns
        assert "capped_rul_400" in self.df_matrix.columns

        for col in CANONICAL_FEATURE_NAMES:
            assert col in self.df_matrix.columns
            assert not self.df_matrix[col].isna().any()
            assert not np.isinf(self.df_matrix[col]).any()

    def test_causal_feature_integrity(self):
        """Asserts feature extraction for snapshot t is independent of future snapshots."""
        pipeline = CanonicalFeaturePipeline()
        rng = np.random.RandomState(123)

        df_t = pd.DataFrame(rng.randn(20480, 4), columns=["ch1", "ch2", "ch3", "ch4"])
        v1 = pipeline.process_snapshot_dataframe(df_t, snapshot_sequence=5)

        # Process future snapshot
        df_t_plus_1 = pd.DataFrame(rng.randn(20480, 4), columns=["ch1", "ch2", "ch3", "ch4"])
        _ = pipeline.process_snapshot_dataframe(df_t_plus_1, snapshot_sequence=6)

        # Process snapshot t again
        v1_again = pipeline.process_snapshot_dataframe(df_t, snapshot_sequence=5)

        np.testing.assert_allclose(v1.feature_values, v1_again.feature_values, rtol=1e-12)

    def test_target_construction_and_bounds(self):
        """Verifies linear RUL formula and capped RUL upper/lower bounds."""
        # Snapshot 0 linear RUL must be 983, Snapshot 983 linear RUL must be 0
        assert self.df_matrix.loc[0, "linear_rul"] == 983
        assert self.df_matrix.loc[983, "linear_rul"] == 0

        # Capped RUL 400 bounds
        assert self.df_matrix["capped_rul_400"].max() == 400
        assert self.df_matrix["capped_rul_400"].min() == 0
        assert self.df_matrix.loc[0, "capped_rul_400"] == 400
        assert self.df_matrix.loc[983, "capped_rul_400"] == 0

    def test_automated_leakage_audit_passes(self):
        """Executes full temporal leakage audit and asserts all checks pass."""
        report = run_full_leakage_audit(self.df_matrix, CANONICAL_FEATURE_NAMES)
        assert report["all_passed"] is True
        assert report["splits_audit"]["is_disjoint"] is True
        assert report["splits_audit"]["is_ordered"] is True
        assert report["preprocessor_isolation_audit"]["train_scaler_fitted"] is True
        assert report["causality_audit"]["is_causal"] is True

    def test_preprocessor_train_only_isolation(self):
        """Verifies scaler fitted parameters differ when fit on train vs full matrix."""
        X_train = self.df_matrix.loc[0:600, CANONICAL_FEATURE_NAMES].values
        X_full = self.df_matrix[CANONICAL_FEATURE_NAMES].values

        scaler_train = FeatureScalerAdapter(scaler_type="robust")
        scaler_train.fit(X_train)

        scaler_full = FeatureScalerAdapter(scaler_type="robust")
        scaler_full.fit(X_full)

        assert not np.allclose(scaler_train.scaler.center_, scaler_full.scaler.center_, atol=1e-5)

    def test_artifact_bundle_integrity_and_loading(self):
        """Verifies native XGBoost model, scaler, and metadata artifact bundle loading."""
        assert self.model_json_path.exists()
        assert self.scaler_path.exists()
        assert self.metadata_path.exists()

        # Load Metadata
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["model_name"] == "MachineMind_XGBoost_RUL_Regressor"
        assert meta["feature_count"] == 36
        assert meta["raw_snapshot_count"] == 984

        # Load XGBoost Model
        model = xgb.XGBRegressor()
        model.load_model(self.model_json_path)

        # Load Scaler
        scaler = joblib.load(self.scaler_path)
        assert scaler.is_fitted is True

        # Test predict on 1 sample
        sample_features = self.df_matrix.loc[0:0, CANONICAL_FEATURE_NAMES].values
        sample_scaled = scaler.transform(sample_features)
        pred = model.predict(sample_scaled)

        assert pred.shape == (1,)
        assert np.isfinite(pred[0])

    def test_deterministic_retraining(self):
        """Verifies re-running training with seed 42 produces identical predictions."""
        X_train_raw = self.df_matrix.loc[0:600, CANONICAL_FEATURE_NAMES].values
        y_train = self.df_matrix.loc[0:600, "capped_rul_400"].values

        scaler1 = FeatureScalerAdapter(scaler_type="robust").fit(X_train_raw)
        X_train_scaled = scaler1.transform(X_train_raw)

        m1 = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=1)
        m1.fit(X_train_scaled, y_train)
        p1 = m1.predict(X_train_scaled)

        m2 = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=1)
        m2.fit(X_train_scaled, y_train)
        p2 = m2.predict(X_train_scaled)

        np.testing.assert_allclose(p1, p2, rtol=1e-6)
