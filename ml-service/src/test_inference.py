"""
test_inference.py - Unit Test Suite for Anomaly Inference Engine.

Phase 10: Model Packaging - Step 2
Target Dataset: NASA IMS Bearing Dataset (Set 2)

Verifies:
- Artifact loading (IForest & PCA)
- Raw snapshot input validation (dimensionality, NaN, Inf, non-numeric, 1D ambiguous arrays)
- Feature ordering alignment (28 time-domain features)
- Inference scoring correctness (finite values, JSON-serializable output)
- Stateful k=3 persistence filter sequence (False -> count 0, True -> count 1,2,3,4, False -> count 0)
- Logical OR system alert policy across channels
- reset_state() counter resetting
- Stateless mode (stateful=False) behavior
- No model fitting calls during inference
- Reference score consistency against features_set2.csv (tolerance <= 1e-5)
"""

import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.data_loading import DEFAULT_RAW_SET2_PATH, get_snapshot_files, load_snapshot
from src.inference import CHANNELS, AnomalyInferenceEngine


class TestAnomalyInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load real packaged metadata and create sample valid snapshot array (20480, 4)."""
        cls.models_dir = Path("models")
        meta_path = cls.models_dir / "model_metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            cls.metadata = json.load(f)

        # Create valid synthetic snapshot array (20480, 4)
        np.random.seed(42)
        cls.sample_valid_snapshot = np.random.normal(0, 0.05, size=(20480, 4))

    def setUp(self):
        """Initialize fresh inference engines for each test."""
        self.engine_iforest = AnomalyInferenceEngine(
            artifacts_dir=self.models_dir, model_type="iforest", stateful=True
        )
        self.engine_pca = AnomalyInferenceEngine(
            artifacts_dir=self.models_dir, model_type="pca", stateful=True
        )

    # -------------------------------------------------------------
    # A. Artifact Loading Tests
    # -------------------------------------------------------------
    def test_artifact_loading(self):
        """Verify IForest and PCA engines load artifacts and metadata cleanly."""
        self.assertEqual(self.engine_iforest.model_type, "iforest")
        self.assertEqual(self.engine_pca.model_type, "pca")
        self.assertEqual(len(self.engine_iforest.ordered_feature_names), 28)
        self.assertEqual(len(self.engine_pca.ordered_feature_names), 28)

        for ch in CHANNELS:
            self.assertIn(ch, self.engine_iforest.thresholds)
            self.assertIn(ch, self.engine_pca.thresholds)
            self.assertTrue(np.isfinite(self.engine_iforest.thresholds[ch]))
            self.assertTrue(np.isfinite(self.engine_pca.thresholds[ch]))

    def test_invalid_model_type(self):
        """Verify invalid model_type raises ValueError."""
        with self.assertRaises(ValueError):
            AnomalyInferenceEngine(artifacts_dir=self.models_dir, model_type="invalid_model")

    # -------------------------------------------------------------
    # B. Input Validation Tests
    # -------------------------------------------------------------
    def test_input_validation_valid(self):
        """Verify valid (20480, 4) numpy array and DataFrame are accepted."""
        valid_arr = self.engine_iforest.validate_raw_snapshot(self.sample_valid_snapshot)
        self.assertEqual(valid_arr.shape, (20480, 4))

        df_valid = pd.DataFrame(self.sample_valid_snapshot)
        valid_arr_df = self.engine_iforest.validate_raw_snapshot(df_valid)
        self.assertEqual(valid_arr_df.shape, (20480, 4))

    def test_input_validation_invalid_rows(self):
        """Verify array with wrong row count (e.g. 1000 rows) is rejected."""
        wrong_rows = np.random.normal(0, 0.05, size=(1000, 4))
        with self.assertRaises(ValueError):
            self.engine_iforest.validate_raw_snapshot(wrong_rows)

    def test_input_validation_invalid_channels(self):
        """Verify array with wrong channel count (e.g. 3 channels) is rejected."""
        wrong_cols = np.random.normal(0, 0.05, size=(20480, 3))
        with self.assertRaises(ValueError):
            self.engine_iforest.validate_raw_snapshot(wrong_cols)

    def test_input_validation_1d_ambiguous(self):
        """Verify 1D ambiguous input is rejected."""
        arr_1d = np.random.normal(0, 0.05, size=(20480,))
        with self.assertRaises(ValueError):
            self.engine_iforest.validate_raw_snapshot(arr_1d)

    def test_input_validation_nan(self):
        """Verify input containing NaN is rejected."""
        nan_arr = self.sample_valid_snapshot.copy()
        nan_arr[10, 2] = np.nan
        with self.assertRaises(ValueError):
            self.engine_iforest.validate_raw_snapshot(nan_arr)

    def test_input_validation_inf(self):
        """Verify input containing Inf is rejected."""
        inf_arr = self.sample_valid_snapshot.copy()
        inf_arr[50, 1] = np.inf
        with self.assertRaises(ValueError):
            self.engine_iforest.validate_raw_snapshot(inf_arr)

    def test_input_validation_non_numeric(self):
        """Verify non-numeric string array is rejected."""
        str_arr = np.full((20480, 4), "text")
        with self.assertRaises(TypeError):
            self.engine_iforest.validate_raw_snapshot(str_arr)

    # -------------------------------------------------------------
    # C & D. Inference Scoring & Output Schema Tests
    # -------------------------------------------------------------
    def test_predict_snapshot_output_schema(self):
        """Verify predict_snapshot produces valid output dictionary with native types."""
        result = self.engine_iforest.predict_snapshot(
            self.sample_valid_snapshot, timestamp="2004-02-12 10:32:39"
        )

        self.assertEqual(result["timestamp"], "2004-02-12 10:32:39")
        self.assertEqual(result["model_type"], "iforest")
        self.assertEqual(len(result["channel_scores"]), 4)
        self.assertEqual(len(result["channel_thresholds"]), 4)
        self.assertEqual(len(result["channel_raw_flags"]), 4)
        self.assertEqual(len(result["channel_sustained_flags"]), 4)
        self.assertIsInstance(result["system_alert"], bool)

        # JSON serializability test
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)

        for ch in CHANNELS:
            self.assertTrue(np.isfinite(result["channel_scores"][ch]))
            self.assertIsInstance(result["channel_scores"][ch], float)
            self.assertIsInstance(result["channel_thresholds"][ch], float)
            self.assertIsInstance(result["channel_raw_flags"][ch], bool)
            self.assertIsInstance(result["channel_sustained_flags"][ch], bool)
            self.assertIsInstance(result["state_consecutive_counts"][ch], int)

    # -------------------------------------------------------------
    # E. Persistence Sequence Tests
    # -------------------------------------------------------------
    def test_persistence_sequence(self):
        """Verify k=3 persistence sequence: False->c0, True->c1, True->c2, True->c3(sustained), True->c4(sustained), False->c0(not sustained)."""
        engine = self.engine_iforest
        engine.reset_state()

        # Mock raw flags sequence for ch1 while keeping ch2..ch4 False
        # Sequence: False, True, True, True, True, False
        flag_sequence = [False, True, True, True, True, False]
        expected_counts = [0, 1, 2, 3, 4, 0]
        expected_sustained = [False, False, False, True, True, False]

        for step, (raw_flag, exp_count, exp_sust) in enumerate(
            zip(flag_sequence, expected_counts, expected_sustained)
        ):
            # Override threshold check for ch1 using mock
            engine.thresholds["ch1"] = 0.0 if raw_flag else 999.0
            engine.thresholds["ch2"] = 999.0
            engine.thresholds["ch3"] = 999.0
            engine.thresholds["ch4"] = 999.0

            res = engine.predict_snapshot(self.sample_valid_snapshot)

            self.assertEqual(
                res["state_consecutive_counts"]["ch1"],
                exp_count,
                f"Step {step} count mismatch",
            )
            self.assertEqual(
                res["channel_sustained_flags"]["ch1"],
                exp_sust,
                f"Step {step} sustained flag mismatch",
            )

    # -------------------------------------------------------------
    # F. Logical OR System Alert Tests
    # -------------------------------------------------------------
    def test_logical_or_system_alert(self):
        """Verify system_alert is True if ANY channel is sustained, False if no channel is sustained."""
        engine = self.engine_iforest
        engine.reset_state()

        # Set all thresholds high initially
        for ch in CHANNELS:
            engine.thresholds[ch] = 999.0

        # 1. Zero sustained channels -> System alert False
        res1 = engine.predict_snapshot(self.sample_valid_snapshot)
        self.assertFalse(res1["system_alert"])

        # 2. Trigger k=3 exceedances on ch3 only by setting ch3 threshold negative
        engine.thresholds["ch3"] = -999.0

        res_step1 = engine.predict_snapshot(self.sample_valid_snapshot)
        res_step2 = engine.predict_snapshot(self.sample_valid_snapshot)
        res_step3 = engine.predict_snapshot(self.sample_valid_snapshot)

        self.assertFalse(res_step1["system_alert"])
        self.assertFalse(res_step2["system_alert"])
        self.assertTrue(res_step3["system_alert"])  # 3rd consecutive -> system alert True
        self.assertTrue(res_step3["channel_sustained_flags"]["ch3"])
        self.assertFalse(res_step3["channel_sustained_flags"]["ch1"])


    # -------------------------------------------------------------
    # G. reset_state() Test
    # -------------------------------------------------------------
    def test_reset_state(self):
        """Verify reset_state() resets all channel counters to zero."""
        engine = self.engine_iforest
        engine.consecutive_exceedances = {"ch1": 5, "ch2": 2, "ch3": 0, "ch4": 1}
        engine.reset_state()

        for ch in CHANNELS:
            self.assertEqual(engine.consecutive_exceedances[ch], 0)

    # -------------------------------------------------------------
    # H. Stateless Mode Test
    # -------------------------------------------------------------
    def test_stateless_mode(self):
        """Verify stateful=False does not accumulate state across calls."""
        engine = AnomalyInferenceEngine(
            artifacts_dir=self.models_dir, model_type="iforest", stateful=False
        )

        # Force raw flag exceedance on ch1
        engine.thresholds["ch1"] = 0.0

        for _ in range(5):
            res = engine.predict_snapshot(self.sample_valid_snapshot)
            self.assertEqual(res["state_consecutive_counts"]["ch1"], 0)
            self.assertFalse(res["channel_sustained_flags"]["ch1"])
            self.assertFalse(res["system_alert"])

    # -------------------------------------------------------------
    # I. No Model Fitting Verification Test
    # -------------------------------------------------------------
    def test_no_fitting_during_inference(self):
        """Verify predict_snapshot does not call fit() on any scaler or model."""
        engine = self.engine_iforest

        for ch in CHANNELS:
            pipeline = engine.artifact["channels"][ch]["pipeline"]
            pipeline.fit = MagicMock(side_effect=RuntimeError("fit() must NOT be called during inference!"))
            pipeline.scaler.fit = MagicMock(side_effect=RuntimeError("scaler.fit() must NOT be called during inference!"))
            pipeline.model.fit = MagicMock(side_effect=RuntimeError("model.fit() must NOT be called during inference!"))

        # Should execute cleanly without triggering fit mock side effects
        res = engine.predict_snapshot(self.sample_valid_snapshot)
        self.assertEqual(len(res["channel_scores"]), 4)

    # -------------------------------------------------------------
    # J. Reference Consistency Check (tolerance <= 1e-5)
    # -------------------------------------------------------------
    def test_reference_score_consistency(self):
        """Compare inference engine scores on real raw snapshots 0..4 against reference pipeline scores from features_set2.csv."""
        files = get_snapshot_files(DEFAULT_RAW_SET2_PATH)
        df_csv = pd.read_csv("data/processed/features_set2.csv")

        max_diff_iforest = 0.0
        max_diff_pca = 0.0

        for i in range(5):
            df_snap = load_snapshot(files[i])
            res_if = self.engine_iforest.predict_snapshot(df_snap)
            res_pca = self.engine_pca.predict_snapshot(df_snap)

            for ch in CHANNELS:
                ch_stats = ["mean", "std", "rms", "p2p", "skewness", "kurtosis", "crest_factor"]
                ch_cols = [f"{ch}_{s}" for s in ch_stats]

                x_csv = df_csv.iloc[i:i + 1][ch_cols].to_numpy(dtype=np.float64)

                ref_score_if = float(
                    self.engine_iforest.artifact["channels"][ch]["pipeline"].compute_anomaly_scores(x_csv)[0]
                )
                ref_score_pca = float(
                    self.engine_pca.artifact["channels"][ch]["pipeline"].compute_anomaly_scores(x_csv)[0]
                )

                diff_if = abs(res_if["channel_scores"][ch] - ref_score_if)
                diff_pca = abs(res_pca["channel_scores"][ch] - ref_score_pca)

                max_diff_iforest = max(max_diff_iforest, diff_if)
                max_diff_pca = max(max_diff_pca, diff_pca)

        print(f"\n[Reference Consistency Check] Max Abs Difference - IForest: {max_diff_iforest:.2e}, PCA: {max_diff_pca:.2e}")
        self.assertLessEqual(max_diff_iforest, 1e-5, f"IForest score diff {max_diff_iforest} exceeds 1e-5")
        self.assertLessEqual(max_diff_pca, 1e-5, f"PCA score diff {max_diff_pca} exceeds 1e-5")


if __name__ == "__main__":
    unittest.main()
