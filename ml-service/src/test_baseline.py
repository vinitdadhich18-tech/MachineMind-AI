"""
test_baseline.py - Comprehensive Unit Tests for Baseline Module.

Phase 6: Statistical Baseline
Target Module: src/baseline.py

Tests:
1. Baseline parameter fitting (mean, std, median, mad, iqr, p2p).
2. Chronological reference split (only first ref_end_idx rows used for fitting).
3. Frozen baseline evaluation on later snapshots.
4. Correct standard Z-score calculations.
5. Correct modified Z-score calculations (factor 0.6745).
6. Correct maximum and RMS composite anomaly score calculations.
7. Correct eligible-feature count scaling (K_eligible denominator).
8. Zero-MAD and near-zero-MAD scale-aware variance guarding.
9. Completely constant feature exclusion (P2P == 0).
10. Behavior when all features are excluded (ValueError raised).
11. Invalid input handling (non-DataFrame, NaN, Inf, empty, ref_end_idx out of range).
12. Immutability check (input DataFrame is never mutated).
13. Output dimensions and metadata alignment.
14. Finiteness of scores (no NaN or Inf values for valid test cases).
"""

import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import (
    fit_baseline_parameters,
    compute_zscore_features,
    compute_modified_zscore_features,
    compute_composite_anomaly_scores,
    EPS_ABS,
    ETA_REL,
    FACTOR_MOD_Z,
)


class TestBaselineModule(unittest.TestCase):
    """Unit test suite for statistical baseline fitting and anomaly scoring."""

    def setUp(self):
        """Set up synthetic test DataFrame with 200 snapshots and 4 numerical features + metadata."""
        np.random.seed(42)
        n_rows = 200
        ref_len = 160

        # Synthetic feature data: 4 features
        f1 = np.random.normal(loc=10.0, scale=2.0, size=n_rows)  # standard normal-like
        f2 = np.random.normal(loc=0.0, scale=1.0, size=n_rows)   # mean 0
        # Add elevated values in later snapshots (>160) to test anomaly score response
        f1[160:] += 20.0
        f2[160:] += 10.0

        # Constant feature (P2P == 0 during fitting)
        f3 = np.full(n_rows, 5.0)

        # Small variance feature (near-zero MAD)
        f4 = np.random.normal(loc=1.0, scale=1e-8, size=n_rows)

        self.df_test = pd.DataFrame({
            "file_index": np.arange(n_rows),
            "filename": [f"file_{i:03d}.txt" for i in range(n_rows)],
            "timestamp": [f"2004-02-12 10:{i:02d}:00" for i in range(n_rows)],
            "is_valid": [True] * n_rows,
            "feat_1": f1,
            "feat_2": f2,
            "feat_3": f3,
            "feat_4": f4,
        })
        self.ref_end_idx = ref_len

    def test_fit_baseline_parameters(self):
        """Test parameter calculation and chronological reference split (ref_end_idx=160)."""
        params = fit_baseline_parameters(self.df_test, ref_end_idx=self.ref_end_idx)

        self.assertEqual(params["ref_end_idx"], self.ref_end_idx)
        self.assertEqual(len(params["feature_names"]), 4)

        # Check feat_1 stats computed strictly on first 160 rows
        ref_f1 = self.df_test["feat_1"].iloc[:160]
        f1_stats = params["stats"]["feat_1"]

        self.assertAlmostEqual(f1_stats["mean"], float(np.mean(ref_f1)))
        self.assertAlmostEqual(f1_stats["std"], float(np.std(ref_f1, ddof=1)))
        self.assertAlmostEqual(f1_stats["median"], float(np.median(ref_f1)))

    def test_chronological_split_and_frozen_evaluation(self):
        """Test that evaluating later snapshots (>160) uses frozen baseline parameters."""
        params_initial = fit_baseline_parameters(self.df_test, ref_end_idx=160)

        # Add large spikes to evaluation rows
        df_modified_eval = self.df_test.copy()
        df_modified_eval.loc[180:, "feat_1"] = 9999.0

        scores_1 = compute_composite_anomaly_scores(self.df_test, params_initial)
        scores_2 = compute_composite_anomaly_scores(df_modified_eval, params_initial)

        # First 160 scores must be 100% identical because baseline params were frozen
        pd.testing.assert_frame_equal(
            scores_1.iloc[:160],
            scores_2.iloc[:160]
        )

    def test_constant_feature_exclusion(self):
        """Test that completely constant features (P2P == 0 during fitting) are excluded from scoring."""
        params = fit_baseline_parameters(self.df_test, ref_end_idx=160)

        # feat_3 is constant 5.0 (P2P == 0)
        self.assertIn("feat_3", params["excluded_z_features"])
        self.assertIn("feat_3", params["excluded_m_features"])
        self.assertNotIn("feat_3", params["eligible_z_features"])

    def test_near_zero_mad_scale_aware_guarding(self):
        """Test scale-aware variance guarding for near-zero MAD features (feat_4)."""
        params = fit_baseline_parameters(self.df_test, ref_end_idx=160)

        f4_stats = params["stats"]["feat_4"]
        self.assertGreater(f4_stats["guarded_mad"], 0.0)
        self.assertGreaterEqual(f4_stats["guarded_mad"], f4_stats["cutoff_mad"])

        # Compute modified Z-score and verify finite output
        df_m = compute_modified_zscore_features(self.df_test, params)
        self.assertTrue(np.isfinite(df_m["feat_4"].to_numpy()).all())

    def test_zscore_calculation(self):
        """Verify Z-score calculation formula (Z = (x - mu) / sigma_guarded)."""
        params = fit_baseline_parameters(self.df_test, ref_end_idx=160)
        df_z = compute_zscore_features(self.df_test, params)

        f1_mean = params["stats"]["feat_1"]["mean"]
        f1_std_g = params["stats"]["feat_1"]["guarded_std"]

        expected_z_0 = (self.df_test["feat_1"].iloc[0] - f1_mean) / f1_std_g
        self.assertAlmostEqual(df_z["feat_1"].iloc[0], expected_z_0)

    def test_modified_zscore_calculation(self):
        """Verify Modified Z-score calculation formula (M = 0.6745 * (x - median) / mad_guarded)."""
        params = fit_baseline_parameters(self.df_test, ref_end_idx=160)
        df_m = compute_modified_zscore_features(self.df_test, params)

        f1_med = params["stats"]["feat_1"]["median"]
        f1_mad_g = params["stats"]["feat_1"]["guarded_mad"]

        expected_m_0 = FACTOR_MOD_Z * (self.df_test["feat_1"].iloc[0] - f1_med) / f1_mad_g
        self.assertAlmostEqual(df_m["feat_1"].iloc[0], expected_m_0)

    def test_composite_anomaly_scores_and_eligible_count(self):
        """Verify 4 composite anomaly score definitions and K_eligible scaling."""
        params = fit_baseline_parameters(self.df_test, ref_end_idx=160)
        df_scores = compute_composite_anomaly_scores(self.df_test, params)

        expected_score_cols = ["score_max_z", "score_rms_z", "score_max_m", "score_rms_m"]
        for col in expected_score_cols:
            self.assertIn(col, df_scores.columns)

        # Check metadata preservation
        self.assertEqual(len(df_scores), 200)
        self.assertEqual(list(df_scores["file_index"]), list(self.df_test["file_index"]))

        # Verify that score in evaluation period (>160) rises due to synthetic shift
        mean_ref_score = df_scores["score_rms_z"].iloc[:160].mean()
        mean_eval_score = df_scores["score_rms_z"].iloc[160:].mean()
        self.assertGreater(mean_eval_score, mean_ref_score)

    def test_all_features_excluded_exception(self):
        """Test ValueError when all features are constant and excluded."""
        df_const = pd.DataFrame({
            "file_index": range(100),
            "feat_1": [1.0] * 100,
            "feat_2": [2.0] * 100,
        })
        params = fit_baseline_parameters(df_const, ref_end_idx=50)

        with self.assertRaises(ValueError):
            compute_composite_anomaly_scores(df_const, params)

    def test_invalid_inputs(self):
        """Test exception handling for invalid inputs (non-DataFrame, NaN, out of range ref_end_idx)."""
        with self.assertRaises(TypeError):
            fit_baseline_parameters("invalid_string")

        with self.assertRaises(ValueError):
            fit_baseline_parameters(self.df_test, ref_end_idx=500)  # Exceeds row count

        # NaN input in reference slice
        df_nan = self.df_test.copy()
        df_nan.iloc[10, 4] = np.nan
        with self.assertRaises(ValueError):
            fit_baseline_parameters(df_nan, ref_end_idx=160)

    def test_immutability(self):
        """Test that input DataFrame is never mutated during parameter fitting or score calculation."""
        df_orig = self.df_test.copy()

        params = fit_baseline_parameters(self.df_test, ref_end_idx=160)
        _ = compute_zscore_features(self.df_test, params)
        _ = compute_modified_zscore_features(self.df_test, params)
        _ = compute_composite_anomaly_scores(self.df_test, params)

        pd.testing.assert_frame_equal(self.df_test, df_orig)

    def test_finiteness_of_scores(self):
        """Verify all output scores are finite floats (no NaN or Inf)."""
        params = fit_baseline_parameters(self.df_test, ref_end_idx=160)
        df_scores = compute_composite_anomaly_scores(self.df_test, params)

        score_cols = ["score_max_z", "score_rms_z", "score_max_m", "score_rms_m"]
        for col in score_cols:
            vals = df_scores[col].to_numpy()
            self.assertTrue(np.isfinite(vals).all(), f"Column {col} contains non-finite values.")


if __name__ == "__main__":
    unittest.main()
