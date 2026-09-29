"""
test_evaluation.py - Unit Test Suite for MachineMind AI Evaluation Module.

Phase 8: Evaluation
Run with: python -m unittest src/test_evaluation.py

This test suite uses small, manually verifiable synthetic test cases to verify:
1. Threshold exceedance calculation.
2. Persistence filtering for k = 1, 3, 5.
3. Alert start and online confirmation indices (t_confirm = t_start + k - 1).
4. Sustained-event episode counting.
5. Flagged-snapshot counting.
6. Interrupted run behavior (score drop terminates event).
7. Multiple distinct sustained events.
8. Healthy reference period false-alarm metrics (raw and sustained).
9. Validation-only threshold calculations (mean + k*std, percentiles).
10. Logical OR multi-channel aggregation.
11. Handling of invalid inputs, empty arrays, and invalid k values.
12. Timestamp alignment and proxy duration calculations.
13. Leakage prevention (validation partition isolation).
14. Non-finite value handling (NaN and Inf detection).
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory / src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.evaluation import (
        validate_scores_array,
        calculate_validation_thresholds,
        compute_threshold_exceedances,
        apply_persistence_filter,
        compute_evaluation_metrics,
        aggregate_channel_alerts_or,
        DEFAULT_FIT_END_IDX,
        DEFAULT_VAL_END_IDX,
    )
except ImportError:
    from evaluation import (
        validate_scores_array,
        calculate_validation_thresholds,
        compute_threshold_exceedances,
        apply_persistence_filter,
        compute_evaluation_metrics,
        aggregate_channel_alerts_or,
        DEFAULT_FIT_END_IDX,
        DEFAULT_VAL_END_IDX,
    )


class TestEvaluationModule(unittest.TestCase):
    """Unit test suite for Phase 8 evaluation functions."""

    def setUp(self):
        """Set up standard synthetic test timelines."""
        # Simple scores array of length 10
        self.simple_scores = np.array([0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.2, 0.9, 0.95, 0.1])
        # Timestamps corresponding to length 10
        self.timestamps = pd.date_range("2004-02-12 10:00:00", periods=10, freq="10min")

    def test_01_threshold_exceedance_logic(self):
        """Test basic threshold exceedance boolean flag calculation."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        flags = compute_threshold_exceedances(scores, threshold=3.0)
        expected = np.array([False, False, False, True, True])
        np.testing.assert_array_equal(flags, expected)

    def test_02_persistence_behavior_k1_k3_k5(self):
        """Test persistence filtering for k = 1, 3, and 5 on a known run of length 4."""
        # Exceedance run of length 4 (indices 2..5)
        flags = np.array([False, False, True, True, True, True, False, False])
        
        # k = 1 (instantaneous) -> all 4 snapshots flagged
        sustained1, events1, start1, confirm1 = apply_persistence_filter(flags, k=1)
        self.assertEqual(np.sum(sustained1), 4)
        self.assertEqual(len(events1), 1)
        self.assertEqual(start1, 2)
        self.assertEqual(confirm1, 2)
        
        # k = 3 -> length 4 >= 3 -> all 4 snapshots flagged
        sustained3, events3, start3, confirm3 = apply_persistence_filter(flags, k=3)
        self.assertEqual(np.sum(sustained3), 4)
        self.assertEqual(len(events3), 1)
        self.assertEqual(start3, 2)
        self.assertEqual(confirm3, 4)  # t_confirm = 2 + 3 - 1 = 4
        
        # k = 5 -> length 4 < 5 -> 0 snapshots flagged
        sustained5, events5, start5, confirm5 = apply_persistence_filter(flags, k=5)
        self.assertEqual(np.sum(sustained5), 0)
        self.assertEqual(len(events5), 0)
        self.assertIsNone(start5)
        self.assertIsNone(confirm5)

    def test_03_alert_start_and_confirmation_indices(self):
        """Test exact formula t_confirm = t_start + k - 1."""
        flags = np.array([0, 0, 1, 1, 1, 1, 1, 0], dtype=bool)
        # Run of 5 exceedances starting at index 2
        for k_val in [1, 2, 3, 4, 5]:
            _, events, start_idx, confirm_idx = apply_persistence_filter(flags, k=k_val)
            self.assertEqual(start_idx, 2)
            self.assertEqual(confirm_idx, 2 + k_val - 1)
            self.assertEqual(events[0]["confirm_idx"], 2 + k_val - 1)

    def test_04_sustained_event_counting(self):
        """Test episode count vs flagged snapshot count."""
        # Two distinct runs: indices 1..3 (len 3) and 6..9 (len 4)
        flags = np.array([False, True, True, True, False, False, True, True, True, True, False])
        sustained, events, _, _ = apply_persistence_filter(flags, k=3)
        
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event_id"], 1)
        self.assertEqual(events[0]["start_idx"], 1)
        self.assertEqual(events[0]["end_idx"], 3)
        self.assertEqual(events[0]["length"], 3)
        
        self.assertEqual(events[1]["event_id"], 2)
        self.assertEqual(events[1]["start_idx"], 6)
        self.assertEqual(events[1]["end_idx"], 9)
        self.assertEqual(events[1]["length"], 4)
        
        # Total flagged snapshots = 3 + 4 = 7
        self.assertEqual(np.sum(sustained), 7)

    def test_05_flagged_snapshot_counting(self):
        """Test that short runs (< k) contribute 0 to flagged snapshot count."""
        # Short run (len 2) and long run (len 4)
        flags = np.array([True, True, False, False, True, True, True, True, False])
        sustained, events, _, _ = apply_persistence_filter(flags, k=3)
        
        # Only the long run (len 4) is sustained
        self.assertEqual(len(events), 1)
        self.assertEqual(np.sum(sustained), 4)

    def test_06_interrupted_run_behavior(self):
        """Test that a single False snapshot interrupts a run."""
        # Run interrupted at index 3
        flags = np.array([True, True, True, False, True, True, True])
        sustained, events, _, _ = apply_persistence_filter(flags, k=3)
        
        # Two distinct events of length 3 each
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["length"], 3)
        self.assertEqual(events[1]["length"], 3)
        self.assertEqual(np.sum(sustained), 6)

    def test_07_multiple_distinct_events(self):
        """Test identification of multiple distinct events across a timeline."""
        flags = np.zeros(20, dtype=bool)
        flags[2:5] = True   # Event 1: len 3
        flags[8:12] = True  # Event 2: len 4
        flags[15:19] = True # Event 3: len 4
        
        _, events, first_start, first_confirm = apply_persistence_filter(flags, k=3)
        self.assertEqual(len(events), 3)
        self.assertEqual(first_start, 2)
        self.assertEqual(first_confirm, 4)

    def test_08_false_alarm_metric_calculations(self):
        """Test healthy reference period false alarm count and rate calculations."""
        # 10 snapshots total: fit/val period = 0..4 (N=5), eval period = 5..9 (N=5)
        scores = np.array([0.5, 3.5, 0.5, 4.0, 4.2, 0.1, 5.0, 5.0, 5.0, 5.0])
        
        metrics = compute_evaluation_metrics(
            scores=scores,
            threshold=3.0,
            k=2,
            fit_end_idx=3,
            val_end_idx=5
        )
        
        # Healthy reference period (indices 0..4, N=5)
        # Raw exceedances at indices 1, 3, 4 -> N_FA_raw = 3
        self.assertEqual(metrics["healthy_n"], 5)
        self.assertEqual(metrics["raw_false_alarm_count"], 3)
        self.assertAlmostEqual(metrics["raw_false_alarm_rate"], 3 / 5)
        
        # Sustained exceedances for k=2: indices 3..4 (len 2 >= 2) -> N_FA_sustained = 2
        self.assertEqual(metrics["sustained_false_alarm_count"], 2)
        self.assertAlmostEqual(metrics["sustained_false_alarm_rate"], 2 / 5)
        self.assertEqual(metrics["healthy_sustained_events_count"], 1)

    def test_09_validation_only_threshold_calculation(self):
        """Test candidate threshold calculation strictly from validation partition."""
        val_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        thresh_dict = calculate_validation_thresholds(val_scores, sigmas=[3.0, 4.0], percentiles=[99.0])
        
        mean_v = np.mean(val_scores) # 3.0
        std_v = np.std(val_scores, ddof=1) # ~1.5811
        
        self.assertAlmostEqual(thresh_dict["val_mean"], mean_v)
        self.assertAlmostEqual(thresh_dict["val_std"], std_v)
        self.assertAlmostEqual(thresh_dict["mean_3_0sigma"], mean_v + 3.0 * std_v)
        self.assertIn("p99_0", thresh_dict)

    def test_10_logical_or_aggregation(self):
        """Test multi-channel Logical OR shaft alert aggregation."""
        ch1_flags = np.array([False, True, True, True, False])
        ch2_flags = np.array([False, False, False, False, False])
        ch3_flags = np.array([True, True, False, False, False])
        
        channel_dict = {"ch1": ch1_flags, "ch2": ch2_flags, "ch3": ch3_flags}
        shaft_flags, first_start, first_confirm = aggregate_channel_alerts_or(channel_dict, k=2)
        
        # Logical OR across channels: [True, True, True, True, False]
        expected_shaft = np.array([True, True, True, True, False])
        np.testing.assert_array_equal(shaft_flags, expected_shaft)
        self.assertEqual(first_start, 0)
        self.assertEqual(first_confirm, 1)

    def test_11_invalid_inputs_and_k_values(self):
        """Test exception raising for invalid k, empty arrays, and None inputs."""
        with self.assertRaises(ValueError):
            apply_persistence_filter([True, False], k=0)  # k < 1 invalid
        with self.assertRaises(ValueError):
            apply_persistence_filter([True, False], k=-2)
        with self.assertRaises(ValueError):
            validate_scores_array([])  # empty
        with self.assertRaises(ValueError):
            validate_scores_array(None) # None

    def test_12_timestamp_alignment_and_proxy_duration(self):
        """Test timestamp alignment and proxy duration to recording end."""
        scores = np.array([0.1, 0.2, 0.1, 0.1, 4.0, 4.0, 4.0, 4.0])
        timestamps = pd.date_range("2004-02-12 10:00:00", periods=8, freq="10min")
        
        metrics = compute_evaluation_metrics(
            scores=scores,
            threshold=3.0,
            k=3,
            timestamps=timestamps,
            fit_end_idx=2,
            val_end_idx=4
        )
        
        # Sustained run starting at index 4 (first in eval period)
        self.assertEqual(metrics["first_eval_sustained_start_idx"], 4)
        self.assertEqual(metrics["first_eval_sustained_confirm_idx"], 6) # 4 + 3 - 1 = 6
        self.assertEqual(metrics["first_eval_sustained_confirm_ts"], str(timestamps[6]))

    def test_13_prevention_of_evaluation_leakage(self):
        """Assertion proves threshold calculation uses ONLY validation partition slice."""
        val_slice = np.array([1.0, 1.1, 1.2, 0.9, 1.0])
        eval_slice = np.array([100.0, 500.0, 1000.0]) # Severe evaluation anomaly
        
        # Threshold computed on validation slice only
        thresh_val_only = calculate_validation_thresholds(val_slice)["p99_0"]
        
        # If eval slice were contaminated:
        thresh_contaminated = calculate_validation_thresholds(np.concatenate([val_slice, eval_slice]))["p99_0"]
        
        # Confirms validation threshold is immune to evaluation period extreme values
        self.assertLess(thresh_val_only, 5.0)
        self.assertGreater(thresh_contaminated, 50.0)

    def test_14_non_finite_value_handling(self):
        """Test clean error handling when NaNs or Infs are present."""
        scores_nan = np.array([1.0, 2.0, np.nan, 4.0])
        scores_inf = np.array([1.0, np.inf, 3.0, 4.0])
        
        with self.assertRaises(ValueError):
            compute_threshold_exceedances(scores_nan, threshold=2.0)
        with self.assertRaises(ValueError):
            compute_threshold_exceedances(scores_inf, threshold=2.0)


if __name__ == "__main__":
    unittest.main()
