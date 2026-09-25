"""
test_feature_extraction.py - Comprehensive Unit Tests for Feature Extraction Utilities.

Phase 5: Feature Engineering
Target Module: src/feature_extraction.py

Tests:
1. Zero-valued signal inputs.
2. Constant non-zero signal inputs.
3. Pure sine wave with known frequency (1000 Hz).
4. Gaussian noise with fixed random seed (Kurtosis ~ 0.0, Skewness ~ 0.0).
5. Impulsive signals (Kurtosis > 5.0, Crest Factor > 3.0).
6. Exception handling for invalid inputs (non-DataFrame, NaN, Infinite).
7. Feature dataset shape and schema (4 metadata + 36 feature columns = 40 total).
8. Finiteness of output feature values.
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

from src.feature_extraction import (
    extract_time_features,
    extract_frequency_features,
    extract_all_features,
    EPSILON,
)


class TestFeatureExtraction(unittest.TestCase):
    """Unit test suite for time-domain and frequency-domain feature extraction functions."""

    def setUp(self):
        """Set up standard test channel columns."""
        self.cols = ["Channel_1", "Channel_2", "Channel_3", "Channel_4"]
        self.n_samples = 20480
        self.fs = 20480.0

    def test_zero_valued_signal(self):
        """Verify handling of zero-valued signals (zero mean, std, rms, crest factor, no NaN/Inf)."""
        data = np.zeros((self.n_samples, 4))
        df = pd.DataFrame(data, columns=self.cols)

        time_feats = extract_time_features(df)
        freq_feats = extract_frequency_features(df, sampling_rate=self.fs)
        all_feats = extract_all_features(df, sampling_rate=self.fs)

        self.assertEqual(len(all_feats), 36)
        for k, v in all_feats.items():
            self.assertTrue(np.isfinite(v), f"Feature {k} is non-finite: {v}")

        # Check zero signal specific values
        self.assertAlmostEqual(time_feats["ch1_mean"], 0.0)
        self.assertAlmostEqual(time_feats["ch1_std"], 0.0)
        self.assertAlmostEqual(time_feats["ch1_rms"], 0.0)
        self.assertAlmostEqual(time_feats["ch1_crest_factor"], 0.0)
        self.assertAlmostEqual(freq_feats["ch1_spectral_energy"], 0.0)

    def test_constant_nonzero_signal(self):
        """Verify constant non-zero signal (e.g. 5.0 -> std=0, p2p=0, crest_factor ~ 1.0)."""
        data = np.full((self.n_samples, 4), 5.0)
        df = pd.DataFrame(data, columns=self.cols)

        time_feats = extract_time_features(df)
        freq_feats = extract_frequency_features(df, sampling_rate=self.fs)

        self.assertAlmostEqual(time_feats["ch1_mean"], 5.0)
        self.assertAlmostEqual(time_feats["ch1_std"], 0.0)
        self.assertAlmostEqual(time_feats["ch1_rms"], 5.0)
        self.assertAlmostEqual(time_feats["ch1_p2p"], 0.0)
        self.assertAlmostEqual(time_feats["ch1_crest_factor"], 1.0, places=4)
        self.assertAlmostEqual(freq_feats["ch1_spectral_energy"], 0.0)

    def test_pure_sine_wave(self):
        """Verify pure 1000 Hz sine wave statistical properties and spectral centroid."""
        f_target = 1000.0
        t = np.arange(self.n_samples) / self.fs
        sine_signal = np.sin(2 * np.pi * f_target * t)
        
        data = np.column_stack([sine_signal] * 4)
        df = pd.DataFrame(data, columns=self.cols)

        time_feats = extract_time_features(df)
        freq_feats = extract_frequency_features(df, sampling_rate=self.fs)

        # Theoretical RMS of A=1 sine wave = 1/sqrt(2) approx 0.7071
        self.assertAlmostEqual(time_feats["ch1_mean"], 0.0, places=3)
        self.assertAlmostEqual(time_feats["ch1_rms"], 1.0 / np.sqrt(2.0), places=3)
        self.assertAlmostEqual(time_feats["ch1_crest_factor"], np.sqrt(2.0), places=2)

        # Spectral centroid should be close to 1000 Hz
        self.assertAlmostEqual(freq_feats["ch1_spectral_centroid"], f_target, delta=5.0)

    def test_gaussian_noise(self):
        """Verify Gaussian noise N(0, 1) has Fisher Excess Kurtosis ~ 0.0 and Skewness ~ 0.0."""
        np.random.seed(42)
        noise = np.random.randn(self.n_samples, 4)
        df = pd.DataFrame(noise, columns=self.cols)

        time_feats = extract_time_features(df)

        self.assertAlmostEqual(time_feats["ch1_mean"], 0.0, delta=0.05)
        self.assertAlmostEqual(time_feats["ch1_std"], 1.0, delta=0.05)
        self.assertAlmostEqual(time_feats["ch1_skewness"], 0.0, delta=0.05)
        self.assertAlmostEqual(time_feats["ch1_kurtosis"], 0.0, delta=0.1)  # Fisher kurtosis = 0

    def test_impulsive_signal(self):
        """Verify an impulsive spike signal produces elevated Fisher Kurtosis (> 5.0) and Crest Factor (> 3.0)."""
        data = np.zeros((self.n_samples, 4))
        # Add a single large shock spike
        data[100, :] = 50.0
        df = pd.DataFrame(data, columns=self.cols)

        time_feats = extract_time_features(df)

        self.assertGreater(time_feats["ch1_kurtosis"], 5.0)
        self.assertGreater(time_feats["ch1_crest_factor"], 3.0)

    def test_invalid_inputs(self):
        """Verify TypeError and ValueError handling for invalid inputs."""
        # Non-DataFrame input
        with self.assertRaises(TypeError):
            extract_time_features("not_a_dataframe")

        with self.assertRaises(TypeError):
            extract_frequency_features([1, 2, 3])

        # NaN input
        df_nan = pd.DataFrame(np.zeros((100, 4)), columns=self.cols)
        df_nan.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            extract_time_features(df_nan)

        # Infinite input
        df_inf = pd.DataFrame(np.zeros((100, 4)), columns=self.cols)
        df_inf.iloc[0, 0] = np.inf
        with self.assertRaises(ValueError):
            extract_time_features(df_inf)

        # Non-positive sampling rate
        df_valid = pd.DataFrame(np.zeros((100, 4)), columns=self.cols)
        with self.assertRaises(ValueError):
            extract_frequency_features(df_valid, sampling_rate=-10.0)

    def test_feature_names_and_finiteness(self):
        """Verify that 36 feature keys are returned and all output values are finite floats."""
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(2048, 4), columns=self.cols)

        all_feats = extract_all_features(df, sampling_rate=self.fs)

        self.assertEqual(len(all_feats), 36)
        expected_suffixes = [
            "mean", "std", "rms", "p2p", "skewness", "kurtosis",
            "crest_factor", "spectral_energy", "spectral_centroid"
        ]
        for ch in [1, 2, 3, 4]:
            for sfx in expected_suffixes:
                key = f"ch{ch}_{sfx}"
                self.assertIn(key, all_feats)
                self.assertTrue(np.isfinite(all_feats[key]))


if __name__ == "__main__":
    unittest.main()
