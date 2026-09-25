"""
feature_extraction.py - Feature Extraction Utility Module for MachineMind AI.

Phase 5: Feature Engineering
Target Dataset: NASA IMS Bearing Dataset (Set 2)

This module provides modular, beginner-friendly helper functions to extract time-domain
and frequency-domain features from multi-channel vibration signal windows.

Calculated Features Per Channel (9 features):
----------------------------------------------
Time-Domain:
1. mean: Arithmetic mean (DC baseline offset).
2. std: Sample standard deviation with ddof=1 (AC signal dispersion around mean).
3. rms: Root Mean Square amplitude sqrt(mean(x^2)) (total signal energy).
4. p2p: Peak-to-peak amplitude max(x) - min(x) (dynamic range).
5. skewness: Third standardized moment (distribution asymmetry around mean).
6. kurtosis: Fisher's excess kurtosis (fourth standardized moment minus 3.0; Gaussian = 0.0).
7. crest_factor: Peak to RMS ratio max(|x|) / (rms + epsilon) (spikiness relative to energy).

Frequency-Domain:
8. spectral_energy: Hann-windowed FFT positive-frequency energy sum( |X(f_k)|^2 ).
   Note: Windowed spectral feature, NOT calibrated physical power.
9. spectral_centroid: Center of mass frequency (Hz) sum( f_k * |X(f_k)| ) / sum( |X(f_k)| ).

Important Methodological Notes & Caveats:
---------------------------------------
- Sampling Rate: f_s = 20,480 Hz is reported in NASA dataset documentation; not independently verified.
- FFT Normalization: FFT magnitudes are normalized by N (window sample count).
- DC Exclusion: The 0 Hz DC spectral bin is excluded from frequency-domain metrics.
- Mean-Centering: Signals are mean-centered before applying the Hann window and FFT.
- Zero Guard: Crest factor handles zero/constant RMS signals using numerical epsilon (1e-12).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Import data loading and preprocessing defaults
from src.data_loading import (
    DEFAULT_RAW_SET2_PATH,
    DEFAULT_CHANNEL_NAMES,
    load_snapshot,
)
from src.preprocessing import (
    DEFAULT_MANIFEST_PATH,
    extract_windows,
)

# EPSILON guard for division-by-zero protection
EPSILON = 1e-12

# Standard 9 feature suffixes per channel
FEATURE_NAMES_PER_CHANNEL = [
    "mean",
    "std",
    "rms",
    "p2p",
    "skewness",
    "kurtosis",
    "crest_factor",
    "spectral_energy",
    "spectral_centroid",
]


def extract_time_features(df_window: pd.DataFrame) -> dict[str, float]:
    """
    Extracts 7 time-domain features for each channel in a vibration signal window DataFrame.

    Parameters
    ----------
    df_window : pd.DataFrame
        Input DataFrame containing numerical vibration channels (N_samples x N_channels).

    Returns
    -------
    dict of str to float
        Dictionary containing 7 time-domain features per channel (28 keys for 4 channels).

    Raises
    ------
    TypeError
        If df_window is not a pandas DataFrame.
    ValueError
        If df_window contains NaN, infinite values, or is empty.
    """
    if not isinstance(df_window, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df_window)}.")

    if df_window.empty:
        raise ValueError("Input DataFrame is empty.")

    if df_window.isna().any().any():
        raise ValueError("Input DataFrame contains NaN values.")

    if np.isinf(df_window.to_numpy()).any():
        raise ValueError("Input DataFrame contains Infinite values.")

    time_features = {}

    for col in df_window.columns:
        # Standardize channel prefix (e.g. Channel_1 -> ch1)
        prefix = f"ch{col.split('_')[-1]}" if "_" in str(col) else f"ch_{col}"

        vals = df_window[col].to_numpy(dtype=np.float64)
        n_samples = len(vals)

        # 1. Mean
        mean_val = float(np.mean(vals))

        # 2. Sample standard deviation (ddof=1)
        std_val = float(np.std(vals, ddof=1)) if n_samples > 1 else 0.0

        # 3. Root Mean Square (RMS)
        rms_val = float(np.sqrt(np.mean(vals**2)))

        # 4. Peak-to-Peak (P2P)
        p2p_val = float(np.ptp(vals))

        # 5. Skewness (3rd standardized moment)
        skew_val = float(stats.skew(vals, bias=False)) if std_val > EPSILON else 0.0

        # 6. Fisher Excess Kurtosis (4th standardized moment - 3.0; Gaussian = 0.0)
        kurt_val = float(stats.kurtosis(vals, fisher=True, bias=False)) if std_val > EPSILON else 0.0

        # 7. Crest Factor (peak magnitude / (rms + epsilon))
        peak_val = float(np.max(np.abs(vals)))
        crest_factor_val = peak_val / (rms_val + EPSILON) if rms_val > EPSILON else 0.0

        time_features[f"{prefix}_mean"] = mean_val
        time_features[f"{prefix}_std"] = std_val
        time_features[f"{prefix}_rms"] = rms_val
        time_features[f"{prefix}_p2p"] = p2p_val
        time_features[f"{prefix}_skewness"] = skew_val
        time_features[f"{prefix}_kurtosis"] = kurt_val
        time_features[f"{prefix}_crest_factor"] = crest_factor_val

    return time_features


def extract_frequency_features(
    df_window: pd.DataFrame, sampling_rate: float = 20480.0
) -> dict[str, float]:
    """
    Extracts 2 frequency-domain features per channel using a Hann-windowed FFT.

    Parameters
    ----------
    df_window : pd.DataFrame
        Input DataFrame containing numerical vibration channels.
    sampling_rate : float, default 20480.0
        Reported sampling frequency in Hz (not independently verified).

    Returns
    -------
    dict of str to float
        Dictionary containing spectral_energy and spectral_centroid per channel.

    Raises
    ------
    TypeError
        If df_window is not a pandas DataFrame.
    ValueError
        If df_window is empty or sampling_rate is non-positive.
    """
    if not isinstance(df_window, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df_window)}.")

    if df_window.empty:
        raise ValueError("Input DataFrame is empty.")

    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}.")

    freq_features = {}
    n_samples = len(df_window)

    # Pre-generate Hann window
    hann_win = np.hanning(n_samples)

    # Frequency bin centers for positive frequencies
    freq_bins = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)

    for col in df_window.columns:
        prefix = f"ch{col.split('_')[-1]}" if "_" in str(col) else f"ch_{col}"
        vals = df_window[col].to_numpy(dtype=np.float64)

        # 1. Mean-centering
        vals_centered = vals - np.mean(vals)

        # 2. Hann windowing
        vals_windowed = vals_centered * hann_win

        # 3. Real FFT with 1/N scale normalization
        fft_complex = np.fft.rfft(vals_windowed) / n_samples
        magnitudes = np.abs(fft_complex)

        # 4. Exclude DC bin (index 0)
        pos_freqs = freq_bins[1:]
        pos_mags = magnitudes[1:]

        # 5. Spectral Energy (sum of squared magnitudes of positive frequencies)
        spectral_energy_val = float(np.sum(pos_mags**2))

        # 6. Spectral Centroid (weighted mean frequency in Hz)
        total_mag = np.sum(pos_mags)
        if total_mag > EPSILON:
            spectral_centroid_val = float(np.sum(pos_freqs * pos_mags) / total_mag)
        else:
            spectral_centroid_val = 0.0

        freq_features[f"{prefix}_spectral_energy"] = spectral_energy_val
        freq_features[f"{prefix}_spectral_centroid"] = spectral_centroid_val

    return freq_features


def extract_all_features(
    df_window: pd.DataFrame,
    sampling_rate: float = 20480.0
) -> dict[str, float]:
    """
    Combines time-domain and frequency-domain feature extraction for a single window DataFrame.

    Parameters
    ----------
    df_window : pd.DataFrame
        Input window DataFrame of shape (N_samples, 4).
    sampling_rate : float, default 20480.0
        Reported sampling rate in Hz.

    Returns
    -------
    dict of str to float
        Dictionary containing 36 feature values (9 features x 4 channels).
    """
    time_feats = extract_time_features(df_window)
    freq_feats = extract_frequency_features(df_window, sampling_rate=sampling_rate)
    
    # Merge dictionaries deterministically
    all_feats = {**time_feats, **freq_feats}
    return all_feats


def extract_dataset_features(
    manifest_csv: str | Path | None = None,
    raw_dir: str | Path | None = None,
    window_length: int = 20480,
    overlap_ratio: float = 0.0,
    sampling_rate: float = 20480.0
) -> pd.DataFrame:
    """
    Extracts features across all valid dataset snapshots defined in the manifest CSV.

    Parameters
    ----------
    manifest_csv : str or Path, optional
        Path to dataset manifest CSV. Defaults to DEFAULT_MANIFEST_PATH.
    raw_dir : str or Path, optional
        Path to raw dataset directory. Defaults to DEFAULT_RAW_SET2_PATH.
    window_length : int, default 20480
        Window length in samples. Default 20480 extracts 1 snapshot-level window per file.
    overlap_ratio : float, default 0.0
        Overlap fraction between contiguous windows.
    sampling_rate : float, default 20480.0
        Reported sampling rate in Hz.

    Returns
    -------
    pd.DataFrame
        DataFrame of extracted features.
        For snapshot-level baseline (window_length=20480): Shape (984, 40)
        (4 metadata columns + 36 feature columns).
    """
    manifest_path = Path(manifest_csv) if manifest_csv else DEFAULT_MANIFEST_PATH
    target_raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_SET2_PATH

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found at: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    dataset_rows = []

    # Process each valid snapshot chronologically
    for _, row in manifest_df.iterrows():
        file_idx = int(row["file_index"])
        fname = str(row["filename"])
        ts = str(row["timestamp"])
        is_valid = bool(row["is_valid"])

        fpath = target_raw_dir / fname

        if not is_valid or not fpath.exists():
            continue

        # Load snapshot
        df_snapshot = load_snapshot(fpath)

        # Slice snapshot into windows
        windows = extract_windows(
            df_snapshot,
            window_length=window_length,
            overlap_ratio=overlap_ratio,
            drop_incomplete=True
        )

        for win_idx, win_df in enumerate(windows):
            meta_dict = {
                "file_index": file_idx,
                "filename": fname,
                "timestamp": ts,
                "is_valid": is_valid,
            }

            if window_length < len(df_snapshot):
                meta_dict["window_index"] = win_idx

            feats_dict = extract_all_features(win_df, sampling_rate=sampling_rate)
            combined_row = {**meta_dict, **feats_dict}
            dataset_rows.append(combined_row)

    features_df = pd.DataFrame(dataset_rows)
    return features_df
