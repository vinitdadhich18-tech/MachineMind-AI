"""
preprocessing.py - Reproducible Signal Preprocessing Foundation for MachineMind AI.

Phase 4: Signal Preprocessing
Target Dataset: NASA IMS Bearing Dataset (Set 2)

This module provides reproducible, beginner-friendly functions to:
1. Generate a deterministic dataset manifest (`manifest_set2.csv`) tracking file integrity and signal observations.
2. Optionally perform snapshot mean-centering (DC-offset removal) while preserving raw input data.
3. Process single snapshots cleanly without applying harmful per-file amplitude normalization.
4. Slice snapshots into contiguous windows with configurable window lengths and overlap ratios.
"""

from pathlib import Path
import datetime
import pandas as pd
import numpy as np

# Import existing data loading utilities
from src.data_loading import (
    DEFAULT_RAW_SET2_PATH,
    DEFAULT_CHANNEL_NAMES,
    get_snapshot_files,
    parse_snapshot_timestamp,
    load_snapshot,
)

# Default path for the dataset manifest CSV
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "manifest_set2.csv"


def create_dataset_manifest(
    raw_dir: Path | None = None,
    output_csv_path: Path | None = None,
    low_std_threshold: float = 0.005
) -> pd.DataFrame:
    """
    Scans all dataset snapshot files, verifies metadata, and generates a deterministic CSV manifest.

    Parameters
    ----------
    raw_dir : Path, optional
        Path to raw snapshot directory. Defaults to DEFAULT_RAW_SET2_PATH.
    output_csv_path : Path, optional
        Path to save output manifest CSV. Defaults to DEFAULT_MANIFEST_PATH.
    low_std_threshold : float, default 0.005
        Threshold for flagging low-amplitude snapshots in notes separately from structural validity failures.

    Returns
    -------
    pd.DataFrame
        Manifest DataFrame containing columns:
        ['file_index', 'filename', 'timestamp', 'file_size_bytes', 'rows', 'cols', 'is_valid', 'notes'].

    Raises
    ------
    FileNotFoundError
        If raw_dir does not exist.
    ValueError
        If raw_dir contains no files.
    """
    target_raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_SET2_PATH
    target_out_path = Path(output_csv_path) if output_csv_path else DEFAULT_MANIFEST_PATH

    files = get_snapshot_files(target_raw_dir)
    manifest_rows = []

    for i, f in enumerate(files):
        ts = parse_snapshot_timestamp(f)
        size_bytes = f.stat().st_size
        is_valid = True
        notes = "Structurally valid snapshot."

        try:
            df = load_snapshot(f)
            rows, cols = df.shape
            
            # Check for low-amplitude observation (flagged separately from structural validity failures)
            channel_stds = df.std()
            if (channel_stds < low_std_threshold).all():
                notes = f"Structurally valid snapshot; Note: Low-amplitude signal observed (all channel std < {low_std_threshold})."
        except Exception as err:
            is_valid = False
            rows, cols = 0, 0
            notes = f"Structural validation failure: {str(err)}"

        manifest_rows.append({
            "file_index": i,
            "filename": f.name,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "file_size_bytes": size_bytes,
            "rows": rows,
            "cols": cols,
            "is_valid": is_valid,
            "notes": notes
        })

    manifest_df = pd.DataFrame(manifest_rows)

    # Save manifest CSV deterministically
    target_out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(target_out_path, index=False)
    print(f"Manifest successfully generated and saved to: {target_out_path}")

    return manifest_df


def remove_dc_offset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optionally centers a snapshot signal by subtracting the mean independently for each channel.

    Educational Concept — What Mean Centering Does & Does Not Do:
    - What it does: Subtracts per-channel arithmetic mean (df - df.mean()), shifting the baseline to 0.0.
    - What it does NOT do: It does NOT alter standard deviation, variance, or peak-to-peak amplitude.
      It preserves signal dynamic range without compressing or normalizing amplitude trends.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing numerical vibration channels.

    Returns
    -------
    pd.DataFrame
        New centered DataFrame with zero mean per channel. Original input DataFrame is unmodified.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If df contains missing (NaN) or infinite values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)}.")

    if df.isna().any().any():
        raise ValueError("Input DataFrame contains NaN values.")

    if np.isinf(df.to_numpy()).any():
        raise ValueError("Input DataFrame contains Infinite values.")

    # Return a new centered DataFrame (original df remains untouched)
    df_centered = df - df.mean()

    # Validate output shape
    if df_centered.shape != df.shape:
        raise ValueError(f"Shape mismatch after mean centering: Original {df.shape}, Centered {df_centered.shape}.")

    return df_centered


def process_snapshot(
    file_path: str | Path,
    remove_dc: bool = False
) -> pd.DataFrame:
    """
    Loads and optionally applies preprocessing (mean centering) to a single snapshot file.

    Parameters
    ----------
    file_path : str or Path
        Path to the snapshot file.
    remove_dc : bool, default False
        If True, applies remove_dc_offset() to center the signal.
        If False, returns raw snapshot DataFrame unchanged.

    Returns
    -------
    pd.DataFrame
        Processed snapshot DataFrame of shape (20480, 4).
    """
    df_raw = load_snapshot(file_path)
    if remove_dc:
        return remove_dc_offset(df_raw)
    return df_raw


def extract_windows(
    df: pd.DataFrame,
    window_length: int = 2048,
    overlap_ratio: float = 0.0,
    drop_incomplete: bool = True
) -> list[pd.DataFrame]:
    """
    Slices a snapshot DataFrame into smaller contiguous windows while preserving all four channels.

    Educational Concept — Windowing Trade-offs:
    - window_length: Controls window length in sample rows (e.g., 2048 samples = ~100 ms at 20.48 kHz).
    - overlap_ratio: Controls overlap fraction (0.0 = non-overlapping, 0.5 = 50% overlap).
    - drop_incomplete: Controls handling of leftover samples at the end of a snapshot.

    Parameters
    ----------
    df : pd.DataFrame
        Input snapshot DataFrame of shape (N_samples, N_channels).
    window_length : int, default 2048
        Number of sample rows per window.
    overlap_ratio : float, default 0.0
        Fraction of window overlap. Must be in range [0.0, 1.0).
    drop_incomplete : bool, default True
        If True, discards leftover samples fewer than window_length.
        If False, includes the final incomplete window.

    Returns
    -------
    list of pd.DataFrame
        List of window DataFrames. Each full window has shape (window_length, N_channels)
        and retains original column names and data types. Original input DataFrame is unmodified.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If window_length <= 0, overlap_ratio is not in [0.0, 1.0), or window_length > len(df).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)}.")

    total_samples = len(df)
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}.")

    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError(f"overlap_ratio must be in range [0.0, 1.0), got {overlap_ratio}.")

    if window_length > total_samples:
        raise ValueError(
            f"window_length ({window_length}) exceeds total DataFrame samples ({total_samples})."
        )

    # Calculate step size based on overlap ratio
    step_size = int(np.floor(window_length * (1.0 - overlap_ratio)))
    if step_size < 1:
        step_size = 1

    windows = []
    start_idx = 0

    while start_idx < total_samples:
        end_idx = start_idx + window_length

        if end_idx <= total_samples:
            # Full window slice (make copy to ensure non-mutating behavior)
            win_df = df.iloc[start_idx:end_idx].copy()
            windows.append(win_df)
        else:
            # Handling incomplete final window
            if not drop_incomplete and start_idx < total_samples:
                win_df = df.iloc[start_idx:total_samples].copy()
                windows.append(win_df)
            break

        start_idx += step_size

    return windows
