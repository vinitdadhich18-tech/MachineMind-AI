"""
data_loading.py - Reusable Data Loading Utility for MachineMind AI.

Phase 3: Exploratory Data Analysis (EDA)
Target Dataset: NASA IMS Bearing Dataset (Set 2)

This module provides modular, beginner-friendly helper functions to:
1. Parse timestamps from raw snapshot filenames.
2. List raw snapshot files sorted explicitly by parsed timestamps.
3. Load individual snapshot files into pandas DataFrames with strict shape and data quality validation.
4. Compute time-domain summary statistics (Mean, Std, Min, Max, RMS, Skewness, Kurtosis).
"""

from pathlib import Path
import datetime
import pandas as pd
import numpy as np


# Default relative path to IMS Set 2 raw snapshot directory
DEFAULT_RAW_SET2_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "IMS" / "2nd_test" / "2nd_test"

# Default column naming for the 4 vibration channels
DEFAULT_CHANNEL_NAMES = ["Channel_1", "Channel_2", "Channel_3", "Channel_4"]


def parse_snapshot_timestamp(file_input: str | Path) -> datetime.datetime:
    """
    Parses a snapshot filename into a Python datetime object.

    Expected filename format: YYYY.MM.DD.HH.MM.SS (e.g., '2004.02.12.10.32.39')

    Parameters
    ----------
    file_input : str or Path
        The filename or full path to the snapshot file.

    Returns
    -------
    datetime.datetime
        Parsed timestamp.

    Raises
    ------
    ValueError
        If the filename format does not match 'YYYY.MM.DD.HH.MM.SS'.
    """
    path_obj = Path(file_input)
    filename = path_obj.name
    try:
        return datetime.datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")
    except ValueError as err:
        raise ValueError(
            f"Invalid snapshot filename format: '{filename}'. "
            f"Expected format: 'YYYY.MM.DD.HH.MM.SS' (e.g., '2004.02.12.10.32.39')."
        ) from err


def get_snapshot_files(raw_dir: Path | None = None) -> list[Path]:
    """
    Discovers and returns all raw snapshot files sorted explicitly by parsed timestamps.

    Parameters
    ----------
    raw_dir : Path, optional
        Path to the directory containing Set 2 raw files.
        If None, uses DEFAULT_RAW_SET2_PATH.

    Returns
    -------
    list of Path
        Chronologically sorted list of snapshot file paths.

    Raises
    ------
    FileNotFoundError
        If the raw directory does not exist.
    ValueError
        If the raw directory exists but contains no snapshot files or contains invalid filenames.
    """
    target_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_SET2_PATH

    if not target_dir.exists():
        raise FileNotFoundError(f"Raw snapshot directory not found at: {target_dir}")

    raw_files = [f for f in target_dir.iterdir() if f.is_file()]
    if not raw_files:
        raise ValueError(f"Snapshot directory exists but contains no files: {target_dir}")

    # Explicitly sort snapshot files using parsed datetime objects
    try:
        files = sorted(raw_files, key=parse_snapshot_timestamp)
    except ValueError as err:
        raise ValueError(f"Failed to sort files in '{target_dir}' due to invalid filename timestamps.") from err

    return files


def load_snapshot(
    file_path: str | Path,
    channel_names: list[str] | None = None,
    expected_rows: int = 20480,
    expected_cols: int = 4
) -> pd.DataFrame:
    """
    Loads a single 1-second vibration snapshot ASCII text file into a pandas DataFrame
    with strict matrix dimensions, channel name length, and missing/infinite value validation.

    Parameters
    ----------
    file_path : str or Path
        Path to the raw snapshot file.
    channel_names : list of str, optional
        Column names to assign to the channels. Defaults to ['Channel_1', 'Channel_2', 'Channel_3', 'Channel_4'].
    expected_rows : int, default 20480
        Expected number of sample rows per snapshot (1 second @ 20.48 kHz).
    expected_cols : int, default 4
        Expected number of vibration channels.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (expected_rows, expected_cols) containing raw float vibration data.

    Raises
    ------
    FileNotFoundError
        If file_path does not exist.
    ValueError
        If channel_names length does not equal expected_cols, if loaded matrix shape
        differs from (expected_rows, expected_cols), or if NaN/infinite values exist.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path_obj}")

    cols = channel_names if channel_names else DEFAULT_CHANNEL_NAMES
    if len(cols) != expected_cols:
        raise ValueError(
            f"Invalid channel_names length: Provided {len(cols)} names ({cols}), "
            f"Expected exactly {expected_cols} names."
        )

    # Load whitespace-delimited ASCII data
    df = pd.read_csv(path_obj, sep=r"\s+", header=None, names=cols)

    # Validate data matrix dimensions
    if df.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"Unexpected shape in snapshot '{path_obj.name}': "
            f"Observed {df.shape}, Expected ({expected_rows}, {expected_cols})."
        )

    # Validate data quality (NaN / missing or Infinite values)
    if df.isna().any().any():
        raise ValueError(f"Snapshot file '{path_obj.name}' contains NaN (missing) values.")

    if np.isinf(df.to_numpy()).any():
        raise ValueError(f"Snapshot file '{path_obj.name}' contains Infinite values.")

    return df


def compute_snapshot_stats(
    df_or_path: pd.DataFrame | str | Path,
    fisher: bool = True
) -> pd.DataFrame:
    """
    Computes time-domain summary statistics for a single snapshot file or DataFrame.

    Metrics calculated per channel:
    - mean: Arithmetic mean of the signal, indicating DC offset or static sensor bias.
    - std: Standard deviation, measuring AC signal dispersion/energy spread around the mean.
    - min: Minimum signal amplitude.
    - max: Maximum signal amplitude.
    - rms: Root Mean Square, quadratic mean sqrt(mean(x^2)) representing total dynamic vibration power.
    - skewness: Third standardized moment (m3 / std^3), quantifying probability density function
      asymmetry around the mean (0.0 for a symmetric Gaussian vibration signal; non-zero indicates
      asymmetric impact profiles or single-sided sensor bias).
    - kurtosis: Fourth standardized moment (m4 / std^4), quantifying distribution spikiness,
      heavy-tailedness, and impulsive mechanical shock peaks.

    Statistical Convention Note on Kurtosis:
    - By default (fisher=True), Fisher's excess kurtosis is computed (Kurtosis_pearson - 3.0),
      where a pure Gaussian vibration signal has an excess kurtosis of 0.0. Values > 0.0 indicate
      spiky/impulsive signal tails caused by bearing impacts.
    - If fisher=False, Pearson's kurtosis is computed, where a normal Gaussian signal equals 3.0.

    Parameters
    ----------
    df_or_path : pd.DataFrame, str, or Path
        Either a loaded snapshot DataFrame or path to a snapshot file.
    fisher : bool, default True
        If True, returns Fisher's excess kurtosis (Normal Gaussian = 0.0).
        If False, returns Pearson's kurtosis (Normal Gaussian = 3.0).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by channel names with columns:
        ['mean', 'std', 'min', 'max', 'rms', 'skewness', 'kurtosis'].
    """
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path
    else:
        df = load_snapshot(df_or_path)

    stats = pd.DataFrame(index=df.columns)

    stats["mean"] = df.mean()
    stats["std"] = df.std()
    stats["min"] = df.min()
    stats["max"] = df.max()

    # Root Mean Square (RMS) = sqrt( mean( x^2 ) )
    stats["rms"] = np.sqrt((df ** 2).mean())

    # Distribution shape metrics
    stats["skewness"] = df.skew()

    # pandas kurtosis() defaults to Fisher excess kurtosis (Normal Gaussian = 0.0)
    fisher_kurt = df.kurtosis()
    stats["kurtosis"] = fisher_kurt if fisher else (fisher_kurt + 3.0)

    return stats
