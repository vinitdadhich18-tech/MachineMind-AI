"""
csv_validation.py - Client-side pre-validation for uploaded snapshot CSV files.
"""

import io
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


def validate_snapshot_csv(file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str], Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Client-side CSV pre-validation helper.

    Checks:
    - Allowed extension (.csv, .txt, .tsv, or NASA IMS timestamp format)
    - Non-empty file
    - Delimiter auto-detection (comma, tab, whitespace)
    - Exactly 20480 rows x 4 columns
    - All numeric and finite values (no NaN / Inf)

    Returns:
    (is_valid: bool, error_message: str or None, details: dict, preview_df: pd.DataFrame or None)
    """
    if not file_bytes or len(file_bytes) == 0:
        return False, "Uploaded file is empty.", {"bytes": 0}, None

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    is_nasa_format = bool(filename.replace(".", "").isdigit())

    if ext not in [".csv", ".txt", ".tsv", ""] and not is_nasa_format:
        return (
            False,
            f"Unsupported file extension '{ext}'. Allowed extensions: .csv, .txt, .tsv",
            {"extension": ext},
            None
        )

    content_str = file_bytes.decode("utf-8", errors="replace")

    df = None
    for sep in [r"\s+", r",", r"\t"]:
        try:
            temp_df = pd.read_csv(
                io.StringIO(content_str),
                sep=sep,
                header=None,
                engine="python"
            )
            if temp_df.shape[1] == 4:
                first_row = temp_df.iloc[0].values
                if any(isinstance(val, str) and not val.replace(".", "", 1).replace("-", "", 1).isdigit() for val in first_row):
                    temp_df = temp_df.iloc[1:].reset_index(drop=True)

            if temp_df.shape[1] == 4:
                df = temp_df
                break
        except Exception:
            continue

    if df is None:
        return (
            False,
            "Could not parse file into 4 vibration data columns. Please ensure 4 delimited numeric columns.",
            {},
            None
        )

    # Convert columns to standard names
    df.columns = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]

    try:
        arr = df.to_numpy(dtype=np.float64)
    except Exception as e:
        return (
            False,
            "Snapshot contains non-numeric text or unparseable characters.",
            {"error": str(e)},
            None
        )

    rows, cols = arr.shape
    if rows != 20480 or cols != 4:
        return (
            False,
            f"Invalid snapshot shape: Expected exactly 20480 rows x 4 columns; got {rows} rows x {cols} columns.",
            {"expected_rows": 20480, "expected_cols": 4, "actual_rows": rows, "actual_cols": cols},
            None
        )

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    if nan_count > 0 or inf_count > 0:
        return (
            False,
            f"Snapshot contains non-finite values (NaN count: {nan_count}, Inf count: {inf_count}).",
            {"nan_count": nan_count, "inf_count": inf_count},
            None
        )

    summary_details = {
        "rows": rows,
        "cols": cols,
        "channel_stats": []
    }
    for idx, col in enumerate(df.columns, start=1):
        summary_details["channel_stats"].append({
            "channel": idx,
            "min": float(np.min(arr[:, idx - 1])),
            "max": float(np.max(arr[:, idx - 1])),
            "mean": float(np.mean(arr[:, idx - 1]))
        })

    return True, None, summary_details, df
