"""
verify_dataset.py - Read-only dataset verification utility for MachineMind AI.

Phase 2: Dataset Acquisition and Organization
Target Dataset: NASA IMS Bearing Dataset (Set 2)
"""

from pathlib import Path
import datetime
import pandas as pd


def verify_set2_dataset(raw_dir: Path) -> dict:
    """
    Verifies IMS Set 2 raw snapshot files without loading everything into memory.
    Operating Mode: Strictly Read-Only.
    """
    set2_path = raw_dir / "IMS" / "2nd_test" / "2nd_test"
    if not set2_path.exists():
        raise FileNotFoundError(f"Set 2 directory not found at: {set2_path}")

    files = sorted([f for f in set2_path.iterdir() if f.is_file()])
    if not files:
        raise ValueError(f"Set 2 directory exists but contains no files: {set2_path}")

    file_count = len(files)
    total_size = sum(f.stat().st_size for f in files)
    min_size = min(f.stat().st_size for f in files)
    max_size = max(f.stat().st_size for f in files)
    avg_size = total_size / file_count

    # Parse timestamps with explicit error handling
    timestamps = []
    for f in files:
        try:
            dt = datetime.datetime.strptime(f.name, "%Y.%m.%d.%H.%M.%S")
            timestamps.append(dt)
        except ValueError as err:
            raise ValueError(
                f"Invalid timestamp format in filename '{f.name}'. "
                f"Expected format 'YYYY.MM.DD.HH.MM.SS'."
            ) from err

    # Handle interval analysis cleanly (accounting for single-file edge cases)
    intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
    if intervals:
        is_strictly_increasing = all(i > 0 for i in intervals)
        all_intervals_600s = all(i == 600.0 for i in intervals)
        unique_intervals = sorted(list(set(intervals)))
    else:
        is_strictly_increasing = None  # Single file; no intervals to evaluate
        all_intervals_600s = None       # Single file; no intervals to evaluate
        unique_intervals = []

    # Sample check shape and nulls on first and last file
    sample_first = pd.read_csv(files[0], sep=r"\s+", header=None)
    sample_last = pd.read_csv(files[-1], sep=r"\s+", header=None)

    first_shape = sample_first.shape
    last_shape = sample_last.shape
    first_nulls = int(sample_first.isna().sum().sum())
    last_nulls = int(sample_last.isna().sum().sum())

    results = {
        "file_count": file_count,
        "total_size_bytes": total_size,
        "min_size_bytes": min_size,
        "max_size_bytes": max_size,
        "avg_size_bytes": avg_size,
        "first_timestamp": timestamps[0].strftime("%Y-%m-%d %H:%M:%S"),
        "last_timestamp": timestamps[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "intervals_count": len(intervals),
        "unique_intervals_seconds": unique_intervals,
        "strictly_increasing": is_strictly_increasing,
        "all_intervals_600s": all_intervals_600s,
        "sample_first_shape": first_shape,
        "sample_last_shape": last_shape,
        "sample_first_nulls": first_nulls,
        "sample_last_nulls": last_nulls
    }

    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_raw_dir = project_root / "data" / "raw"
    print("Running read-only verification for IMS Set 2...")
    res = verify_set2_dataset(data_raw_dir)
    print("--- Verification Results ---")
    for k, v in res.items():
        print(f"{k}: {v}")
