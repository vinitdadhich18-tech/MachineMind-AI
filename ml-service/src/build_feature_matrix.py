"""
MachineMind AI — Feature Matrix Generator & Target Engineering Module
Phase 5: Supervised Data Pipeline & Feature Matrix Construction

Processes all 984 raw NASA IMS Set 2 snapshot files using the Phase 4 CanonicalFeaturePipeline
and saves the resulting feature matrix with snapshot metadata and RUL targets.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.canonical_feature_pipeline import CanonicalFeaturePipeline

DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "IMS" / "2nd_test" / "2nd_test"
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DEFAULT_MATRIX_PATH = DEFAULT_PROCESSED_DIR / "ims_set2_feature_matrix.csv"


def build_feature_matrix(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Optional[Path] = DEFAULT_MATRIX_PATH,
    sampling_rate_hz: float = 20480.0
) -> pd.DataFrame:
    """
    Builds the 984-row feature matrix from raw IMS snapshot files using the canonical feature pipeline.

    Args:
        raw_dir: Directory containing the 984 raw snapshot files.
        output_path: Optional path to save the generated feature matrix CSV.
        sampling_rate_hz: Sampling frequency in Hz.

    Returns:
        DataFrame containing snapshot metadata, 36 canonical features, and RUL targets.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dir}")

    files = sorted([f for f in os.listdir(raw_dir) if not f.startswith(".")])
    if len(files) == 0:
        raise ValueError(f"No snapshot files found in {raw_dir}")

    pipeline = CanonicalFeaturePipeline(sampling_rate_hz=sampling_rate_hz)
    records: List[Dict] = []

    for idx, filename in enumerate(files):
        filepath = raw_dir / filename
        df_snap = pd.read_csv(filepath, sep="\t", header=None, names=["ch1", "ch2", "ch3", "ch4"])
        vector = pipeline.process_snapshot_dataframe(
            df_snap,
            original_timestamp=filename,
            snapshot_sequence=idx
        )
        row = {
            "snapshot_sequence": idx,
            "original_timestamp": filename,
        }
        row.update(vector.feature_dict)
        records.append(row)

    df_matrix = pd.DataFrame(records)

    # Attach derived RUL targets
    total_snapshots = len(df_matrix)
    df_matrix["linear_rul"] = (total_snapshots - 1) - df_matrix["snapshot_sequence"]
    
    # Capped RUL targets (e.g. cap at 500, 400, 300 snapshots)
    df_matrix["capped_rul_500"] = np.minimum(df_matrix["linear_rul"], 500)
    df_matrix["capped_rul_400"] = np.minimum(df_matrix["linear_rul"], 400)
    df_matrix["capped_rul_300"] = np.minimum(df_matrix["linear_rul"], 300)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_matrix.to_csv(output_path, index=False)

    return df_matrix


def load_or_build_feature_matrix(
    matrix_path: Path = DEFAULT_MATRIX_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR
) -> pd.DataFrame:
    """Loads feature matrix from disk if it exists; otherwise generates it."""
    matrix_path = Path(matrix_path)
    if matrix_path.exists():
        df = pd.read_csv(matrix_path)
        if len(df) == 984 and "snapshot_sequence" in df.columns:
            return df
    return build_feature_matrix(raw_dir=raw_dir, output_path=matrix_path)


if __name__ == "__main__":
    print("Building canonical feature matrix for NASA IMS Set 2 dataset...")
    df = build_feature_matrix()
    print(f"Successfully generated feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Features: {[c for c in df.columns if c.startswith('ch')]}")
