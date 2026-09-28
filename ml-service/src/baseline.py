"""
baseline.py - Statistical Baseline Module for MachineMind AI.

Phase 6: Statistical Baseline
Target Dataset: NASA IMS Bearing Dataset (Set 2)

This module implements a transparent, non-machine-learning statistical baseline
to quantify feature deviations and anomaly scores across dataset snapshots.

Chronological Dataset Split Specification:
------------------------------------------
1. Baseline Fitting Period (Snapshots 0 to 159, N=160 snapshots, ~26.7 hours):
   Used strictly to fit baseline distribution parameters (mean, std, median, MAD, IQR, P2P).
   Assumed to represent normal, un-degraded operation (provisional reference period).
2. Reference Evaluation Period (Snapshots 160 to 199, N=40 snapshots, ~6.7 hours):
   Evaluated using frozen baseline parameters to verify baseline stability on unseen provisional normal data.
3. Sequential Evaluation Period (Snapshots 200 to 983, N=784 snapshots, ~130.7 hours):
   Evaluated sequentially using frozen baseline parameters to monitor degradation trajectory online.

Numerical Safeguards & Scale-Aware Variance Guarding:
---------------------------------------------------
- Absolute minimum cutoff: EPS_ABS = 1e-6.
- Scale-relative cutoff factor: ETA_REL = 1e-4.
- Standard Deviation Cutoff: max(EPS_ABS, ETA_REL * P2P_fit).
- Guarded Standard Deviation: max(std_fit, Cutoff_std).
- MAD Cutoff: max(EPS_ABS, ETA_REL * IQR_fit).
- Guarded MAD: max(MAD_fit, Cutoff_MAD).
- Completely constant features (P2P_fit == 0) are excluded from scoring.
- Composite score RMS denominator uses exact count of eligible features (K_eligible).
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Numerical guard constants
EPS_ABS = 1e-6
ETA_REL = 1e-4
DEFAULT_REF_END_IDX = 160
FACTOR_MOD_Z = 0.6745

# Known metadata column names in processed feature CSVs
METADATA_COLUMNS = [
    "file_index",
    "filename",
    "timestamp",
    "is_valid",
    "window_index",
    "window_start_sample",
    "window_end_sample",
]


def _get_feature_and_metadata_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Helper to separate numerical feature columns from metadata columns."""
    meta_cols = [c for c in df.columns if c in METADATA_COLUMNS]
    feat_cols = [
        c for c in df.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    return feat_cols, meta_cols


def fit_baseline_parameters(
    df_features: pd.DataFrame,
    ref_end_idx: int = DEFAULT_REF_END_IDX
) -> dict:
    """
    Fits statistical baseline parameters using strictly the first ref_end_idx snapshots.

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame containing extracted features and metadata columns.
    ref_end_idx : int, default 160
        Number of initial snapshots to use for fitting baseline parameters.

    Returns
    -------
    dict
        Dictionary containing baseline parameters:
        - 'ref_end_idx': int
        - 'feature_names': list of str (all 36 features)
        - 'stats': dict mapping feature_name -> dict of fit parameters:
            ('mean', 'std', 'median', 'mad', 'iqr', 'p2p',
             'cutoff_std', 'guarded_std', 'cutoff_mad', 'guarded_mad',
             'is_eligible_z', 'is_eligible_m')
        - 'eligible_z_features': list of str (features eligible for Z-score)
        - 'eligible_m_features': list of str (features eligible for Modified Z-score)
        - 'excluded_z_features': list of str (features excluded from Z-score)
        - 'excluded_m_features': list of str (features excluded from Modified Z-score)

    Raises
    ------
    TypeError
        If df_features is not a pandas DataFrame.
    ValueError
        If df_features is empty, contains NaN/Inf values, or has fewer rows than ref_end_idx.
    """
    if not isinstance(df_features, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df_features)}.")

    if df_features.empty:
        raise ValueError("Input feature DataFrame is empty.")

    total_rows = len(df_features)
    if ref_end_idx <= 0:
        raise ValueError(f"ref_end_idx must be positive, got {ref_end_idx}.")

    if ref_end_idx > total_rows:
        raise ValueError(
            f"ref_end_idx ({ref_end_idx}) exceeds total DataFrame rows ({total_rows})."
        )

    feat_cols, _ = _get_feature_and_metadata_cols(df_features)
    if not feat_cols:
        raise ValueError("No numerical feature columns found in input DataFrame.")

    # Slice strictly the reference fitting period (snapshots 0 to ref_end_idx - 1)
    df_ref = df_features.iloc[:ref_end_idx][feat_cols]

    # Check for NaN / Infinite values in reference slice
    if df_ref.isna().any().any():
        raise ValueError("Reference fitting slice contains NaN values.")

    if np.isinf(df_ref.to_numpy()).any():
        raise ValueError("Reference fitting slice contains Infinite values.")

    stats_dict = {}
    eligible_z = []
    eligible_m = []
    excluded_z = []
    excluded_m = []

    for col in feat_cols:
        vals = df_ref[col].to_numpy(dtype=np.float64)
        n_samples = len(vals)

        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if n_samples > 1 else 0.0
        median_val = float(np.median(vals))

        # Median Absolute Deviation (MAD = median(|x - median|))
        mad_val = float(np.median(np.abs(vals - median_val)))

        # Interquartile Range (IQR = Q3 - Q1)
        q75, q25 = np.percentile(vals, [75, 25])
        iqr_val = float(q75 - q25)

        # Peak-to-Peak Range (P2P = max - min)
        p2p_val = float(np.ptp(vals))

        # Scale-aware cutoffs
        cutoff_std = max(EPS_ABS, ETA_REL * p2p_val)
        cutoff_mad = max(EPS_ABS, ETA_REL * iqr_val)

        guarded_std = max(std_val, cutoff_std)
        guarded_mad = max(mad_val, cutoff_mad)

        # Feature eligibility policy:
        # If P2P is completely zero (constant signal across all fitting files), exclude from scoring.
        is_eligible_z = (p2p_val > 0.0)
        is_eligible_m = (p2p_val > 0.0)

        if is_eligible_z:
            eligible_z.append(col)
        else:
            excluded_z.append(col)

        if is_eligible_m:
            eligible_m.append(col)
        else:
            excluded_m.append(col)

        stats_dict[col] = {
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "mad": mad_val,
            "iqr": iqr_val,
            "p2p": p2p_val,
            "cutoff_std": cutoff_std,
            "guarded_std": guarded_std,
            "cutoff_mad": cutoff_mad,
            "guarded_mad": guarded_mad,
            "is_eligible_z": is_eligible_z,
            "is_eligible_m": is_eligible_m,
        }

    return {
        "ref_end_idx": ref_end_idx,
        "feature_names": feat_cols,
        "stats": stats_dict,
        "eligible_z_features": eligible_z,
        "eligible_m_features": eligible_m,
        "excluded_z_features": excluded_z,
        "excluded_m_features": excluded_m,
    }


def compute_zscore_features(
    df_features: pd.DataFrame,
    baseline_params: dict
) -> pd.DataFrame:
    """
    Computes standard Z-scores for all feature columns using frozen baseline parameters.

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame containing features to evaluate.
    baseline_params : dict
        Baseline parameters returned by fit_baseline_parameters().

    Returns
    -------
    pd.DataFrame
        DataFrame preserving metadata columns and containing Z-score feature values.
    """
    if not isinstance(df_features, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df_features)}.")

    if not isinstance(baseline_params, dict) or "stats" not in baseline_params:
        raise TypeError("Invalid baseline_params dictionary.")

    feat_cols, meta_cols = _get_feature_and_metadata_cols(df_features)
    stats_map = baseline_params["stats"]

    df_out = df_features[meta_cols].copy() if meta_cols else pd.DataFrame(index=df_features.index)

    for col in feat_cols:
        if col in stats_map:
            mu = stats_map[col]["mean"]
            sigma_g = stats_map[col]["guarded_std"]
            vals = df_features[col].to_numpy(dtype=np.float64)
            df_out[col] = (vals - mu) / sigma_g

    return df_out


def compute_modified_zscore_features(
    df_features: pd.DataFrame,
    baseline_params: dict
) -> pd.DataFrame:
    """
    Computes robust modified Z-scores for all feature columns using frozen baseline parameters.

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame containing features to evaluate.
    baseline_params : dict
        Baseline parameters returned by fit_baseline_parameters().

    Returns
    -------
    pd.DataFrame
        DataFrame preserving metadata columns and containing modified Z-score feature values.
    """
    if not isinstance(df_features, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df_features)}.")

    if not isinstance(baseline_params, dict) or "stats" not in baseline_params:
        raise TypeError("Invalid baseline_params dictionary.")

    feat_cols, meta_cols = _get_feature_and_metadata_cols(df_features)
    stats_map = baseline_params["stats"]

    df_out = df_features[meta_cols].copy() if meta_cols else pd.DataFrame(index=df_features.index)

    for col in feat_cols:
        if col in stats_map:
            med = stats_map[col]["median"]
            mad_g = stats_map[col]["guarded_mad"]
            vals = df_features[col].to_numpy(dtype=np.float64)
            df_out[col] = FACTOR_MOD_Z * (vals - med) / mad_g

    return df_out


def compute_composite_anomaly_scores(
    df_features: pd.DataFrame,
    baseline_params: dict
) -> pd.DataFrame:
    """
    Computes four composite statistical anomaly scores for each snapshot.

    Calculated Composite Scores:
    1. score_max_z: Maximum absolute Z-score across eligible Z features.
    2. score_rms_z: RMS composite Z-score = sqrt( sum(Z^2) / K_eligible_z ).
    3. score_max_m: Maximum absolute modified Z-score across eligible M features.
    4. score_rms_m: RMS composite modified Z-score = sqrt( sum(M^2) / K_eligible_m ).

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame containing features to evaluate.
    baseline_params : dict
        Baseline parameters returned by fit_baseline_parameters().

    Returns
    -------
    pd.DataFrame
        DataFrame preserving metadata columns with 4 composite score columns:
        ['score_max_z', 'score_rms_z', 'score_max_m', 'score_rms_m'].

    Raises
    ------
    ValueError
        If all features are excluded from scoring (K_eligible == 0).
    """
    if not isinstance(df_features, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df_features)}.")

    if not isinstance(baseline_params, dict) or "stats" not in baseline_params:
        raise TypeError("Invalid baseline_params dictionary.")

    eligible_z = baseline_params.get("eligible_z_features", [])
    eligible_m = baseline_params.get("eligible_m_features", [])

    k_z = len(eligible_z)
    k_m = len(eligible_m)

    if k_z == 0 and k_m == 0:
        raise ValueError(
            "All features are excluded from scoring due to zero variance during fitting."
        )

    # Compute feature Z-scores and modified Z-scores
    df_z = compute_zscore_features(df_features, baseline_params)
    df_m = compute_modified_zscore_features(df_features, baseline_params)

    _, meta_cols = _get_feature_and_metadata_cols(df_features)
    df_out = df_features[meta_cols].copy() if meta_cols else pd.DataFrame(index=df_features.index)

    # 1 & 2. Z-score composite metrics
    if k_z > 0:
        z_matrix = df_z[eligible_z].to_numpy(dtype=np.float64)
        df_out["score_max_z"] = np.max(np.abs(z_matrix), axis=1)
        df_out["score_rms_z"] = np.sqrt(np.mean(z_matrix ** 2, axis=1))
    else:
        raise ValueError("All features excluded from Z-score composite calculation.")

    # 3 & 4. Modified Z-score composite metrics
    if k_m > 0:
        m_matrix = df_m[eligible_m].to_numpy(dtype=np.float64)
        df_out["score_max_m"] = np.max(np.abs(m_matrix), axis=1)
        df_out["score_rms_m"] = np.sqrt(np.mean(m_matrix ** 2, axis=1))
    else:
        raise ValueError("All features excluded from Modified Z-score composite calculation.")

    return df_out
