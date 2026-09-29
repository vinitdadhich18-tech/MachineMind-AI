"""
evaluation.py - Evaluation Module for MachineMind AI.

Phase 8: Evaluation
Target Dataset: NASA IMS Bearing Dataset (Set 2)

This module implements modular, reproducible, and leakage-safe functions to evaluate
unsupervised vibration anomaly detection models (Statistical Baseline, Isolation Forest, and PCA).

Key Methodological Specifications (from reports/evaluation_protocol.md):
-----------------------------------------------------------------------
1. Chronological Partitioning:
   - Fit Period: Snapshots 0..159 (N=160, ~26.7 hrs). Assumed healthy training period.
   - Validation Period: Snapshots 160..199 (N=40, ~6.7 hrs). Threshold selection period.
   - Evaluation Period: Snapshots 200..983 (N=784, ~130.7 hrs). Sequential testing period.

2. Persistence Filtering & Timing Definitions:
   - First Exceedance Index (t_exceed): First snapshot index t >= 200 with S_t > T (k=1).
   - Start Index of Sustained Alert (t_start): Index t where a continuous run of >= k
     consecutive snapshots (S_tau > T for tau in [t, t+k-1]) begins.
   - Alert Confirmation Index (t_confirm): Exact index where a causal online monitoring system
     confirms k consecutive exceedances: t_confirm = t_start + k - 1.
   - Alert Confirmation Timestamp: Timestamp corresponding to t_confirm.

3. Metric Definitions & Event Counting:
   - Sustained Event Episode: Contiguous run of snapshots exceeding threshold for length L >= k.
   - Raw False Alarm Count (N_FA_raw): Snapshots in 0..199 with S_t > T (k=1).
   - Sustained False Alarm Count (N_FA_sustained): Snapshots in 0..199 belonging to sustained runs (L >= k).
   - False Alarm Rate (R_FA): Fraction of assumed healthy snapshots (N=200) flagged.

4. Multi-Channel Aggregation:
   - Logical OR Shaft Alert: Active at snapshot t iff at least one channel satisfies its sustained alert.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd

# Default chronological partition constants
DEFAULT_FIT_END_IDX = 160   # Snapshots 0..159
DEFAULT_VAL_END_IDX = 200   # Snapshots 160..199
DEFAULT_EVAL_END_IDX = 984  # Snapshots 200..983 (Total N=984)


def validate_scores_array(scores: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
    """
    Validates and converts input scores to a 1D float64 numpy array.

    Raises
    ------
    ValueError
        If array is empty, multi-dimensional, contains NaNs or infinite values.
    TypeError
        If input cannot be converted to a numeric numpy array.
    """
    if scores is None:
        raise ValueError("Scores input cannot be None.")
        
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array of scores, got shape {arr.shape}.")
    if len(arr) == 0:
        raise ValueError("Scores array is empty.")
    if np.isnan(arr).any():
        raise ValueError("Scores array contains NaN values.")
    if np.isinf(arr).any():
        raise ValueError("Scores array contains Infinite values.")
        
    return arr


def calculate_validation_thresholds(
    val_scores: Union[np.ndarray, pd.Series, List[float]],
    sigmas: List[float] = [3.0, 4.0, 6.0],
    percentiles: List[float] = [99.0, 99.5]
) -> Dict[str, float]:
    """
    Calculates candidate anomaly thresholds strictly from validation partition scores.

    Parameters
    ----------
    val_scores : array-like
        Scores from the validation partition (e.g., snapshots 160..199).
    sigmas : list of float, default [3.0, 4.0, 6.0]
        Standard deviation multipliers above validation mean.
    percentiles : list of float, default [99.0, 99.5]
        Percentiles of validation score distribution.

    Returns
    -------
    dict
        Mapping of threshold rule name to float threshold value:
        - 'mean_3sigma', 'mean_4sigma', 'mean_6sigma'
        - 'p99_0', 'p99_5'
        - 'val_mean', 'val_std', 'val_max'
    """
    arr = validate_scores_array(val_scores)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    thresholds = {
        "val_mean": mean_val,
        "val_std": std_val,
        "val_max": float(np.max(arr)),
    }
    
    for k in sigmas:
        k_str = str(k).replace(".", "_")
        thresholds[f"mean_{k_str}sigma"] = mean_val + k * std_val
        
    for p in percentiles:
        p_str = str(p).replace(".", "_")
        thresholds[f"p{p_str}"] = float(np.percentile(arr, p))

    return thresholds


def compute_threshold_exceedances(
    scores: Union[np.ndarray, pd.Series, List[float]],
    threshold: float
) -> np.ndarray:
    """
    Computes instantaneous boolean threshold exceedance flags (score > threshold).

    Parameters
    ----------
    scores : array-like
        1D array of anomaly scores.
    threshold : float
        Scalar anomaly threshold.

    Returns
    -------
    np.ndarray
        1D boolean array where True indicates score > threshold.
    """
    arr = validate_scores_array(scores)
    if not np.isfinite(threshold):
        raise ValueError(f"Threshold must be a finite float, got {threshold}.")
    return arr > threshold


def apply_persistence_filter(
    exceedance_flags: Union[np.ndarray, List[bool]],
    k: int = 1
) -> Tuple[np.ndarray, List[Dict[str, Any]], Optional[int], Optional[int]]:
    """
    Applies a k-consecutive persistence filter to threshold exceedance flags.

    Parameters
    ----------
    exceedance_flags : array-like of bool
        1D boolean array of instantaneous threshold exceedances.
    k : int, default 1
        Number of consecutive exceedances required to confirm a sustained alert. Must be >= 1.

    Returns
    -------
    sustained_flags : np.ndarray of bool
        Boolean array of same length as input, where snapshot i is True iff it belongs
        to a contiguous run of exceedances of length >= k.
    events : list of dict
        Metadata for each distinct sustained event episode:
        - 'event_id': int (1-indexed)
        - 'start_idx': int (start snapshot index t_start)
        - 'confirm_idx': int (online confirmation index t_confirm = t_start + k - 1)
        - 'end_idx': int (last snapshot index of the contiguous run)
        - 'length': int (total consecutive snapshots in run)
    first_sustained_start_idx : int or None
        t_start of the first sustained event, or None if no event occurred.
    first_sustained_confirm_idx : int or None
        t_confirm of the first sustained event, or None if no event occurred.

    Raises
    ------
    ValueError
        If k < 1 or exceedance_flags is invalid.
    """
    if k < 1:
        raise ValueError(f"Persistence k must be an integer >= 1, got {k}.")
        
    flags = np.asarray(exceedance_flags, dtype=bool)
    if flags.ndim != 1:
        raise ValueError(f"Expected 1D boolean array, got shape {flags.shape}.")
    if len(flags) == 0:
        raise ValueError("Exceedance flags array is empty.")

    n = len(flags)
    sustained_flags = np.zeros(n, dtype=bool)
    events = []
    
    # Identify contiguous runs of True values
    i = 0
    event_counter = 1
    
    while i < n:
        if flags[i]:
            run_start = i
            while i < n and flags[i]:
                i += 1
            run_end = i - 1
            run_length = run_end - run_start + 1
            
            if run_length >= k:
                # Mark all snapshots in this sustained run as True
                sustained_flags[run_start:run_end + 1] = True
                confirm_idx = run_start + k - 1
                
                events.append({
                    "event_id": event_counter,
                    "start_idx": run_start,
                    "confirm_idx": confirm_idx,
                    "end_idx": run_end,
                    "length": run_length
                })
                event_counter += 1
        else:
            i += 1
            
    first_start = events[0]["start_idx"] if events else None
    first_confirm = events[0]["confirm_idx"] if events else None

    return sustained_flags, events, first_start, first_confirm


def compute_evaluation_metrics(
    scores: Union[np.ndarray, pd.Series, List[float]],
    threshold: float,
    k: int = 1,
    timestamps: Optional[Union[np.ndarray, pd.Series, List[Any]]] = None,
    fit_end_idx: int = DEFAULT_FIT_END_IDX,
    val_end_idx: int = DEFAULT_VAL_END_IDX
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for a single score vector under a given
    threshold and persistence rule k.

    Parameters
    ----------
    scores : array-like
        1D array of anomaly scores across all dataset snapshots (e.g. N=984).
    threshold : float
        Frozen anomaly score threshold.
    k : int, default 1
        Persistence filter (consecutive exceedances required).
    timestamps : array-like, optional
        1D array of snapshot timestamps corresponding to scores.
    fit_end_idx : int, default 160
        Fit partition boundary.
    val_end_idx : int, default 200
        Validation partition boundary (eval partition starts at val_end_idx).

    Returns
    -------
    dict
        Comprehensive evaluation metrics dictionary:
        - Partition bounds and parameters (fit_end_idx, val_end_idx, threshold, k)
        - Exceedance counts (raw and sustained)
        - False alarm counts and rates (raw and sustained) for reference period 0..val_end_idx-1
        - Evaluation partition metrics (200..N-1):
          - eval_flagged_count, eval_flagged_fraction
          - first_eval_sustained_start_idx
          - first_eval_sustained_confirm_idx
          - first_eval_sustained_confirm_timestamp
          - proxy_duration_to_end
          - post_confirm_flagged_fraction
    """
    arr = validate_scores_array(scores)
    n_total = len(arr)
    
    if val_end_idx >= n_total:
        raise ValueError(
            f"val_end_idx ({val_end_idx}) must be strictly less than total rows ({n_total})."
        )
    if fit_end_idx >= val_end_idx:
        raise ValueError(f"fit_end_idx ({fit_end_idx}) must be < val_end_idx ({val_end_idx}).")

    ts_arr = None
    if timestamps is not None:
        ts_arr = np.asarray(timestamps)
        if len(ts_arr) != n_total:
            raise ValueError(f"Timestamps length ({len(ts_arr)}) does not match scores length ({n_total}).")

    # Step 1: Raw instantaneous exceedance flags
    raw_flags = compute_threshold_exceedances(arr, threshold)
    
    # Step 2: Persistence filtering
    sustained_flags, events, _, _ = apply_persistence_filter(raw_flags, k=k)
    
    # Step 3: Reference period false alarm metrics (snapshots 0 .. val_end_idx - 1, N=val_end_idx)
    healthy_n = val_end_idx
    raw_fa_count = int(np.sum(raw_flags[:healthy_n]))
    sustained_fa_count = int(np.sum(sustained_flags[:healthy_n]))
    
    raw_fa_rate = float(raw_fa_count / healthy_n)
    sustained_fa_rate = float(sustained_fa_count / healthy_n)
    
    # Healthy period distinct sustained event episodes
    healthy_events = [e for e in events if e["start_idx"] < healthy_n]
    healthy_event_count = len(healthy_events)

    # Step 4: Evaluation period metrics (snapshots val_end_idx .. n_total - 1)
    eval_n = n_total - val_end_idx
    eval_raw_flags = raw_flags[val_end_idx:]
    eval_sustained_flags = sustained_flags[val_end_idx:]
    
    eval_exceed_count = int(np.sum(eval_raw_flags))
    eval_flagged_fraction = float(eval_exceed_count / eval_n) if eval_n > 0 else 0.0
    
    # Filter distinct sustained events that START in the evaluation period
    eval_events = [e for e in events if e["start_idx"] >= val_end_idx]
    eval_event_count = len(eval_events)
    
    first_eval_start = eval_events[0]["start_idx"] if eval_events else None
    first_eval_confirm = eval_events[0]["confirm_idx"] if eval_events else None
    
    first_eval_confirm_ts = None
    proxy_duration = None
    post_confirm_flagged_fraction = None
    
    if first_eval_confirm is not None and ts_arr is not None:
        raw_ts = ts_arr[first_eval_confirm]
        try:
            ts_confirm_dt = pd.to_datetime(raw_ts)
            first_eval_confirm_ts = str(ts_confirm_dt)
            last_ts = ts_arr[-1]
            ts_last_dt = pd.to_datetime(last_ts)
            proxy_duration = str(ts_last_dt - ts_confirm_dt)
        except Exception:
            first_eval_confirm_ts = str(raw_ts)
            proxy_duration = "N/A"
            
    if first_eval_confirm is not None:
        post_confirm_slice = raw_flags[first_eval_confirm:]
        post_confirm_len = len(post_confirm_slice)
        if post_confirm_len > 0:
            post_confirm_flagged_fraction = float(np.sum(post_confirm_slice) / post_confirm_len)

    metrics = {
        "n_total": n_total,
        "fit_end_idx": fit_end_idx,
        "val_end_idx": val_end_idx,
        "threshold": float(threshold),
        "k": int(k),
        
        # Total counts across whole timeline
        "total_exceed_count": int(np.sum(raw_flags)),
        "total_sustained_flagged_count": int(np.sum(sustained_flags)),
        "total_sustained_events": len(events),
        
        # Healthy reference period metrics (0..val_end_idx-1)
        "healthy_n": healthy_n,
        "raw_false_alarm_count": raw_fa_count,
        "raw_false_alarm_rate": raw_fa_rate,
        "sustained_false_alarm_count": sustained_fa_count,
        "sustained_false_alarm_rate": sustained_fa_rate,
        "healthy_sustained_events_count": healthy_event_count,
        
        # Evaluation period metrics (val_end_idx..n_total-1)
        "eval_n": eval_n,
        "eval_exceed_count": eval_exceed_count,
        "eval_flagged_fraction": eval_flagged_fraction,
        "eval_sustained_events_count": eval_event_count,
        
        # First sustained alert in evaluation period
        "first_eval_sustained_start_idx": first_eval_start,
        "first_eval_sustained_confirm_idx": first_eval_confirm,
        "first_eval_sustained_confirm_ts": first_eval_confirm_ts,
        "proxy_duration_to_end": proxy_duration,
        "post_confirm_flagged_fraction": post_confirm_flagged_fraction,
    }

    return metrics


def aggregate_channel_alerts_or(
    channel_flags_dict: Dict[str, np.ndarray],
    k: int = 1
) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    """
    Computes multi-channel shaft aggregate alert using Logical OR combination.

    Parameters
    ----------
    channel_flags_dict : dict
        Mapping of channel name (e.g., 'ch1', 'ch2') to 1D boolean array of sustained flags.
    k : int, default 1
        Persistence value used to determine confirmation offsets if raw exceedances passed.

    Returns
    -------
    shaft_sustained_flags : np.ndarray of bool
        1D boolean array where snapshot t is True iff at least one channel is True at t.
    first_shaft_start_idx : int or None
        Minimum start index across all channels that triggered a sustained alert.
    first_shaft_confirm_idx : int or None
        Minimum confirmation index across all channels that triggered a sustained alert.
    """
    if not channel_flags_dict:
        raise ValueError("channel_flags_dict is empty.")

    arrays = list(channel_flags_dict.values())
    n = len(arrays[0])
    for ch, arr in channel_flags_dict.items():
        if len(arr) != n:
            raise ValueError(f"Channel '{ch}' array length ({len(arr)}) mismatch with {n}.")

    # Logical OR across channels
    shaft_flags = np.zeros(n, dtype=bool)
    for arr in arrays:
        shaft_flags |= np.asarray(arr, dtype=bool)

    # Find first sustained event start and confirm index across channels
    _, events, first_start, first_confirm = apply_persistence_filter(shaft_flags, k=k)

    return shaft_flags, first_start, first_confirm
