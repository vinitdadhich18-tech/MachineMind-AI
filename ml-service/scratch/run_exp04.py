"""
scratch/run_exp04.py - Script to execute EXP-04 Feature Set Ablation.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import AnomalyDetectionPipeline, calculate_validation_thresholds
from src.evaluation import compute_threshold_exceedances, apply_persistence_filter, compute_evaluation_metrics

def run_exp04():
    features_csv = Path(__file__).resolve().parent.parent / "data" / "processed" / "features_set2.csv"
    if not features_csv.exists():
        raise FileNotFoundError(f"features_set2.csv not found at {features_csv}")

    df_features = pd.read_csv(features_csv)
    
    # Partitions
    fit_end = 160
    cal_end = 180
    diag_end = 200

    channels = ["ch1", "ch2", "ch3", "ch4"]
    
    # 1. Full Feature Set (36 features)
    full_cols = [c for c in df_features.columns if any(c.startswith(f"{ch}_") for ch in channels)]
    
    # 2. Time-Domain Feature Set (28 features)
    time_cols = [c for c in full_cols if not ("spectral_energy" in c or "spectral_centroid" in c)]
    
    print(f"Total rows in features_set2.csv: {len(df_features)}")
    print(f"Full feature set count: {len(full_cols)}")
    print(f"Time-domain feature set count: {len(time_cols)}")
    
    results = {}
    
    for config_name, feature_list in [("Full Feature Set (36)", full_cols), ("Time-Domain Subset (28)", time_cols)]:
        print(f"\n==========================================")
        print(f"Evaluating Config: {config_name}")
        print(f"==========================================")
        
        config_res = {}
        total_diag_raw_fa = 0
        total_diag_sust_fa = 0
        
        for ch in channels:
            ch_cols = [c for c in feature_list if c.startswith(f"{ch}_")]
            X_all = df_features[ch_cols].to_numpy(dtype=np.float64)
            
            # Check shape and NaN/Inf
            assert not np.isnan(X_all).any(), f"NaN found in {ch}"
            assert not np.isinf(X_all).any(), f"Inf found in {ch}"
            
            # Fit model strictly on 0..159
            X_fit = X_all[:fit_end]
            pipeline = AnomalyDetectionPipeline(model_type="iforest", random_state=42)
            pipeline.fit(X_fit)
            
            # Compute scores across all snapshots
            scores_all = pipeline.compute_anomaly_scores(X_all)
            
            # Calibration scores (160..179)
            scores_cal = scores_all[fit_end:cal_end]
            sigma_cal = float(np.std(scores_cal, ddof=1))
            t_cal_p99 = float(np.percentile(scores_cal, 99.0))
            
            # Diagnostic scores (180..199)
            scores_diag = scores_all[cal_end:diag_end]
            raw_exceed_diag = scores_diag > t_cal_p99
            
            # Sustained exceedances (k=3) on diagnostic window
            # Apply persistence filter on diagnostic period
            sust_flags_diag, events_diag, _, _ = apply_persistence_filter(raw_exceed_diag, k=3)
            
            raw_fa_count_diag = int(np.sum(raw_exceed_diag))
            sust_fa_count_diag = int(np.sum(sust_flags_diag))
            
            total_diag_raw_fa += raw_fa_count_diag
            total_diag_sust_fa += sust_fa_count_diag
            
            config_res[ch] = {
                "n_features": len(ch_cols),
                "sigma_cal": sigma_cal,
                "t_cal_p99": t_cal_p99,
                "diag_raw_fa": raw_fa_count_diag,
                "diag_sust_fa": sust_fa_count_diag
            }
            
            print(f"  Channel {ch}: features={len(ch_cols)}, sigma_cal={sigma_cal:.6f}, T_cal_P99={t_cal_p99:.6f}, Diag Raw FA (180..199)={raw_fa_count_diag}/20, Diag Sust FA (k=3)={sust_fa_count_diag}/20")
            
        print(f"Total Diagnostic Raw False-Alarm Snapshots (across 4 channels, N=20 each, max 80): {total_diag_raw_fa}")
        print(f"Total Diagnostic Sustained False-Alarm Snapshots (across 4 channels, k=3, N=20 each, max 80): {total_diag_sust_fa}")
        
        results[config_name] = {
            "per_channel": config_res,
            "total_diag_raw_fa": total_diag_raw_fa,
            "total_diag_sust_fa": total_diag_sust_fa
        }
        
    return results

if __name__ == "__main__":
    run_exp04()
