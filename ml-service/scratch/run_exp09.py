"""
scratch/run_exp09.py - Script to execute EXP-09 Persistence Filter Comparison.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import AnomalyDetectionPipeline
from src.evaluation import compute_threshold_exceedances, apply_persistence_filter

def run_exp09():
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
    
    persistence_k_list = [1, 3, 5]
    feature_sets = [("Primary 36-Feature Set", full_cols), ("28-Feature Time-Domain Set", time_cols)]
    
    results = {}
    
    for fset_name, feature_list in feature_sets:
        print(f"\n=======================================================")
        print(f"FEATURE SET: {fset_name} ({len(feature_list)} features)")
        print(f"=======================================================")
        
        # Fit pipelines once per channel (fixed model, scaler, threshold)
        pipelines = {}
        scores_dict = {}
        thresholds_dict = {}
        raw_exceed_diag_dict = {}
        
        for ch in channels:
            ch_cols = [c for c in feature_list if c.startswith(f"{ch}_")]
            X_all = df_features[ch_cols].to_numpy(dtype=np.float64)
            X_fit = X_all[:fit_end]
            
            pipeline = AnomalyDetectionPipeline(
                model_type="iforest",
                scaler_type="robust",
                model_params={"n_estimators": 100, "max_samples": "auto"},
                random_state=42
            )
            pipeline.fit(X_fit)
            
            scores_all = pipeline.compute_anomaly_scores(X_all)
            scores_cal = scores_all[fit_end:cal_end]
            t_cal_p99 = float(np.percentile(scores_cal, 99.0))
            
            scores_diag = scores_all[cal_end:diag_end]
            raw_exceed_diag = scores_diag > t_cal_p99
            
            pipelines[ch] = pipeline
            scores_dict[ch] = scores_all
            thresholds_dict[ch] = t_cal_p99
            raw_exceed_diag_dict[ch] = raw_exceed_diag

        for k in persistence_k_list:
            is_control = (k == 3)
            k_label = f"k={k} ({'Control' if is_control else 'Candidate'})"
            delay_mins = (k - 1) * 10
            
            print(f"\n--- Evaluating Persistence Filter: {k_label} (Confirmation Delay: {delay_mins} mins) ---")
            
            config_res = {}
            total_raw_exceed = 0
            total_events = 0
            total_sust_snapshots = 0
            
            for ch in channels:
                raw_flags = raw_exceed_diag_dict[ch]
                sust_flags, events, first_start, first_confirm = apply_persistence_filter(raw_flags, k=k)
                
                raw_cnt = int(np.sum(raw_flags))
                event_cnt = len(events)
                sust_cnt = int(np.sum(sust_flags))
                
                total_raw_exceed += raw_cnt
                total_events += event_cnt
                total_sust_snapshots += sust_cnt
                
                config_res[ch] = {
                    "raw_exceed_snapshots": raw_cnt,
                    "sustained_events": event_cnt,
                    "sustained_snapshots": sust_cnt,
                    "first_start": first_start,
                    "first_confirm": first_confirm
                }
                
                print(f"  Channel {ch}: Raw Exceedances={raw_cnt}/20, Sustained Events={event_cnt}, Sustained Snapshots={sust_cnt}/20")
                
            print(f"-> Total Raw Threshold Exceedances ({k_label}, N=80 total): {total_raw_exceed}")
            print(f"-> Total Sustained Alert Events ({k_label}, N=80 total): {total_events}")
            print(f"-> Total Sustained Anomalous Snapshots ({k_label}, N=80 total): {total_sust_snapshots}")
            print(f"-> Online Confirmation Delay ({k_label}): {delay_mins} minutes ({k-1} snapshots)")
            
            key = f"{fset_name} | {k_label}"
            results[key] = {
                "per_channel": config_res,
                "total_raw_exceed": total_raw_exceed,
                "total_events": total_events,
                "total_sust_snapshots": total_sust_snapshots,
                "delay_mins": delay_mins
            }
            
    return results

if __name__ == "__main__":
    run_exp09()
