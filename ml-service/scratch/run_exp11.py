"""
scratch/run_exp11.py - Script to execute EXP-11 Per-Channel vs Pooled Modeling.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import AnomalyDetectionPipeline
from src.evaluation import compute_threshold_exceedances, apply_persistence_filter

def run_exp11():
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
    
    feature_sets = [("Primary 36-Feature Set", full_cols), ("28-Feature Time-Domain Set", time_cols)]
    
    results = {}
    
    for fset_name, feature_list in feature_sets:
        print(f"\n=======================================================")
        print(f"FEATURE SET: {fset_name} ({len(feature_list)} features)")
        print(f"=======================================================")
        
        # --- Config A: Per-Channel Modeling (Control) ---
        print("\n--- Evaluating Config A: Per-Channel Modeling (Control) ---")
        per_ch_res = {}
        total_per_ch_raw_fa = 0
        total_per_ch_sust_fa = 0
        
        for ch in channels:
            ch_cols = [c for c in feature_list if c.startswith(f"{ch}_")]
            X_all_ch = df_features[ch_cols].to_numpy(dtype=np.float64)
            X_fit_ch = X_all_ch[:fit_end]
            
            pipe_ch = AnomalyDetectionPipeline(
                model_type="iforest",
                scaler_type="robust",
                model_params={"n_estimators": 100, "max_samples": "auto"},
                random_state=42
            )
            pipe_ch.fit(X_fit_ch)
            
            scores_all_ch = pipe_ch.compute_anomaly_scores(X_all_ch)
            scores_cal_ch = scores_all_ch[fit_end:cal_end]
            t_cal_p99_ch = float(np.percentile(scores_cal_ch, 99.0))
            
            scores_diag_ch = scores_all_ch[cal_end:diag_end]
            raw_exceed_diag_ch = scores_diag_ch > t_cal_p99_ch
            sust_flags_diag_ch, _, _, _ = apply_persistence_filter(raw_exceed_diag_ch, k=3)
            
            raw_fa_cnt = int(np.sum(raw_exceed_diag_ch))
            sust_fa_cnt = int(np.sum(sust_flags_diag_ch))
            
            total_per_ch_raw_fa += raw_fa_cnt
            total_per_ch_sust_fa += sust_fa_cnt
            
            per_ch_res[ch] = {
                "t_cal_p99": t_cal_p99_ch,
                "diag_raw_fa": raw_fa_cnt,
                "diag_sust_fa": sust_fa_cnt
            }
            print(f"  Channel {ch}: T_cal_P99={t_cal_p99_ch:.6f}, Diag Raw FA={raw_fa_cnt}/20, Diag Sust FA (k=3)={sust_fa_cnt}/20")
            
        print(f"-> Total Diagnostic Raw False-Alarm Snapshots (Per-Channel Control, N=80): {total_per_ch_raw_fa}")
        print(f"-> Total Diagnostic Sustained False-Alarm Snapshots (Per-Channel Control, k=3, N=80): {total_per_ch_sust_fa}")
        
        # --- Config B: Pooled Modeling (Candidate) ---
        print("\n--- Evaluating Config B: Pooled Joint Modeling (Candidate) ---")
        X_all_pooled = df_features[feature_list].to_numpy(dtype=np.float64)
        X_fit_pooled = X_all_pooled[:fit_end]
        
        pipe_pooled = AnomalyDetectionPipeline(
            model_type="iforest",
            scaler_type="robust",
            model_params={"n_estimators": 100, "max_samples": "auto"},
            random_state=42
        )
        pipe_pooled.fit(X_fit_pooled)
        
        scores_all_pooled = pipe_pooled.compute_anomaly_scores(X_all_pooled)
        scores_cal_pooled = scores_all_pooled[fit_end:cal_end]
        t_cal_p99_pooled = float(np.percentile(scores_cal_pooled, 99.0))
        
        scores_diag_pooled = scores_all_pooled[cal_end:diag_end]
        raw_exceed_diag_pooled = scores_diag_pooled > t_cal_p99_pooled
        sust_flags_diag_pooled, _, _, _ = apply_persistence_filter(raw_exceed_diag_pooled, k=3)
        
        pooled_raw_fa_cnt = int(np.sum(raw_exceed_diag_pooled))
        pooled_sust_fa_cnt = int(np.sum(sust_flags_diag_pooled))
        
        print(f"  Pooled System Score: T_cal_P99={t_cal_p99_pooled:.6f}, Diag Raw FA={pooled_raw_fa_cnt}/20, Diag Sust FA (k=3)={pooled_sust_fa_cnt}/20")
        print(f"-> Total Diagnostic Raw False-Alarm Snapshots (Pooled Candidate, N=20): {pooled_raw_fa_cnt}")
        print(f"-> Total Diagnostic Sustained False-Alarm Snapshots (Pooled Candidate, k=3, N=20): {pooled_sust_fa_cnt}")
        
        results[f"{fset_name} | Per-Channel Control"] = {
            "per_channel": per_ch_res,
            "total_diag_raw_fa": total_per_ch_raw_fa,
            "total_diag_sust_fa": total_per_ch_sust_fa
        }
        
        results[f"{fset_name} | Pooled Candidate"] = {
            "t_cal_p99": t_cal_p99_pooled,
            "total_diag_raw_fa": pooled_raw_fa_cnt,
            "total_diag_sust_fa": pooled_sust_fa_cnt
        }
        
    return results

if __name__ == "__main__":
    run_exp11()
