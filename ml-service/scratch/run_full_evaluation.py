"""
scratch/run_full_evaluation.py - Script to compute Phase 8 full-dataset evaluation metrics
and generate plots for MachineMind AI.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import (
    calculate_validation_thresholds,
    compute_threshold_exceedances,
    apply_persistence_filter,
    compute_evaluation_metrics,
    aggregate_channel_alerts_or,
    DEFAULT_FIT_END_IDX,
    DEFAULT_VAL_END_IDX,
    DEFAULT_EVAL_END_IDX,
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def run_evaluation():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "processed"
    fig_dir = base_dir / "reports" / "figures" / "evaluation"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load score files
    df_baseline = pd.read_csv(data_dir / "baseline_scores_set2.csv")
    df_models = pd.read_csv(data_dir / "model_scores_set2.csv")

    timestamps = pd.to_datetime(df_baseline["timestamp"])
    snapshots = df_baseline["file_index"].to_numpy()

    print(f"Loaded {len(df_baseline)} snapshots from baseline and model score CSVs.")

    # Primary models and score columns
    # Frozen validation P99 thresholds from Phase 7 report:
    # ch1: iForest=0.5310, PCA=1.0721
    # ch2: iForest=0.5184, PCA=0.8657
    # ch3: iForest=0.5990, PCA=0.6793
    # ch4: iForest=0.6247, PCA=1.3281
    # Baseline: score_rms_z T=3.0

    # 1. Evaluate Primary Metrics across k=1, 3, 5
    score_columns = {
        "Baseline RMS Z": (df_baseline["score_rms_z"].to_numpy(), 3.0),
        "iForest Ch1": (df_models["score_iforest_ch1"].to_numpy(), 0.5310),
        "iForest Ch2": (df_models["score_iforest_ch2"].to_numpy(), 0.5184),
        "iForest Ch3": (df_models["score_iforest_ch3"].to_numpy(), 0.5990),
        "iForest Ch4": (df_models["score_iforest_ch4"].to_numpy(), 0.6247),
        "PCA Ch1": (df_models["score_pca_ch1"].to_numpy(), 1.0721),
        "PCA Ch2": (df_models["score_pca_ch2"].to_numpy(), 0.8657),
        "PCA Ch3": (df_models["score_pca_ch3"].to_numpy(), 0.6793),
        "PCA Ch4": (df_models["score_pca_ch4"].to_numpy(), 1.3281),
    }

    results_primary = []

    for name, (scores, thresh) in score_columns.items():
        for k in [1, 3, 5]:
            m = compute_evaluation_metrics(
                scores=scores,
                threshold=thresh,
                k=k,
                timestamps=timestamps
            )
            m["model_name"] = name
            results_primary.append(m)

    df_primary_results = pd.DataFrame(results_primary)
    print("\n=== Primary Evaluation Results (Sample) ===")
    cols_to_show = ["model_name", "k", "raw_false_alarm_count", "sustained_false_alarm_count", "eval_exceed_count", "first_eval_sustained_start_idx", "first_eval_sustained_confirm_idx", "first_eval_sustained_confirm_ts", "proxy_duration_to_end"]
    print(df_primary_results[cols_to_show].to_string())

    # Save summary table CSV for notebook/report usage
    df_primary_results.to_csv(base_dir / "scratch" / "primary_evaluation_summary.csv", index=False)

    # 2. Multi-channel shaft aggregation (Logical OR)
    print("\n=== Logical OR Multi-Channel Shaft Aggregation ===")
    for m_type, prefix in [("Isolation Forest", "score_iforest_"), ("PCA Error", "score_pca_")]:
        for k in [1, 3, 5]:
            ch_flags = {}
            for ch in ["ch1", "ch2", "ch3", "ch4"]:
                col = f"{prefix}{ch}"
                if m_type == "Isolation Forest":
                    t_val = {"ch1":0.5310, "ch2":0.5184, "ch3":0.5990, "ch4":0.6247}[ch]
                else:
                    t_val = {"ch1":1.0721, "ch2":0.8657, "ch3":0.6793, "ch4":1.3281}[ch]
                raw = compute_threshold_exceedances(df_models[col].to_numpy(), t_val)
                sust, _, _, _ = apply_persistence_filter(raw, k=k)
                ch_flags[ch] = sust

            shaft_flags, start_idx, confirm_idx = aggregate_channel_alerts_or(ch_flags, k=k)
            confirm_ts = timestamps[confirm_idx] if confirm_idx is not None else "None"
            proxy_dur = timestamps.iloc[-1] - pd.to_datetime(confirm_ts) if confirm_idx is not None else "N/A"
            print(f"[{m_type}] k={k} -> Shaft Start Index: {start_idx}, Confirm Index: {confirm_idx}, Confirm TS: {confirm_ts}, Proxy Lead: {proxy_dur}")

    # 3. Sensitivity Analysis across threshold rules (mean+3s, mean+4s, mean+6s, P99, P99.5)
    print("\n=== Sensitivity Analysis ===")
    sensitivity_rows = []
    channels = ["ch1", "ch2", "ch3", "ch4"]

    for m_type in ["iforest", "pca"]:
        for ch in channels:
            col = f"score_{m_type}_{ch}"
            scores = df_models[col].to_numpy()
            val_scores = scores[160:200]
            thresh_rules = calculate_validation_thresholds(val_scores, sigmas=[3.0, 4.0, 6.0], percentiles=[99.0, 99.5])
            
            for r_name, t_val in thresh_rules.items():
                if r_name in ["val_mean", "val_std", "val_max"]:
                    continue
                for k in [1, 3, 5]:
                    m = compute_evaluation_metrics(scores=scores, threshold=t_val, k=k, timestamps=timestamps)
                    m["model_type"] = m_type
                    m["channel"] = ch
                    m["threshold_rule"] = r_name
                    m["threshold_value"] = t_val
                    sensitivity_rows.append(m)

    df_sensitivity = pd.DataFrame(sensitivity_rows)
    df_sensitivity.to_csv(base_dir / "scratch" / "sensitivity_analysis_summary.csv", index=False)
    print(f"Generated {len(df_sensitivity)} sensitivity experiment combinations.")

    # 4. Generate Figures
    print("\n=== Generating Figures ===")

    # Figure 1: Baseline RMS Z-Score Timeline with T=3.0
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(snapshots, df_baseline["score_rms_z"], color="#1f77b4", linewidth=1.2, label="RMS Z-Score (score_rms_z)")
    ax.axhline(3.0, color="#d62728", linestyle="--", linewidth=1.5, label="Primary Baseline Threshold (T = 3.0)")
    ax.axvline(160, color="#7f7f7f", linestyle=":", label="Fit / Validation Split (idx 160)")
    ax.axvline(200, color="#2ca02c", linestyle=":", label="Validation / Eval Split (idx 200)")
    ax.set_title("Statistical Baseline: RMS Composite Z-Score Timeline (NASA IMS Set 2)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Snapshot Index (0 to 983)", fontsize=11)
    ax.set_ylabel("RMS Z-Score ($S_{RMS\_Z}$)", fontsize=11)
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "eval_baseline_timeline.png", dpi=300)
    plt.close(fig)

    # Figure 2: Isolation Forest Score Timelines Per Channel
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    ch_thresh_iforest = {"ch1":0.5310, "ch2":0.5184, "ch3":0.5990, "ch4":0.6247}
    for idx, ch in enumerate(channels):
        ax = axes[idx//2, idx%2]
        scores = df_models[f"score_iforest_{ch}"]
        t_val = ch_thresh_iforest[ch]
        ax.plot(snapshots, scores, color="#2b5c8f", linewidth=1.0, label=f"iForest {ch.upper()} Score")
        ax.axhline(t_val, color="#e74c3c", linestyle="--", label=f"Validation $P_{{99}}$ Threshold ({t_val:.4f})")
        ax.axvline(160, color="#7f7f7f", linestyle=":", alpha=0.7)
        ax.axvline(200, color="#27ae60", linestyle=":", alpha=0.7)
        ax.set_title(f"Isolation Forest — Channel {ch.upper()}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Anomaly Score", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Snapshot Index", fontsize=10)
    fig.suptitle("Isolation Forest Anomaly Scores across 4 Channels with Validation $P_{99}$ Thresholds", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "eval_iforest_channels.png", dpi=300)
    plt.close(fig)

    # Figure 3: PCA Reconstruction Error Timelines Per Channel
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    ch_thresh_pca = {"ch1":1.0721, "ch2":0.8657, "ch3":0.6793, "ch4":1.3281}
    for idx, ch in enumerate(channels):
        ax = axes[idx//2, idx%2]
        scores = df_models[f"score_pca_{ch}"]
        t_val = ch_thresh_pca[ch]
        ax.plot(snapshots, scores, color="#8e44ad", linewidth=1.0, label=f"PCA {ch.upper()} Error")
        ax.axhline(t_val, color="#e74c3c", linestyle="--", label=f"Validation $P_{{99}}$ Threshold ({t_val:.4f})")
        ax.axvline(160, color="#7f7f7f", linestyle=":", alpha=0.7)
        ax.axvline(200, color="#27ae60", linestyle=":", alpha=0.7)
        ax.set_yscale("log")
        ax.set_title(f"PCA Reconstruction Error — Channel {ch.upper()}", fontsize=12, fontweight="bold")
        ax.set_ylabel("MSE Reconstruction Error (log scale)", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Snapshot Index", fontsize=10)
    fig.suptitle("PCA Reconstruction Error across 4 Channels with Validation $P_{99}$ Thresholds", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "eval_pca_channels.png", dpi=300)
    plt.close(fig)

    # Figure 4: Persistence Comparison (k=1 vs k=3 vs k=5) on Channel 1
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    s_ch1 = df_models["score_iforest_ch1"].to_numpy()
    t_ch1 = 0.5310
    raw_flags = compute_threshold_exceedances(s_ch1, t_ch1)
    
    for i, k_val in enumerate([1, 3, 5]):
        ax = axes[i]
        sust_flags, events, s_idx, c_idx = apply_persistence_filter(raw_flags, k=k_val)
        ax.plot(snapshots, s_ch1, color="#7f8c8d", alpha=0.5, linewidth=0.8, label="iForest Ch1 Score")
        ax.axhline(t_ch1, color="#c0392b", linestyle="--", alpha=0.7)
        ax.fill_between(snapshots, 0.3, 0.8, where=sust_flags, color="#e74c3c", alpha=0.3, label=f"Sustained Alert (k={k_val})")
        if c_idx is not None:
            ax.axvline(c_idx, color="#27ae60", linestyle="-", linewidth=1.5, label=f"Online Confirm (idx {c_idx})")
        ax.set_title(f"Persistence k = {k_val} (Confirmed at idx {c_idx if c_idx is not None else 'None'})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
    axes[2].set_xlabel("Snapshot Index", fontsize=10)
    fig.suptitle("Effect of Persistence Filtering (k=1, 3, 5) on Isolation Forest Bearing 1 Alerts", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "eval_persistence_comparison.png", dpi=300)
    plt.close(fig)

    # Figure 5: Threshold Sensitivity Comparison (Alert Confirmation Index across rules)
    fig, ax = plt.subplots(figsize=(10, 5))
    df_sens_k3 = df_sensitivity[df_sensitivity["k"] == 3]
    sns.barplot(data=df_sens_k3, x="channel", y="first_eval_sustained_confirm_idx", hue="threshold_rule", ax=ax, palette="Blues")
    ax.set_title("First Sustained Alert Confirmation Index under Persistence k=3 across Threshold Rules", fontsize=13, fontweight="bold")
    ax.set_xlabel("Vibration Channel", fontsize=11)
    ax.set_ylabel("Confirmation Snapshot Index (t_confirm)", fontsize=11)
    ax.set_ylim(200, 984)
    ax.legend(title="Threshold Rule", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "eval_threshold_sensitivity.png", dpi=300)
    plt.close(fig)

    # Figure 6: Evaluation Period Score Distributions (Boxplot / Violinplot)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Slice evaluation period
    eval_df = df_models.iloc[200:].copy()
    iforest_cols = [f"score_iforest_{ch}" for ch in channels]
    pca_cols = [f"score_pca_{ch}" for ch in channels]

    sns.boxplot(data=eval_df[iforest_cols], ax=axes[0], palette="Set2")
    axes[0].set_title("Isolation Forest Score Distributions (Eval Partition 200..983)", fontsize=11, fontweight="bold")
    axes[0].set_xticklabels(["Ch1", "Ch2", "Ch3", "Ch4"])
    axes[0].set_ylabel("Anomaly Score")

    sns.boxplot(data=eval_df[pca_cols], ax=axes[1], palette="Set2")
    axes[1].set_title("PCA Reconstruction Error Distributions (Eval Partition 200..983)", fontsize=11, fontweight="bold")
    axes[1].set_xticklabels(["Ch1", "Ch2", "Ch3", "Ch4"])
    axes[1].set_ylabel("MSE Error (log scale)")
    axes[1].set_yscale("log")

    fig.suptitle("Evaluation Partition Anomaly Score Distributions across Bearings", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "eval_score_distributions.png", dpi=300)
    plt.close(fig)

    print("All 6 figures saved successfully under reports/figures/evaluation/.")

if __name__ == "__main__":
    run_evaluation()
