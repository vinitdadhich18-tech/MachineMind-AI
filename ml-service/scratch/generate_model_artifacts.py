"""
scratch/generate_model_artifacts.py

Runs model pipelines on real features_set2.csv dataset, generates plots,
saves figures to reports/figures/models/ and exports model_scores_set2.csv.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import (
    train_and_evaluate_bearing_models,
    DEFAULT_REF_END_IDX,
    DEFAULT_VAL_END_IDX,
    DEFAULT_RANDOM_STATE,
)

# Aligned styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Ensure output directories exist
fig_dir = Path("reports/figures/models")
fig_dir.mkdir(parents=True, exist_ok=True)
proc_dir = Path("data/processed")

# 1. Load data
df_features = pd.read_csv(proc_dir / "features_set2.csv")
df_baseline = pd.read_csv(proc_dir / "baseline_scores_set2.csv")

# 2. Fit models & evaluate scores
df_scores, results_meta = train_and_evaluate_bearing_models(
    df_features,
    ref_end_idx=DEFAULT_REF_END_IDX,
    val_end_idx=DEFAULT_VAL_END_IDX,
    model_types=["iforest", "pca"],
    random_state=DEFAULT_RANDOM_STATE
)

# Merge overall baseline Z-score
if "score_rms_z" in df_baseline.columns:
    df_scores["score_rms_z_overall"] = df_baseline["score_rms_z"]

# Save merged model scores to data/processed/
df_scores.to_csv(proc_dir / "model_scores_set2.csv", index=False)
print(f"Saved model scores to {proc_dir / 'model_scores_set2.csv'}")

# Define color palette
COLORS = {
    "baseline": "#1f77b4",  # Blue
    "iforest": "#ff7f0e",   # Orange
    "pca": "#2ca02c",       # Green
    "fit_bg": "#e6f2ff",
    "val_bg": "#fff2e6",
    "seq_bg": "#ffffff",
}

# Figure 1: Bearing 1 Model Scores Comparison over Time
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# (A) Baseline Z-Score
ax0 = axes[0]
ax0.plot(df_scores["file_index"], df_scores["score_rms_z_overall"], color=COLORS["baseline"], linewidth=1.2, label="Baseline Overall RMS Z-Score")
ax0.axhline(3.0, color="red", linestyle="--", linewidth=1.0, label="Baseline Alert Threshold (Z=3.0)")
ax0.set_ylabel("RMS Z-Score")
ax0.set_title("Phase 6 Statistical Baseline Composite RMS Z-Score", fontsize=12, fontweight="bold")
ax0.legend(loc="upper left")

# (B) Isolation Forest Score
ax1 = axes[1]
iforest_thresh = results_meta["thresholds"]["iforest"]["ch1"]["p99_0"]
ax1.plot(df_scores["file_index"], df_scores["score_iforest_ch1"], color=COLORS["iforest"], linewidth=1.2, label="Isolation Forest Score (Ch1)")
ax1.axhline(iforest_thresh, color="red", linestyle="--", linewidth=1.0, label=f"Validation P99 Threshold ({iforest_thresh:.3f})")
ax1.set_ylabel("iForest Score (0.5 - dec_func)")
ax1.set_title("Bearing 1: Isolation Forest Anomaly Score", fontsize=12, fontweight="bold")
ax1.legend(loc="upper left")

# (C) PCA Reconstruction Error
ax2 = axes[2]
pca_thresh = results_meta["thresholds"]["pca"]["ch1"]["p99_0"]
ax2.plot(df_scores["file_index"], df_scores["score_pca_ch1"], color=COLORS["pca"], linewidth=1.2, label="PCA Reconstruction Error (Ch1)")
ax2.axhline(pca_thresh, color="red", linestyle="--", linewidth=1.0, label=f"Validation P99 Threshold ({pca_thresh:.3f})")
ax2.set_ylabel("MSE Reconstruction Error")
ax2.set_xlabel("Snapshot File Index (0..983)")
ax2.set_title("Bearing 1: PCA Reconstruction Error (k=2 Components)", fontsize=12, fontweight="bold")
ax2.legend(loc="upper left")

# Add period markers to all subplots
for ax in axes:
    ax.axvline(DEFAULT_REF_END_IDX, color="gray", linestyle=":", linewidth=1.2)
    ax.axvline(DEFAULT_VAL_END_IDX, color="gray", linestyle=":", linewidth=1.2)
    ax.axvspan(0, DEFAULT_REF_END_IDX, color=COLORS["fit_bg"], alpha=0.5, zorder=0)
    ax.axvspan(DEFAULT_REF_END_IDX, DEFAULT_VAL_END_IDX, color=COLORS["val_bg"], alpha=0.5, zorder=0)

plt.tight_layout()
fig.savefig(fig_dir / "model_scores_ch1_comparison.png", dpi=300)
plt.close(fig)
print("Saved model_scores_ch1_comparison.png")

# Figure 2: Isolation Forest Scores Across All 4 Channels
fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
axes = axes.flatten()
channels = ["ch1", "ch2", "ch3", "ch4"]

for i, ch in enumerate(channels):
    ax = axes[i]
    thresh = results_meta["thresholds"]["iforest"][ch]["p99_0"]
    ax.plot(df_scores["file_index"], df_scores[f"score_iforest_{ch}"], color=COLORS["iforest"], linewidth=1.1, label=f"Ch {i+1} iForest Score")
    ax.axhline(thresh, color="red", linestyle="--", linewidth=1.0, label=f"Val P99 ({thresh:.3f})")
    ax.axvline(DEFAULT_REF_END_IDX, color="gray", linestyle=":")
    ax.axvline(DEFAULT_VAL_END_IDX, color="gray", linestyle=":")
    ax.axvspan(0, DEFAULT_REF_END_IDX, color=COLORS["fit_bg"], alpha=0.4)
    ax.axvspan(DEFAULT_REF_END_IDX, DEFAULT_VAL_END_IDX, color=COLORS["val_bg"], alpha=0.4)
    ax.set_title(f"Bearing {i+1} (Channel {ch.upper()}) Isolation Forest", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    if i >= 2:
        ax.set_xlabel("Snapshot Index")
    if i % 2 == 0:
        ax.set_ylabel("Score (Higher = More Anomalous)")

plt.suptitle("Isolation Forest Anomaly Scores Across All 4 Bearings (Set 2)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(fig_dir / "model_scores_all_channels_iforest.png", dpi=300)
plt.close(fig)
print("Saved model_scores_all_channels_iforest.png")

# Figure 3: PCA Reconstruction Error Across All 4 Channels
fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
axes = axes.flatten()

for i, ch in enumerate(channels):
    ax = axes[i]
    thresh = results_meta["thresholds"]["pca"][ch]["p99_0"]
    ax.plot(df_scores["file_index"], df_scores[f"score_pca_{ch}"], color=COLORS["pca"], linewidth=1.1, label=f"Ch {i+1} PCA Error")
    ax.axhline(thresh, color="red", linestyle="--", linewidth=1.0, label=f"Val P99 ({thresh:.3f})")
    ax.axvline(DEFAULT_REF_END_IDX, color="gray", linestyle=":")
    ax.axvline(DEFAULT_VAL_END_IDX, color="gray", linestyle=":")
    ax.axvspan(0, DEFAULT_REF_END_IDX, color=COLORS["fit_bg"], alpha=0.4)
    ax.axvspan(DEFAULT_REF_END_IDX, DEFAULT_VAL_END_IDX, color=COLORS["val_bg"], alpha=0.4)
    ax.set_title(f"Bearing {i+1} (Channel {ch.upper()}) PCA Reconstruction Error", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    if i >= 2:
        ax.set_xlabel("Snapshot Index")
    if i % 2 == 0:
        ax.set_ylabel("MSE Error")

plt.suptitle("PCA Reconstruction Error Across All 4 Bearings (Set 2)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(fig_dir / "model_scores_all_channels_pca.png", dpi=300)
plt.close(fig)
print("Saved model_scores_all_channels_pca.png")

# Figure 4: Score Distributions across Periods for Bearing 1
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Slice periods
fit_idx = slice(0, DEFAULT_REF_END_IDX)
val_idx = slice(DEFAULT_REF_END_IDX, DEFAULT_VAL_END_IDX)
seq_idx = slice(DEFAULT_VAL_END_IDX, len(df_scores))

metrics = [("score_rms_z_overall", "Baseline Z-Score", COLORS["baseline"]),
           ("score_iforest_ch1", "Isolation Forest Score", COLORS["iforest"]),
           ("score_pca_ch1", "PCA Error (Log Scale)", COLORS["pca"])]

for ax, (col, title, color) in zip(axes, metrics):
    sns.kdeplot(df_scores.iloc[fit_idx][col], ax=ax, label="Baseline Fit (0..159)", color="blue", fill=True, alpha=0.2)
    sns.kdeplot(df_scores.iloc[val_idx][col], ax=ax, label="Validation (160..199)", color="orange", fill=True, alpha=0.2)
    sns.kdeplot(df_scores.iloc[seq_idx][col], ax=ax, label="Sequential Eval (200..983)", color="red", fill=True, alpha=0.2)
    if "PCA" in title:
        ax.set_yscale("log")
    ax.set_title(f"Bearing 1: {title} Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("Score Value")
    ax.legend()

plt.tight_layout()
fig.savefig(fig_dir / "model_score_distribution_comparison.png", dpi=300)
plt.close(fig)
print("Saved model_score_distribution_comparison.png")

print("All Phase 7 model artifacts generated successfully!")
