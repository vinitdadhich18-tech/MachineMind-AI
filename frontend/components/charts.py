"""
charts.py - Industrial control-room visual comparison charts for vibration features across channels.
Provides full backward compatibility for legacy pages (anomaly scores, waveforms, history charts).
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st


def render_rms_comparison_chart(channels: List[Dict[str, Any]]):
    """
    Renders bar chart comparing RMS vibration power (g) across Channels 1–4.
    """
    if not channels:
        st.info("No channel data available for visualization.")
        return

    data = [
        {"Channel": f"CH{ch.get('channel', i+1)}", "RMS (g)": float(ch.get("rms", 0.0))}
        for i, ch in enumerate(channels)
    ]
    df = pd.DataFrame(data)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#00E5FF", width=36)
        .encode(
            x=alt.X("Channel:N", title=None, axis=alt.Axis(labelColor="#F1F5F9", labelFontSize=11)),
            y=alt.Y("RMS (g):Q", title="RMS Amplitude (g)", axis=alt.Axis(gridColor="#1C2436", labelColor="#94A3B8", titleColor="#94A3B8")),
            tooltip=["Channel", "RMS (g)"]
        )
        .properties(height=220)
    )

    st.altair_chart(chart, use_container_width=True)


def render_crest_factor_chart(channels: List[Dict[str, Any]]):
    """
    Renders bar chart comparing Crest Factor (Peakiness) across Channels 1–4.
    """
    if not channels:
        return

    data = [
        {"Channel": f"CH{ch.get('channel', i+1)}", "Crest Factor": float(ch.get("crest_factor", 0.0))}
        for i, ch in enumerate(channels)
    ]
    df = pd.DataFrame(data)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#3B82F6", width=36)
        .encode(
            x=alt.X("Channel:N", title=None, axis=alt.Axis(labelColor="#F1F5F9", labelFontSize=11)),
            y=alt.Y("Crest Factor:Q", title="Peakiness Ratio", axis=alt.Axis(gridColor="#1C2436", labelColor="#94A3B8", titleColor="#94A3B8")),
            tooltip=["Channel", "Crest Factor"]
        )
        .properties(height=220)
    )

    st.altair_chart(chart, use_container_width=True)


def render_spectral_centroid_chart(channels: List[Dict[str, Any]]):
    """
    Renders bar chart comparing Spectral Centroid (Center Frequency in Hz) across Channels 1–4.
    """
    if not channels:
        return

    data = [
        {"Channel": f"CH{ch.get('channel', i+1)}", "Spectral Centroid (Hz)": float(ch.get("spectral_centroid", 0.0))}
        for i, ch in enumerate(channels)
    ]
    df = pd.DataFrame(data)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#14B8A6", width=36)
        .encode(
            x=alt.X("Channel:N", title=None, axis=alt.Axis(labelColor="#F1F5F9", labelFontSize=11)),
            y=alt.Y("Spectral Centroid (Hz):Q", title="Frequency (Hz)", axis=alt.Axis(gridColor="#1C2436", labelColor="#94A3B8", titleColor="#94A3B8")),
            tooltip=["Channel", "Spectral Centroid (Hz)"]
        )
        .properties(height=220)
    )

    st.altair_chart(chart, use_container_width=True)


# ============================================================
# Backward Compatibility Wrappers for Legacy Pages
# ============================================================

def render_anomaly_score_chart(channels: List[Dict[str, Any]]):
    """
    Renders bar chart comparison of per-channel anomaly score vs P99 baseline threshold.
    """
    chart_data = []
    for ch in channels:
        name = ch.get("name", f"Channel {ch.get('channel', 1)}")
        score = float(ch.get("anomaly_score", 0.0))
        thresh = float(ch.get("threshold", 0.0))
        chart_data.append({"Channel": name, "Metric": "Anomaly Score", "Value": score})
        chart_data.append({"Channel": name, "Metric": "P99 Threshold", "Value": thresh})

    df = pd.DataFrame(chart_data)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("Metric:N", title=None, axis=alt.Axis(labels=True, labelColor="#94A3B8")),
            y=alt.Y("Value:Q", title="Score Value", axis=alt.Axis(gridColor="#1E2638", labelColor="#94A3B8", titleColor="#94A3B8")),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(domain=["Anomaly Score", "P99 Threshold"], range=["#00E5FF", "#64748B"]),
                legend=alt.Legend(title=None, orient="top", labelColor="#94A3B8")
            ),
            column=alt.Column("Channel:N", title=None, header=alt.Header(labelColor="#F8FAFC", labelFontSize=12))
        )
        .properties(width=160, height=220)
    )

    st.altair_chart(chart, use_container_width=True)


def render_waveform_preview(df: pd.DataFrame, max_points: int = 2000):
    """
    Renders downsampled 4-channel vibration signal waveform plot.
    """
    num_samples = len(df)
    if num_samples > max_points:
        stride = num_samples // max_points
        downsampled = df.iloc[::stride].copy()
    else:
        downsampled = df.copy()

    downsampled["Sample Index"] = downsampled.index

    # Melt dataframe for Altair line chart
    melted = downsampled.melt(id_vars=["Sample Index"], var_name="Channel", value_name="Vibration (g)")
    colors = ["#00E5FF", "#3B82F6", "#F59E0B", "#EC4899"]

    chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("Sample Index:Q", title="Sample Index (20,480 total samples)", axis=alt.Axis(gridColor="#1E2638", labelColor="#94A3B8", titleColor="#94A3B8")),
            y=alt.Y("Vibration (g):Q", title="Amplitude (g)", axis=alt.Axis(gridColor="#1E2638", labelColor="#94A3B8", titleColor="#94A3B8")),
            color=alt.Color("Channel:N", title="Channel", scale=alt.Scale(range=colors), legend=alt.Legend(orient="top", labelColor="#94A3B8", titleColor="#94A3B8")),
            tooltip=["Sample Index", "Channel", "Vibration (g)"]
        )
        .properties(height=300)
    )

    st.altair_chart(chart, use_container_width=True)


def render_history_chart(predictions: List[Dict[str, Any]]):
    """
    Renders full-width time-series line chart of per-channel anomaly scores over time.
    """
    if not predictions:
        st.info("No historical prediction data available.")
        return

    records = []
    for pred in predictions:
        ts = str(pred.get("timestamp", ""))[:19].replace("T", " ")
        for ch in pred.get("channels", []):
            records.append({
                "Timestamp": ts,
                "Channel": ch.get("name", f"Channel {ch.get('channel', 1)}"),
                "Score": float(ch.get("anomaly_score", 0.0)),
                "Threshold": float(ch.get("threshold", 0.0)),
                "Flagged": "Flagged" if ch.get("snapshot_flagged") else "Normal"
            })

    df = pd.DataFrame(records)
    colors = ["#00E5FF", "#3B82F6", "#F59E0B", "#EC4899"]

    chart = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("Timestamp:O", title="Dataset Snapshot Time (Chronological)", axis=alt.Axis(gridColor="#1E2638", labelColor="#94A3B8", titleColor="#94A3B8")),
            y=alt.Y("Score:Q", title="Anomaly Score", axis=alt.Axis(gridColor="#1E2638", labelColor="#94A3B8", titleColor="#94A3B8")),
            color=alt.Color("Channel:N", title="Channel", scale=alt.Scale(range=colors), legend=alt.Legend(orient="top", labelColor="#94A3B8", titleColor="#94A3B8")),
            tooltip=["Timestamp", "Channel", "Score", "Threshold", "Flagged"]
        )
        .properties(height=380)
    )

    st.altair_chart(chart, use_container_width=True)
