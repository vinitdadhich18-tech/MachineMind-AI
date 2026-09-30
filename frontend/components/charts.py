"""
charts.py - Visualization builders for anomaly scores, vibration waveforms, and prediction history.
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st


def render_anomaly_score_chart(channels: List[Dict[str, Any]]):
    """
    Renders grouped bar chart of anomaly score vs threshold per channel using Altair.
    """
    chart_data = []
    for ch in channels:
        name = ch.get("name", f"Channel {ch['channel']}")
        score = ch.get("anomaly_score", 0.0)
        thresh = ch.get("threshold", 0.0)
        chart_data.append({"Channel": name, "Metric": "Anomaly Score", "Value": score})
        chart_data.append({"Channel": name, "Metric": "P99 Threshold", "Value": thresh})

    df = pd.DataFrame(chart_data)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title=None, axis=alt.Axis(labels=True)),
            y=alt.Y("Value:Q", title="Value"),
            color=alt.Color("Metric:N", scale=alt.Scale(domain=["Anomaly Score", "P99 Threshold"], range=["#e74c3c", "#34495e"])),
            column=alt.Column("Channel:N", title=None)
        )
        .properties(width=140, height=220)
    )

    st.altair_chart(chart, use_container_width=False)


def render_waveform_preview(df: pd.DataFrame, max_points: int = 2000):
    """
    Renders downsampled 4-channel vibration signal waveform plot (<= 2000 points per channel).
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

    chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("Sample Index:Q", title="Sample Index (20,480 total samples)"),
            y=alt.Y("Vibration (g):Q", title="Amplitude (g)"),
            color=alt.Color("Channel:N", title="Channel"),
            tooltip=["Sample Index", "Channel", "Vibration (g)"]
        )
        .properties(height=300)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def render_history_chart(predictions: List[Dict[str, Any]]):
    """
    Renders time-series line chart of per-channel anomaly scores over time.
    """
    if not predictions:
        st.info("No historical prediction data available.")
        return

    records = []
    for pred in predictions:
        ts = pred.get("timestamp", "")[:19]
        for ch in pred.get("channels", []):
            records.append({
                "Timestamp": ts,
                "Channel": ch.get("name", f"Channel {ch['channel']}"),
                "Score": ch.get("anomaly_score", 0.0),
                "Threshold": ch.get("threshold", 0.0),
                "Flagged": ch.get("snapshot_flagged", False)
            })

    df = pd.DataFrame(records)

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Timestamp:O", title="Snapshot Time (Oldest -> Newest)"),
            y=alt.Y("Score:Q", title="Anomaly Score"),
            color=alt.Color("Channel:N", title="Channel"),
            tooltip=["Timestamp", "Channel", "Score", "Threshold", "Flagged"]
        )
        .properties(height=350)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)
