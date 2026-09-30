"""
channel_cards.py - Render helper for 4-channel analysis cards.
"""

from typing import Any, Dict, List
import streamlit as st
from frontend.components.status import render_status_badge


def render_channel_cards(channels: List[Dict[str, Any]], required_consecutive: int = 3):
    """
    Renders 4 columns (Channel 1 to 4) displaying score, threshold, flag status, and consecutive count.
    """
    cols = st.columns(4)

    for idx, ch in enumerate(channels):
        ch_num = ch.get("channel", idx + 1)
        ch_name = ch.get("name", f"Channel {ch_num}")
        score = ch.get("anomaly_score", 0.0)
        thresh = ch.get("threshold", 0.0)
        snap_flag = ch.get("snapshot_flagged", False)
        count_val = ch.get("consecutive_flagged_count", 0)
        state = ch.get("state", "normal")

        with cols[idx]:
            st.markdown(f"#### {ch_name}")
            delta_text = f"Thresh: {thresh:.4f} ({'+' if score > thresh else ''}{(score - thresh):.4f})"
            st.metric(
                label="Anomaly Score",
                value=f"{score:.4f}",
                delta=delta_text,
                delta_color="inverse" if snap_flag else "normal"
            )

            render_status_badge(state, short=True)

            st.caption(f"Consecutive flagged: **{count_val} / {required_consecutive}**")

            if snap_flag:
                st.caption(f"⚠️ *{ch_name} anomaly score exceeded threshold.*")
            else:
                st.caption(f"✅ *{ch_name} score within normal bounds.*")
