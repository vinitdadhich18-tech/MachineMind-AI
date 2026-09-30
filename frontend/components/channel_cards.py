"""
channel_cards.py - Compact horizontal channel condition strip renderer for Control-Room UI.
"""

from typing import Any, Dict, List
import streamlit as st


def render_channel_cards(channels: List[Dict[str, Any]], required_consecutive: int = 3):
    """
    Renders compact horizontal 4-channel condition strip.
    """
    cols = st.columns(4)

    for idx, ch in enumerate(channels):
        ch_num = ch.get("channel", idx + 1)
        score = ch.get("anomaly_score", 0.0)
        thresh = ch.get("threshold", 0.0)
        snap_flag = ch.get("snapshot_flagged", False)
        count_val = ch.get("consecutive_flagged_count", 0)
        pers_conf = ch.get("persistence_confirmed", False)

        if pers_conf:
            st_text = "ANOMALY CONFIRMED"
            st_class = "pill-anomaly"
            dot_class = "status-dot-anomaly"
            border_color = "#EF4444"
        elif snap_flag:
            st_text = "WATCH"
            st_class = "pill-watch"
            dot_class = "status-dot-watch"
            border_color = "#F59E0B"
        else:
            st_text = "NORMAL"
            st_class = "pill-normal"
            dot_class = "status-dot-normal"
            border_color = "#1C2436"

        with cols[idx]:
            st.markdown(
                f"""
                <div style="background-color: #101520; border: 1px solid {border_color}; border-radius: 4px; padding: 10px 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em;">CH{ch_num}</span>
                        <span class="{st_class}"><span class="{dot_class}"></span>{st_text}</span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #F1F5F9; line-height: 1.1;">
                        {score:.4f}
                    </div>
                    <div style="font-size: 0.7rem; color: #64748B; margin-top: 4px; display: flex; justify-content: space-between;">
                        <span>Thresh: <strong style="color: #94A3B8;">{thresh:.4f}</strong></span>
                        <span>Count: <strong style="color: #00E5FF;">{count_val}/{required_consecutive}</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
