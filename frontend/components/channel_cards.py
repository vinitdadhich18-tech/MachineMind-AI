"""
channel_cards.py - Compact horizontal channel condition strip renderer for Control-Room UI.
"""

from typing import Any, Dict, List
import textwrap
import streamlit as st


def render_channel_cards(channels: List[Dict[str, Any]], required_consecutive: int = 3):
    """
    Renders 4 compact horizontal vibration channel condition cards for CH1..CH4.
    Displays RMS, P2P, Crest Factor, and Spectral Centroid for each channel.
    Supports optional required_consecutive argument for backward compatibility.
    Uses textwrap.dedent to prevent Markdown code block raw HTML rendering bugs.
    """
    if not channels:
        st.warning("No channel telemetry available.")
        return

    cols = st.columns(4)

    for idx, channel in enumerate(channels):
        ch_num = channel.get("channel", idx + 1)
        rms_val = float(channel.get("rms", 0.0))
        p2p_val = float(channel.get("p2p", 0.0))
        crest_val = float(channel.get("crest_factor", 0.0))
        centroid_val = float(channel.get("spectral_centroid", 0.0))

        with cols[idx]:
            with st.container(border=True):
                header_html = textwrap.dedent(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid #1C2436; padding-bottom: 4px;">
                    <span style="font-size: 0.88rem; font-weight: 800; color: #F1F5F9; letter-spacing: 0.04em;">CH{ch_num} TELEMETRY</span>
                    <span class="pill-live"><span class="status-dot-live"></span>LIVE</span>
                </div>
                """).strip()
                st.markdown(header_html, unsafe_allow_html=True)

                st.metric("RMS Vibration", f"{rms_val:.5f} g")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("P2P", f"{p2p_val:.4f} g")
                with col_b:
                    st.metric("Crest Factor", f"{crest_val:.2f}")

                st.caption(f"Spectral Centroid: **{centroid_val:,.1f} Hz**")