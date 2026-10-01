"""
app.py - MachineMind AI Live MQTT Industrial Control-Room Command Center.

Displays live, deterministic vibration features generated from incoming MQTT telemetry
replayed from the NASA IMS Set 2 Bearing Dataset.
"""

import json
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from frontend.components.theme import inject_theme
from frontend.components.status import (
    render_global_header,
    render_sidebar_shell,
    render_footer
)
from frontend.components.channel_cards import render_channel_cards
from frontend.components.charts import (
    render_rms_comparison_chart,
    render_crest_factor_chart,
    render_spectral_centroid_chart
)

# Live telemetry state file written by mqtt_dashboard_bridge.py
DATA_FILE = Path(__file__).resolve().parent / "data" / "live_telemetry.json"


# Page Configuration
st.set_page_config(
    page_title="MachineMind AI — Industrial Command Center",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_live_telemetry() -> Optional[Dict[str, Any]]:
    """Loads latest telemetry JSON state produced by MQTT dashboard bridge."""
    if not DATA_FILE.exists():
        return None
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    inject_theme()

    # Sidebar Auto-Refresh Toggle (Refresh every 2000 ms)
    with st.sidebar:
        auto_refresh = st.checkbox("Auto-Refresh Telemetry (2s)", value=True)
        if auto_refresh:
            st_autorefresh(interval=2000, key="telemetry_autorefresh")

    # Load latest MQTT Telemetry State
    telemetry = load_live_telemetry()

    # Render Sidebar Shell
    render_sidebar_shell(telemetry)

    # Render Top Application Header
    machine_id = telemetry.get("machine_id", "ims_set2_rig") if telemetry else "ims_set2_rig"
    render_global_header(active_machine_id=machine_id, telemetry_info=telemetry)

    # --------------------------------------------------------
    # IF NO TELEMETRY DATA AVAILABLE YET
    # --------------------------------------------------------
    if telemetry is None:
        awaiting_html = textwrap.dedent("""
        <div class="ctrl-panel-highlight">
            <div style="font-size: 1.1rem; font-weight: 800; color: #F1F5F9; margin-bottom: 6px;">
                ⚠️ AWAITING LIVE MQTT TELEMETRY
            </div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px;">
                No live telemetry state found at <code>frontend/data/live_telemetry.json</code>.
            </div>
            <div style="font-size: 0.78rem; color: #64748B;">
                To view live telemetry on this dashboard:
                <ol style="margin-top: 6px; padding-left: 20px;">
                    <li>Start the MQTT Dashboard Bridge: <code>$env:PYTHONPATH="ml-service"; python ml-service/src/mqtt_dashboard_bridge.py</code></li>
                    <li>Replay NASA IMS snapshots: <code>$env:PYTHONPATH="ml-service"; python ml-service/src/simulator.py --start-idx 0 --end-idx 10 --interval 2</code></li>
                </ol>
            </div>
        </div>
        """).strip()
        st.markdown(awaiting_html, unsafe_allow_html=True)
        render_footer()
        return

    # Extract Telemetry Fields
    snapshot_sequence = telemetry.get("snapshot_sequence", "N/A")
    orig_ts = telemetry.get("original_timestamp", "N/A")
    ingest_ts = telemetry.get("ingest_timestamp", "N/A")
    sampling_rate = float(telemetry.get("sampling_rate_hz", 20480.0))
    sample_count = int(telemetry.get("sample_count", 20480))
    source_tag = telemetry.get("source", "replayed_nasa_ims")
    feature_count = int(telemetry.get("feature_count", 36))
    channels = telemetry.get("channels", [])
    ingestion = telemetry.get("ingestion", {})

    processing_latency = float(ingestion.get("processing_latency_ms", 0.0))
    is_duplicate = ingestion.get("is_duplicate", False)
    is_out_of_order = ingestion.get("is_out_of_order", False)
    is_sequence_gap = ingestion.get("is_sequence_gap", False)

    # --------------------------------------------------------
    # 1. MACHINE STATUS BAR & TIMING PANEL
    # --------------------------------------------------------
    status_panel_html = textwrap.dedent(f"""
    <div class="ctrl-panel-highlight">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div>
                <span style="font-size: 1.35rem; font-weight: 900; color: #F1F5F9;">{machine_id}</span>
                <span style="font-size: 0.82rem; color: #64748B; margin-left: 12px;">Target Dataset: <strong style="color: #00E5FF;">NASA IMS Bearing Set 2</strong></span>
            </div>
            <div>
                <span class="pill-live"><span class="status-dot-live"></span>LIVE TELEMETRY</span>
            </div>
        </div>
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            font-size: 0.78rem;
            color: #94A3B8;
            border-top: 1px solid #1C2436;
            padding-top: 8px;
            margin-top: 6px;
        ">
            <span>Snapshot Index: <strong style="color: #F1F5F9;">#{snapshot_sequence}</strong></span>
            <span>Source Telemetry Time: <strong style="color: #F1F5F9;">{orig_ts}</strong></span>
            <span>System Ingestion Time: <strong style="color: #F1F5F9;">{ingest_ts}</strong></span>
            <span>Telemetry Origin: <code style="color: #00E5FF;">{source_tag}</code></span>
        </div>
    </div>
    """).strip()
    st.markdown(status_panel_html, unsafe_allow_html=True)

    # --------------------------------------------------------
    # 2. KEY PIPELINE PERFORMANCE METRICS
    # --------------------------------------------------------
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Sampling Rate", f"{sampling_rate:,.0f} Hz")
    with m2:
        st.metric("Samples / Snapshot", f"{sample_count:,}")
    with m3:
        st.metric("Canonical Features", f"{feature_count}")
    with m4:
        st.metric("Processing Latency", f"{processing_latency:.2f} ms")
    with m5:
        st.metric("MQTT QoS Level", "QoS 1")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 3. LIVE CHANNEL TELEMETRY (4 CARDS)
    # --------------------------------------------------------
    sec1_html = textwrap.dedent("""
    <div class="section-hdr">
        <span>⚡ LIVE CHANNEL TELEMETRY (CHANNELS 1–4)</span>
    </div>
    """).strip()
    st.markdown(sec1_html, unsafe_allow_html=True)

    render_channel_cards(channels)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 4. VIBRATION FEATURE COMPARISON CHARTS
    # --------------------------------------------------------
    sec2_html = textwrap.dedent("""
    <div class="section-hdr">
        <span>📊 VIBRATION FEATURE OVERVIEW</span>
    </div>
    """).strip()
    st.markdown(sec2_html, unsafe_allow_html=True)

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        st.markdown("<div style='font-size:0.75rem; color:#94A3B8; font-weight:700; margin-bottom:4px;'>RMS AMPLITUDE (g)</div>", unsafe_allow_html=True)
        render_rms_comparison_chart(channels)
    with chart_col2:
        st.markdown("<div style='font-size:0.75rem; color:#94A3B8; font-weight:700; margin-bottom:4px;'>CREST FACTOR (PEAKINESS)</div>", unsafe_allow_html=True)
        render_crest_factor_chart(channels)
    with chart_col3:
        st.markdown("<div style='font-size:0.75rem; color:#94A3B8; font-weight:700; margin-bottom:4px;'>SPECTRAL CENTROID (Hz)</div>", unsafe_allow_html=True)
        render_spectral_centroid_chart(channels)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 5. CANONICAL FEATURE MATRIX TABLE
    # --------------------------------------------------------
    with st.expander("📋 CANONICAL FEATURE MATRIX (36 FEATURES)", expanded=False):
        matrix_rows = []
        for ch in channels:
            ch_num = ch.get("channel", "")
            matrix_rows.append({
                "Channel": f"CH{ch_num}",
                "Mean": round(float(ch.get("mean", 0.0)), 6),
                "Std": round(float(ch.get("std", 0.0)), 6),
                "RMS": round(float(ch.get("rms", 0.0)), 6),
                "P2P": round(float(ch.get("p2p", 0.0)), 4),
                "Skewness": round(float(ch.get("skewness", 0.0)), 4),
                "Kurtosis": round(float(ch.get("kurtosis", 0.0)), 4),
                "Crest Factor": round(float(ch.get("crest_factor", 0.0)), 2),
                "Spectral Energy": round(float(ch.get("spectral_energy", 0.0)), 6),
                "Spectral Centroid (Hz)": round(float(ch.get("spectral_centroid", 0.0)), 1)
            })
        df_matrix = pd.DataFrame(matrix_rows)
        st.dataframe(df_matrix, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 6. MQTT INGESTION INTEGRITY & PIPELINE HEALTH
    # --------------------------------------------------------
    sec3_html = textwrap.dedent("""
    <div class="section-hdr">
        <span>🛡️ MQTT INGESTION INTEGRITY & PIPELINE HEALTH</span>
    </div>
    """).strip()
    st.markdown(sec3_html, unsafe_allow_html=True)

    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    with h_col1:
        st.metric("Duplicate Flag", "YES" if is_duplicate else "NO")
    with h_col2:
        st.metric("Out-of-Order Flag", "YES" if is_out_of_order else "NO")
    with h_col3:
        st.metric("Sequence Gap Flag", "YES" if is_sequence_gap else "NO")
    with h_col4:
        st.metric("Consumer Latency", f"{processing_latency:.2f} ms")

    # --------------------------------------------------------
    # 7. SYSTEM ARCHITECTURE & RESEARCH BOUNDARY NOTE
    # --------------------------------------------------------
    with st.expander("🌐 SYSTEM ARCHITECTURE & DATA FLOW", expanded=False):
        st.markdown(textwrap.dedent("""
        <div style="background: #0B0E14; border: 1px solid #1C2436; border-radius: 4px; padding: 14px; font-family: monospace; font-size: 0.8rem; color: #00E5FF; line-height: 1.8;">
            NASA IMS Bearing Dataset (Set 2)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            Replay Simulator (simulator.py)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓ MQTT / TLS (port 8883, QoS 1)<br>
            HiveMQ Cloud Broker<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            MQTT Telemetry Consumer (ingestion_consumer.py)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            Canonical Feature Pipeline (canonical_feature_pipeline.py)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            MQTT Dashboard Bridge (mqtt_dashboard_bridge.py)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            Live Telemetry State (frontend/data/live_telemetry.json)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            Streamlit Command Center (frontend/app.py)
        </div>
        """).strip(), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "Research Prototype Boundary: This live telemetry command center displays deterministic vibration features "
            "generated from incoming MQTT telemetry. It does not claim physical machine failure, remaining useful life (RUL), "
            "or anomaly classification unless an explicitly connected model produces that result."
        )

    # Footer
    render_footer()


if __name__ == "__main__":
    main()