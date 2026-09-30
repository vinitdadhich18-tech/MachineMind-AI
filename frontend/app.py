"""
app.py - MachineMind AI Command Center Overview.
"""

import streamlit as st
from frontend.services.api_client import get_health, list_machines, list_alerts, get_predictions, ApiError
from frontend.components.theme import inject_theme
from frontend.components.status import (
    render_global_header,
    render_sidebar_shell,
    render_status_badge,
    render_footer
)
from frontend.components.channel_cards import render_channel_cards
from frontend.components.charts import render_history_chart
from frontend.utils.formatting import format_timestamp

st.set_page_config(
    page_title="MachineMind AI — Command Center",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    inject_theme()

    # Fetch backend health
    try:
        health = get_health()
    except ApiError as err:
        st.error(f"Backend Server Unreachable: {err.message}")
        st.info("Please start the Flask backend server on http://localhost:5000.")
        st.stop()

    render_sidebar_shell(health)

    # Fetch machines & alerts
    try:
        machines = list_machines()
        alerts_data = list_alerts(limit=50)
    except ApiError as err:
        st.error(f"Failed to load system data: {err.message}")
        st.stop()

    # Selected / Active Machine Determination
    if not machines:
        render_global_header("NO MACHINERY REGISTERED")
        st.warning("No machines registered yet. Select MACHINES in the sidebar to register equipment.")
        render_footer()
        return

    machine_ids = [m["machine_id"] for m in machines]
    selected_id = st.session_state.get("selected_machine_id")
    if selected_id not in machine_ids:
        selected_id = machine_ids[0]
        st.session_state["selected_machine_id"] = selected_id

    # Sidebar Machine Picker
    selected_id = st.sidebar.selectbox(
        "MONITORED EQUIPMENT",
        options=machine_ids,
        index=machine_ids.index(selected_id),
        key="app_machine_select"
    )
    st.session_state["selected_machine_id"] = selected_id

    # Active Machine Object
    active_m = next((m for m in machines if m["machine_id"] == selected_id), machines[0])
    m_name = active_m.get("name", selected_id)
    m_status = active_m.get("status", "no_data")
    m_updated = format_timestamp(active_m.get("updated_at", ""))
    latest_pred = active_m.get("latest_prediction") or {}
    overall = latest_pred.get("overall", {})

    render_global_header(active_machine_id=m_name)

    # Fetch prediction history for active machine
    try:
        pred_data = get_predictions(selected_id, limit=50)
        predictions = pred_data.get("predictions", [])
    except ApiError:
        predictions = []

    # Persistence count and channels determination (using latest_prediction or predictions history fallback)
    channels_data = latest_pred.get("channels", [])
    if not channels_data and predictions:
        channels_data = predictions[0].get("channels", [])

    dataset_ts = latest_pred.get("timestamp") or (predictions[0].get("timestamp") if predictions else None)
    formatted_dataset_ts = format_timestamp(dataset_ts) if dataset_ts else "N/A"

    max_count = max([c.get("consecutive_flagged_count", 0) for c in channels_data], default=0) if channels_data else 0
    pers_text = f"{max_count} / 3 Snapshots"

    # 1. TOP MACHINE PANEL
    st_dot = "status-dot-normal" if m_status == "normal" else ("status-dot-watch" if m_status == "watch" else ("status-dot-anomaly" if m_status == "anomaly_detected" else "status-dot-gray"))
    st_pill = "pill-normal" if m_status == "normal" else ("pill-watch" if m_status == "watch" else ("pill-anomaly" if m_status == "anomaly_detected" else "pill-nodata"))
    st_label = "NORMAL" if m_status == "normal" else ("WATCH — AWAITING PERSISTENCE" if m_status == "watch" else ("VIBRATION ANOMALY CONFIRMED" if m_status == "anomaly_detected" else "NO DATA"))

    st.markdown(
        f"""
        <div class="ctrl-panel-highlight">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <span style="font-size: 1.25rem; font-weight: 800; color: #F1F5F9;">{m_name}</span>
                    <span style="font-size: 0.8rem; color: #64748B; margin-left: 10px;">ID: <code style="color: #94A3B8;">{selected_id}</code></span>
                </div>
                <div>
                    <span class="{st_pill}"><span class="{st_dot}"></span>{st_label}</span>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem; color: #94A3B8; border-top: 1px solid #1C2436; padding-top: 8px; margin-top: 6px;">
                <span>Last Analysis Run: <strong style="color: #F1F5F9;">{m_updated}</strong></span>
                <span>Dataset Snapshot Time: <strong style="color: #F1F5F9;">{formatted_dataset_ts}</strong></span>
                <span>Persistence State: <strong style="color: #00E5FF;">{pers_text}</strong></span>
                <span>Model Target: <strong style="color: #F1F5F9;">NASA IMS Set 2</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. MAIN VISUALIZATION: Large Anomaly Score Trend
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin-bottom: 6px;'>TELEMETRY ANOMALY SCORE TREND (CHANNELS 1–4)</div>", unsafe_allow_html=True)
    if predictions:
        render_history_chart(list(reversed(predictions)))
    else:
        st.info("No prediction telemetry history recorded yet. Upload a snapshot on ANALYZE page to begin.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. CHANNEL CONDITION STRIP
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin-bottom: 6px;'>CHANNEL CONDITION STRIP</div>", unsafe_allow_html=True)
    if channels_data:
        render_channel_cards(channels_data)
    else:
        # Default placeholder strip
        cols = st.columns(4)
        for i in range(1, 5):
            with cols[i-1]:
                st.markdown(f'<div class="channel-strip-item"><span style="font-size:0.75rem; color:#64748B;">CH{i} NO DATA</span></div>', unsafe_allow_html=True)

    # 4. BOTTOM MODULES: RECENT INCIDENTS & MACHINE SUMMARY
    st.markdown("<br>", unsafe_allow_html=True)
    col_events, col_summary = st.columns([3, 2])

    with col_events:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin-bottom: 8px;'>RECENT INCIDENT EVENTS</div>", unsafe_allow_html=True)
        recent_alerts = alerts_data.get("alerts", [])[:5]
        if not recent_alerts:
            st.markdown('<div class="ctrl-panel" style="color:#10B981; font-size:0.8rem;">● No active anomaly events recorded. Systems operating within baseline boundaries.</div>', unsafe_allow_html=True)
        else:
            for alt_item in recent_alerts:
                sev = alt_item.get("severity", "warning")
                dot_c = "status-dot-anomaly" if sev == "high" else "status-dot-watch"
                st.markdown(
                    f"""
                    <div style="background:#101520; border:1px solid #1C2436; border-radius:4px; padding:8px 12px; margin-bottom:6px; font-size:0.78rem;">
                        <div style="display:flex; justify-content:space-between;">
                            <span><span class="{dot_c}"></span><strong>{alt_item.get('machine_id')}</strong> — {alt_item.get('affected_channels')}</span>
                            <span style="color:#64748B;">Source Telemetry Time: {format_timestamp(alt_item.get('timestamp'))}</span>
                        </div>
                        <div style="color:#94A3B8; margin-top:2px;">{alt_item.get('message')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with col_summary:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin-bottom: 8px;'>EQUIPMENT CONSOLE SUMMARY</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="ctrl-panel" style="font-size: 0.78rem; color: #94A3B8;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span>Total Registered Equipment:</span>
                    <strong style="color: #F1F5F9;">{len(machines)}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span>Normal State Units:</span>
                    <strong style="color: #10B981;">{sum(1 for m in machines if m.get('status') == 'normal')}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span>Watch State Units:</span>
                    <strong style="color: #F59E0B;">{sum(1 for m in machines if m.get('status') == 'watch')}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span>Anomaly Confirmed Units:</span>
                    <strong style="color: #EF4444;">{sum(1 for m in machines if m.get('status') == 'anomaly_detected')}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span>Unanalyzed / No Data Units:</span>
                    <strong style="color: #64748B;">{sum(1 for m in machines if m.get('status') not in ('normal', 'watch', 'anomaly_detected'))}</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Active System Alerts:</span>
                    <strong style="color: #EF4444;">{sum(1 for a in alerts_data.get('alerts', []) if a.get('status') == 'open')}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    render_footer()


if __name__ == "__main__":
    main()
