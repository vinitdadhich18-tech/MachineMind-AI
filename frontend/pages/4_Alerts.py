"""
4_Alerts.py - Operations Incident Console & Alert Management.
"""

import streamlit as st
from frontend.services.api_client import list_alerts, update_alert_status, list_machines, get_health, ApiError
from frontend.components.theme import inject_theme
from frontend.components.status import (
    render_global_header,
    render_sidebar_shell,
    render_status_badge,
    render_footer
)
from frontend.utils.formatting import format_timestamp

st.set_page_config(page_title="MachineMind AI — Alerts", page_icon="🚨", layout="wide")


def main():
    inject_theme()

    try:
        health = get_health()
        render_sidebar_shell(health)
    except ApiError:
        pass

    # Sidebar Filter Controls
    st.sidebar.markdown("### 🔍 INCIDENT FILTERS")

    try:
        machines = list_machines()
        machine_options = ["All Machines"] + [m["machine_id"] for m in machines]
    except Exception:
        machine_options = ["All Machines"]

    sel_machine = st.sidebar.selectbox("Equipment Unit", options=machine_options)
    sel_status = st.sidebar.selectbox("Alert Status", options=["all", "open", "acknowledged", "resolved"])
    sel_severity = st.sidebar.selectbox("Severity Level", options=["all", "warning", "high"])

    filter_m_id = None if sel_machine == "All Machines" else sel_machine
    filter_status = None if sel_status == "all" else sel_status
    filter_severity = None if sel_severity == "all" else sel_severity

    try:
        data = list_alerts(status=filter_status, severity=filter_severity, limit=100)
    except ApiError as err:
        st.error(f"Failed to load alerts: {err.message}")
        st.stop()

    alerts_list = data.get("alerts", [])
    if filter_m_id:
        alerts_list = [a for a in alerts_list if a.get("machine_id") == filter_m_id]

    open_c = sum(1 for a in alerts_list if a.get("status") == "open")
    ack_c = sum(1 for a in alerts_list if a.get("status") == "acknowledged")
    res_c = sum(1 for a in alerts_list if a.get("status") == "resolved")

    render_global_header("OPERATIONS INCIDENT CONSOLE")

    # Stat Counter Strip
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="ctrl-panel" style="padding:10px;"><div style="font-size:0.7rem; color:#94A3B8; font-weight:700;">TOTAL FILTERED INCIDENTS</div><div style="font-size:1.4rem; font-weight:800; color:#F1F5F9;">{len(alerts_list)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="ctrl-panel" style="padding:10px;"><div style="font-size:0.7rem; color:#EF4444; font-weight:700;">OPEN ALERTS</div><div style="font-size:1.4rem; font-weight:800; color:#EF4444;">{open_c}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="ctrl-panel" style="padding:10px;"><div style="font-size:0.7rem; color:#F59E0B; font-weight:700;">ACKNOWLEDGED</div><div style="font-size:1.4rem; font-weight:800; color:#F59E0B;">{ack_c}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="ctrl-panel" style="padding:10px;"><div style="font-size:0.7rem; color:#10B981; font-weight:700;">RESOLVED</div><div style="font-size:1.4rem; font-weight:800; color:#10B981;">{res_c}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin: 10px 0 8px 0;'>ACTIVE SYSTEM INCIDENTS & ACTION LOG</div>", unsafe_allow_html=True)

    if not alerts_list:
        st.info("No incident alerts match the selected criteria.")
    else:
        for alt_item in alerts_list:
            alert_id = alt_item["alert_id"]
            severity = alt_item.get("severity", "warning")
            status_curr = alt_item.get("status", "open")
            dot_cls = "status-dot-anomaly" if severity == "high" else "status-dot-watch"
            pill_cls = "pill-anomaly" if severity == "high" else "pill-watch"

            col_info, col_act = st.columns([4, 1])

            with col_info:
                st.markdown(
                    f"""
                    <div class="ctrl-panel" style="margin-bottom: 6px; padding: 10px 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span><span class="{dot_cls}"></span><strong style="color:#F1F5F9;">{alt_item.get('machine_id')}</strong> — Channels: <strong style="color:#00E5FF;">{alt_item.get('affected_channels')}</strong></span>
                            <div>
                                <span class="{pill_cls}">{severity.upper()}</span>
                                <span class="pill-nodata" style="margin-left: 6px;">{status_curr.upper()}</span>
                            </div>
                        </div>
                        <div style="font-size: 0.75rem; color: #94A3B8; margin-bottom: 4px;">
                            🕒 Source Telemetry Time: {format_timestamp(alt_item.get('timestamp'))} | Alert ID: <code>{alert_id}</code>
                        </div>
                        <div style="font-size: 0.8rem; color: #E2E8F0;">
                            {alt_item.get('message')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_act:
                if status_curr == "open":
                    if st.button("ACKNOWLEDGE", key=f"ack_{alert_id}"):
                        try:
                            update_alert_status(alert_id, "acknowledged")
                            st.rerun()
                        except ApiError as err:
                            st.error(err.message)
                    if st.button("RESOLVE", key=f"res_{alert_id}"):
                        try:
                            update_alert_status(alert_id, "resolved")
                            st.rerun()
                        except ApiError as err:
                            st.error(err.message)
                elif status_curr == "acknowledged":
                    if st.button("RESOLVE", key=f"res_{alert_id}"):
                        try:
                            update_alert_status(alert_id, "resolved")
                            st.rerun()
                        except ApiError as err:
                            st.error(err.message)
                else:
                    st.markdown("<div style='color:#10B981; font-weight:700; font-size:0.75rem; padding-top:8px;'>✓ RESOLVED</div>", unsafe_allow_html=True)

    render_footer()


if __name__ == "__main__":
    main()
