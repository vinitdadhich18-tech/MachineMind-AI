"""
4_Alerts.py - System Alerts & Notification Management Page.
"""

import streamlit as st
from frontend.services.api_client import list_alerts, update_alert_status, list_machines, ApiError
from frontend.components.status import render_footer, DISCLAIMER_TEXT

st.set_page_config(page_title="MachineMind AI — Alerts", page_icon="🚨", layout="wide")


def main():
    st.title("🚨 System Anomaly Alerts")
    st.caption(DISCLAIMER_TEXT)
    st.markdown("---")

    # Sidebar Filter Controls
    st.sidebar.markdown("### 🔍 Filter Alerts")

    try:
        machines = list_machines()
        machine_options = ["All Machines"] + [m["machine_id"] for m in machines]
    except Exception:
        machine_options = ["All Machines"]

    sel_machine = st.sidebar.selectbox("Filter by Machine", options=machine_options)
    sel_status = st.sidebar.selectbox("Filter by Status", options=["all", "open", "acknowledged", "resolved"])
    sel_severity = st.sidebar.selectbox("Filter by Severity", options=["all", "warning", "high"])

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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Filtered Alerts", str(len(alerts_list)))
    with col2:
        st.metric("⚠️ Open", str(open_c))
    with col3:
        st.metric("👀 Acknowledged", str(ack_c))
    with col4:
        st.metric("✅ Resolved", str(res_c))

    st.markdown("---")

    if not alerts_list:
        st.info("No alerts match the selected criteria. Alerts are created only when a vibration anomaly persists for 3 consecutive snapshots.")
    else:
        st.subheader("Alert Log & Actions")
        for alt_item in alerts_list:
            alert_id = alt_item["alert_id"]
            severity = alt_item.get("severity", "warning")
            sev_icon = "🔴" if severity == "high" else "🟠"
            status_curr = alt_item.get("status", "open")

            col_info, col_act = st.columns([4, 1])

            with col_info:
                st.markdown(
                    f"#### {sev_icon} Alert for Machine `{alt_item.get('machine_id')}` — `{severity.upper()}`  \n"
                    f"**Timestamp:** `{alt_item.get('timestamp')[:19]}` | **Status:** `{status_curr.upper()}` | **Channels:** `{alt_item.get('affected_channels')}`  \n"
                    f"{alt_item.get('message')}"
                )

            with col_act:
                if status_curr == "open":
                    if st.button("👀 Acknowledge", key=f"ack_{alert_id}"):
                        try:
                            update_alert_status(alert_id, "acknowledged")
                            st.success("Acknowledged!")
                            st.rerun()
                        except ApiError as err:
                            st.error(err.message)

                    if st.button("✅ Resolve", key=f"res_{alert_id}"):
                        try:
                            update_alert_status(alert_id, "resolved")
                            st.success("Resolved!")
                            st.rerun()
                        except ApiError as err:
                            st.error(err.message)
                elif status_curr == "acknowledged":
                    if st.button("✅ Resolve", key=f"res_{alert_id}"):
                        try:
                            update_alert_status(alert_id, "resolved")
                            st.success("Resolved!")
                            st.rerun()
                        except ApiError as err:
                            st.error(err.message)
                else:
                    st.caption("✅ Resolved")

            st.markdown("---")

    render_footer()


if __name__ == "__main__":
    main()
