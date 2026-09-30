"""
app.py - MachineMind AI Streamlit Dashboard Entrypoint (Overview Page).
"""

import streamlit as st
from frontend.services.api_client import get_health, list_machines, list_alerts, ApiError
from frontend.components.status import render_footer, render_status_badge, DISCLAIMER_TEXT
from frontend.components.kpi import render_kpi_cards

st.set_page_config(
    page_title="MachineMind AI — Overview",
    page_icon="⚙️",
    layout="wide"
)


def main():
    st.title("⚙️ MachineMind AI — Vibration Anomaly Monitoring")
    st.markdown("### Unsupervised Research Prototype (NASA IMS Bearing Dataset)")
    st.caption(DISCLAIMER_TEXT)
    st.markdown("---")

    # 1. Fetch Backend Health
    try:
        health = get_health()
    except ApiError as err:
        st.error(f"🔴 **Backend Unreachable:** {err.message}")
        st.info("💡 Please start the Flask backend server on `http://localhost:5000` (`python app.py` inside `backend/`).")
        st.stop()

    # System status badges in header
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.markdown(f"**Backend Service:** `{'🟢 OK' if health.get('status') == 'ok' else '🟡 DEGRADED'}`")
    with col_h2:
        st.markdown(f"**MongoDB Database:** `{'🟢 CONNECTED' if health.get('database') == 'connected' else '🔴 UNAVAILABLE'}`")
    with col_h3:
        model_info = health.get("model", {})
        st.markdown(f"**ML Model Status:** `{'🟢 LOADED' if model_info.get('loaded') else '🔴 NOT LOADED'}` (v{model_info.get('version', '1')})")

    st.markdown("---")

    # 2. Fetch Machines & Alerts for KPIs
    try:
        machines = list_machines()
        alerts_data = list_alerts(limit=100)
    except ApiError as err:
        st.error(f"Failed to fetch data: {err.message}")
        st.stop()

    machines_count = len(machines)
    normal_count = sum(1 for m in machines if m.get("status") == "normal")
    watch_count = sum(1 for m in machines if m.get("status") == "watch")
    anomaly_count = sum(1 for m in machines if m.get("status") == "anomaly_detected")
    open_alerts_count = sum(1 for a in alerts_data.get("alerts", []) if a.get("status") == "open")

    # KPI Row
    render_kpi_cards(
        machines_count=machines_count,
        normal_count=normal_count,
        watch_count=watch_count,
        anomaly_count=anomaly_count,
        open_alerts_count=open_alerts_count
    )

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Current Machinery Status")
        if not machines:
            st.info("No machines registered yet. Navigate to **1_Machines** in the sidebar to add your first machine.")
        else:
            table_data = []
            for m in machines:
                table_data.append({
                    "Machine ID": m["machine_id"],
                    "Name": m["name"],
                    "Status": m.get("status", "no_data"),
                    "Latest Update": m.get("updated_at", "")[:19]
                })
            st.dataframe(table_data, use_container_width=True)

    with col_right:
        st.subheader("Recent Anomaly Alerts")
        recent_alerts = alerts_data.get("alerts", [])[:5]
        if not recent_alerts:
            st.success("✅ No recent alerts. Persistent vibration alerts are created only when an anomaly persists for 3 consecutive snapshots.")
        else:
            for alt_item in recent_alerts:
                severity_icon = "🔴" if alt_item.get("severity") == "high" else "🟠"
                st.markdown(
                    f"**{severity_icon} {alt_item.get('machine_id')}** — `{alt_item.get('severity').upper()}`  \n"
                    f"_{alt_item.get('timestamp')[:19]}_  \n"
                    f"{alt_item.get('message')}"
                )
                st.markdown("---")

    render_footer()


if __name__ == "__main__":
    main()
