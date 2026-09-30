"""
kpi.py - Render helper for KPI summary metric cards.
"""

import streamlit as st


def render_kpi_cards(
    machines_count: int,
    normal_count: int,
    watch_count: int,
    anomaly_count: int,
    open_alerts_count: int
):
    """Renders 5 KPI metric cards across page layout columns."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="Total Machines", value=str(machines_count))
    with col2:
        st.metric(label="🟢 Normal", value=str(normal_count))
    with col3:
        st.metric(label="🟡 Watch State", value=str(watch_count))
    with col4:
        st.metric(label="🔴 Anomaly Confirmed", value=str(anomaly_count))
    with col5:
        st.metric(label="⚠️ Open Alerts", value=str(open_alerts_count))
