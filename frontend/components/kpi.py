"""
kpi.py - High-density industrial KPI card renderer for Overview page.
"""

import streamlit as st


def render_kpi_cards(
    machines_count: int,
    normal_count: int,
    watch_count: int,
    anomaly_count: int,
    open_alerts_count: int
):
    """Renders 5 styled industrial KPI metric cards."""
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f"""
            <div class="ind-card ind-card-accent-blue">
                <div class="kpi-title">REGISTERED MACHINES</div>
                <div class="kpi-value" style="color: #F8FAFC;">{machines_count}</div>
                <div class="kpi-sub">Active Machinery Units</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="ind-card ind-card-accent-green">
                <div class="kpi-title">NORMAL STATE</div>
                <div class="kpi-value" style="color: #10B981;">{normal_count}</div>
                <div class="kpi-sub">Healthy Baseline</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="ind-card ind-card-accent-amber">
                <div class="kpi-title">WATCH STATE</div>
                <div class="kpi-value" style="color: #F59E0B;">{watch_count}</div>
                <div class="kpi-sub">Awaiting Persistence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="ind-card ind-card-accent-red">
                <div class="kpi-title">ANOMALY DETECTED</div>
                <div class="kpi-value" style="color: #EF4444;">{anomaly_count}</div>
                <div class="kpi-sub">Persistence Confirmed</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:
        st.markdown(
            f"""
            <div class="ind-card ind-card-accent-red">
                <div class="kpi-title">OPEN ALERTS</div>
                <div class="kpi-value" style="color: {'#EF4444' if open_alerts_count > 0 else '#94A3B8'};">{open_alerts_count}</div>
                <div class="kpi-sub">System Notifications</div>
            </div>
            """,
            unsafe_allow_html=True
        )
