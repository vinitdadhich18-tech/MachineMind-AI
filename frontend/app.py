"""
app.py - MachineMind AI Streamlit Dashboard Entrypoint (Overview Page).
"""

import streamlit as st
from frontend.components.status import render_footer, render_status_badge, DISCLAIMER_TEXT

st.set_page_config(
    page_title="MachineMind AI — Overview",
    page_icon="⚙️",
    layout="wide"
)


def main():
    st.title("⚙️ MachineMind AI — Vibration Anomaly Detection")
    st.markdown("### Unsupervised Vibration Anomaly Monitoring Dashboard (NASA IMS Bearing Dataset)")
    st.caption(DISCLAIMER_TEXT)

    st.markdown("---")

    # Overview Placeholders
    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Machines", value="0")
    with col2:
        st.metric(label="Normal State", value="0")
    with col3:
        st.metric(label="Watch State", value="0")
    with col4:
        st.metric(label="Open Alerts", value="0")

    st.markdown("---")

    st.info("👈 Select a page from the sidebar to manage machines, upload 4-channel snapshots, inspect predictions, or view alerts.")

    render_footer()


if __name__ == "__main__":
    main()
