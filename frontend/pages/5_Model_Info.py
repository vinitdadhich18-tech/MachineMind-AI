"""
5_Model_Info.py - Model Architecture, Methodology, and Limitations Page.
"""

import streamlit as st
from frontend.services.api_client import get_health, ApiError
from frontend.components.status import render_footer, DISCLAIMER_TEXT

st.set_page_config(page_title="MachineMind AI — Model Info", page_icon="ℹ️", layout="wide")


def main():
    st.title("ℹ️ ML Pipeline Architecture & Model Card")
    st.caption(DISCLAIMER_TEXT)
    st.markdown("---")

    # Fetch backend model version
    try:
        health = get_health()
        model_info = health.get("model", {})
        st.success(f"🟢 **Backend Model Service Active:** Version `{model_info.get('version', 'v1')}` | Loaded: `{model_info.get('loaded')}`")
    except ApiError:
        st.warning("⚠️ Backend service currently unreachable.")

    st.markdown("### Technical Methodology & Pipeline Summary")

    with st.expander("1. Purpose & Boundary Scope", expanded=True):
        st.markdown(
            "- **Scientific Purpose:** Unsupervised statistical vibration anomaly detection for rotating machinery.\n"
            "- **Evaluation Dataset:** NASA IMS Bearing Dataset (Set 2).\n"
            "- **System Boundary:** Detects statistical exceedance from healthy baseline behavior. "
            "It does **not** predict remaining useful life (RUL), predict physical failures, or classify fault types."
        )

    with st.expander("2. Input Specifications & Feature Extraction", expanded=True):
        st.markdown(
            "- **Input Shape:** One raw 1-second vibration snapshot = `20,480 samples × 4 channels`.\n"
            "- **Preprocessing:** DC mean-centering per channel.\n"
            "- **28 Time-Domain Features (7 per channel):** Mean, Standard Deviation, RMS, Peak-to-Peak, Skewness, Kurtosis, Crest Factor.\n"
            "- **Scaling:** `RobustScaler` (fit on healthy baseline snapshots 0–159)."
        )

    with st.expander("3. ML Algorithms & Threshold Calibration", expanded=True):
        st.markdown(
            "- **Primary Engine:** Isolation Forest (`n_estimators=100`, `max_samples='auto'`, `random_state=42`).\n"
            "- **Alternative Engine:** PCA Reconstruction Error (`n_components=3`, `random_state=42`).\n"
            "- **Threshold Calibration:** P99 non-parametric empirical quantile threshold, calibrated on healthy validation window (snapshots 160–179)."
        )

    with st.expander("4. Decision Logic & Persistence Rule", expanded=True):
        st.markdown(
            "- **3-Snapshot Persistence Rule:** Single snapshot exceedances are tagged as `watch` state. "
            "An anomaly is confirmed (`anomaly_detected`) **only** when a channel score exceeds threshold for **3 consecutive snapshots**.\n"
            "- **Multi-Channel Aggregation:** Logical OR policy across the 4 channels. If any channel confirms persistence, the overall system reports `Vibration anomaly detected`."
        )

    with st.expander("5. System Limitations & Disclaimer", expanded=False):
        st.markdown(
            "- **Research Dataset Scope:** Trained and calibrated specifically on the NASA IMS bearing test rig.\n"
            "- **Statistical Anomaly $\\neq$ Physical Fault:** Exceeding a statistical threshold indicates unusual vibration behavior relative to baseline, not necessarily irreversible physical damage.\n"
            "- **Not Certified:** This application is an educational/research learning prototype, not an industrially certified predictive maintenance system."
        )

    render_footer()


if __name__ == "__main__":
    main()
