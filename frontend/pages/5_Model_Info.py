"""
5_Model_Info.py - Technical Model Specification & Engineering Architecture Panel.
"""

import streamlit as st
from frontend.services.api_client import get_health, ApiError
from frontend.components.theme import inject_theme
from frontend.components.status import (
    render_global_header,
    render_sidebar_shell,
    render_status_badge,
    render_footer,
    DISCLAIMER_TEXT
)

st.set_page_config(page_title="MachineMind AI — Model Info", page_icon="ℹ️", layout="wide")


def main():
    inject_theme()

    # Fetch backend health
    try:
        health = get_health()
        render_sidebar_shell(health)
        model_info = health.get("model", {})
        version_str = model_info.get('version', 'v1')
        loaded_str = "LOADED" if model_info.get('loaded') else "NOT LOADED"
    except ApiError:
        version_str = "v1"
        loaded_str = "SERVICE OFFLINE"

    render_global_header("MODEL SPECIFICATION & ENGINEERING ARCHITECTURE")

    st.markdown(
        f"""
        <div class="ctrl-panel" style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.95rem; font-weight: 800; color: #F1F5F9;">
                ENGINEERING SPECIFICATION: <span style="color: #00E5FF;">iforest_pipeline_{version_str}.joblib</span>
            </span>
            <div>
                <span class="pill-normal">MODEL {version_str}</span>
                <span class="pill-normal" style="margin-left: 6px;">{loaded_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin-bottom: 8px;'>1. TECHNICAL SPECIFICATIONS TABLE</div>", unsafe_allow_html=True)
    
    spec_table = [
        {"PARAMETER": "PRIMARY TASK", "SPECIFICATION": "Unsupervised statistical vibration anomaly detection", "DETAILS": "Exceedance from healthy baseline window"},
        {"PARAMETER": "INPUT MATRIX", "SPECIFICATION": "20,480 samples × 4 channels", "DETAILS": "1-Second snapshot sampled at 20 kHz"},
        {"PARAMETER": "PREPROCESSING", "SPECIFICATION": "DC Mean-Centering per channel", "DETAILS": "Signal DC bias removal prior to extraction"},
        {"PARAMETER": "FEATURE EXTRACTION", "SPECIFICATION": "28 Time-Domain Features (7 per channel)", "DETAILS": "Mean, Std, RMS, Peak-to-Peak, Skewness, Kurtosis, Crest Factor"},
        {"PARAMETER": "FEATURE SCALING", "SPECIFICATION": "RobustScaler", "DETAILS": "Fit strictly on healthy baseline snapshots 0–159"},
        {"PARAMETER": "PRIMARY MODEL", "SPECIFICATION": "Isolation Forest (n_estimators=100)", "DETAILS": "Random forest anomaly isolation scoring"},
        {"PARAMETER": "ALTERNATIVE ENGINE", "SPECIFICATION": "PCA Reconstruction Error (n_components=3)", "DETAILS": "Dimensionality reduction reconstruction residual"},
        {"PARAMETER": "THRESHOLD CALIBRATION", "SPECIFICATION": "P99 Empirical Quantile Threshold", "DETAILS": "Non-parametric threshold fit on validation window 160–179"},
        {"PARAMETER": "PERSISTENCE RULE", "SPECIFICATION": "3 Consecutive Snapshots (N ≥ 3)", "DETAILS": "Eliminates single snapshot transient noise false-positives"},
        {"PARAMETER": "ALERT AGGREGATION", "SPECIFICATION": "Logical OR Policy", "DETAILS": "Triggered if any channel satisfies N ≥ 3 persistence"}
    ]

    st.dataframe(spec_table, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin-bottom: 8px;'>2. HORIZONTAL PIPELINE FLOW ARCHITECTURE</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ctrl-panel" style="padding: 16px;">
            <div style="display: flex; align-items: center; justify-content: space-between; text-align: center; font-size: 0.75rem; font-weight: 700;">
                <div style="background:#07090E; border:1px solid #1C2436; padding:8px 12px; border-radius:3px; color:#00E5FF;">
                    RAW SIGNAL<br><span style="font-size:0.65rem; color:#64748B;">20,480 × 4</span>
                </div>
                <div style="color:#64748B;">→</div>
                <div style="background:#07090E; border:1px solid #1C2436; padding:8px 12px; border-radius:3px; color:#10B981;">
                    DC CENTER<br><span style="font-size:0.65rem; color:#64748B;">Mean Removed</span>
                </div>
                <div style="color:#64748B;">→</div>
                <div style="background:#07090E; border:1px solid #1C2436; padding:8px 12px; border-radius:3px; color:#3B82F6;">
                    FEATURES<br><span style="font-size:0.65rem; color:#64748B;">28 Time Domain</span>
                </div>
                <div style="color:#64748B;">→</div>
                <div style="background:#07090E; border:1px solid #1C2436; padding:8px 12px; border-radius:3px; color:#F59E0B;">
                    ROBUST SCALE<br><span style="font-size:0.65rem; color:#64748B;">Baseline Scale</span>
                </div>
                <div style="color:#64748B;">→</div>
                <div style="background:#07090E; border:1px solid #1C2436; padding:8px 12px; border-radius:3px; color:#00E5FF;">
                    ISOLATION FOREST<br><span style="font-size:0.65rem; color:#64748B;">Anomaly Scoring</span>
                </div>
                <div style="color:#64748B;">→</div>
                <div style="background:#07090E; border:1px solid #1C2436; padding:8px 12px; border-radius:3px; color:#F59E0B;">
                    P99 THRESHOLD<br><span style="font-size:0.65rem; color:#64748B;">Exceedance</span>
                </div>
                <div style="color:#64748B;">→</div>
                <div style="background:#07090E; border:1px solid #1C2436; padding:8px 12px; border-radius:3px; color:#EF4444;">
                    3x PERSISTENCE<br><span style="font-size:0.65rem; color:#64748B;">SYSTEM ALERT</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin-bottom: 8px;'>3. SCIENTIFIC BOUNDARY DISCLOSURE</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ctrl-panel" style="font-size: 0.78rem; color: #94A3B8; line-height: 1.5; border-left: 3px solid #00E5FF;">
            <strong>Research Prototype Scope (NASA IMS Bearing Dataset - Set 2):</strong><br>
            • System detects statistical anomaly exceedances relative to healthy baseline vibration behavior.<br>
            • Does <em>not</em> predict remaining useful life (RUL), calculate time-to-failure, or classify physical bearing fault modes.<br>
            • Provided strictly as an educational / research presentation prototype.
        </div>
        """,
        unsafe_allow_html=True
    )

    render_footer()


if __name__ == "__main__":
    main()
