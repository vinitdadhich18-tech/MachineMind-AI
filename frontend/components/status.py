"""
status.py - Control-room application shell components, status indicators, and scientific boundary definitions.
"""

import streamlit as st
from typing import Optional, Dict, Any

STATUS_MAP = {
    "no_data": {
        "label": "No Data Yet",
        "short_label": "No Data",
        "pill_class": "pill-nodata",
        "dot_class": "status-dot-gray",
        "color": "#64748B"
    },
    "normal": {
        "label": "Normal Baseline Vibration",
        "short_label": "Normal",
        "pill_class": "pill-normal",
        "dot_class": "status-dot-normal",
        "color": "#10B981"
    },
    "watch": {
        "label": "Anomalous snapshot — awaiting persistence confirmation",
        "short_label": "Watch",
        "pill_class": "pill-watch",
        "dot_class": "status-dot-watch",
        "color": "#F59E0B"
    },
    "anomaly_detected": {
        "label": "Vibration anomaly confirmed",
        "short_label": "Anomaly Confirmed",
        "pill_class": "pill-anomaly",
        "dot_class": "status-dot-anomaly",
        "color": "#EF4444"
    }
}

DISCLAIMER_TEXT = (
    "Research prototype. Detects statistical vibration anomalies based on healthy baseline behavior; "
    "it does not confirm physical failures, predict remaining useful life (RUL), or establish physical failure prediction."
)


def render_global_header(
    active_machine_id: Optional[str] = None,
    telemetry_info: Optional[Dict[str, Any]] = None
):
    """Renders compact top application bar for control-room UI."""
    mach_display = active_machine_id or "E2E NASA TEST RIG"
    status_text = "ONLINE (ML v1)"
    if telemetry_info:
        seq = telemetry_info.get("sequence_id")
        if seq is not None:
            status_text = f"ONLINE (SEQ #{seq})"

    st.markdown(
        f"""
        <div class="ctrl-header-shell">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div>
                    <div class="ctrl-header-title">MachineMind AI <span style="font-weight: 400; color: #64748B;">|</span> <span style="font-size: 0.88rem; color: #94A3B8; font-weight: 500;">Vibration Monitoring Platform</span></div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 16px;">
                <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 600;">
                    ACTIVE MACHINE: <span style="color: #00E5FF; font-weight: 700;">{mach_display}</span>
                </div>
                <div style="font-size: 0.75rem; color: #10B981; font-weight: 700;">
                    <span class="status-dot-normal"></span> {status_text}
                </div>
                <span class="ctrl-tag-proto">Research Prototype</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar_shell(health: Optional[Dict[str, Any]] = None):
    """Renders sidebar logo branding and bottom system status indicator."""
    st.sidebar.markdown(
        """
        <div style="padding: 6px 0 12px 0; border-bottom: 1px solid #1C2436; margin-bottom: 14px;">
            <div style="font-size: 1.1rem; font-weight: 900; letter-spacing: 0.06em; color: #F1F5F9;">
                MM <span style="color: #00E5FF;">MACHINE MIND</span>
            </div>
            <div style="font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em; color: #64748B; margin-top: 1px;">
                VIBRATION MONITORING
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if health:
        backend_ok = health.get("status") == "ok"
        db_ok = health.get("database") == "connected"
        model_info = health.get("model", {})
        model_ok = model_info.get("loaded", False)
        version = model_info.get("version", "v1")

        b_dot = "status-dot-normal" if backend_ok else "status-dot-anomaly"
        d_dot = "status-dot-normal" if db_ok else "status-dot-anomaly"
        m_dot = "status-dot-normal" if model_ok else "status-dot-anomaly"

        st.sidebar.markdown(
            f"""
            <div style="background: #0B0E14; border: 1px solid #1C2436; border-radius: 4px; padding: 10px; margin-top: 20px;">
                <div style="font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; margin-bottom: 6px;">
                    SYSTEM STATUS
                </div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-bottom: 3px;">
                    <span class="{b_dot}"></span> API {'ONLINE' if backend_ok else 'OFFLINE'}
                </div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-bottom: 3px;">
                    <span class="{d_dot}"></span> DATABASE {'ONLINE' if db_ok else 'ERROR'}
                </div>
                <div style="font-size: 0.72rem; color: #94A3B8;">
                    <span class="{m_dot}"></span> MODEL {version} LOADED
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_status_badge(state: str, short: bool = False):
    """Renders HTML control-room status pill."""
    info = STATUS_MAP.get(state, STATUS_MAP["no_data"])
    label = info["short_label"] if short else info["label"]
    pill_cls = info["pill_class"]
    dot_cls = info["dot_class"]

    st.markdown(
        f'<span class="{pill_cls}"><span class="{dot_cls}"></span>{label}</span>',
        unsafe_allow_html=True
    )


def render_footer():
    """Renders single compact persistent research boundary caption."""
    st.markdown("---")
    st.caption(f"Research Prototype Boundary: {DISCLAIMER_TEXT}")
