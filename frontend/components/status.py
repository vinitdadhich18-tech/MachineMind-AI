"""
status.py - Single source of truth for status badges, indicators, and scientific wording.
"""

import streamlit as st

STATUS_MAP = {
    "no_data": {
        "label": "No data yet",
        "short_label": "No Data",
        "icon": "⚪",
        "color": "gray"
    },
    "normal": {
        "label": "Normal",
        "short_label": "Normal",
        "icon": "🟢",
        "color": "green"
    },
    "watch": {
        "label": "Anomalous snapshot — awaiting persistence confirmation",
        "short_label": "Watch",
        "icon": "🟡",
        "color": "amber"
    },
    "anomaly_detected": {
        "label": "Vibration anomaly detected",
        "short_label": "Anomaly Detected",
        "icon": "🔴",
        "color": "red"
    }
}

SEVERITY_MAP = {
    "warning": {
        "label": "Warning (1 Channel)",
        "icon": "🟠"
    },
    "high": {
        "label": "High Anomaly (Multi-Channel)",
        "icon": "🔴"
    }
}

DISCLAIMER_TEXT = (
    "*Research prototype. Detects statistical vibration anomalies; "
    "it does not confirm physical faults or predict failure.*"
)


def render_status_badge(state: str, short: bool = False):
    """Renders a status badge based on state string."""
    info = STATUS_MAP.get(state, STATUS_MAP["no_data"])
    label = info["short_label"] if short else info["label"]
    icon = info["icon"]

    if state == "normal":
        st.success(f"{icon} {label}")
    elif state == "watch":
        st.warning(f"{icon} {label}")
    elif state == "anomaly_detected":
        st.error(f"{icon} {label}")
    else:
        st.info(f"{icon} {label}")


def render_footer():
    """Renders the mandatory persistent scientific boundary footer caption."""
    st.markdown("---")
    st.caption(DISCLAIMER_TEXT)
