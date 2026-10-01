"""
theme.py - Control-Room Industrial Visual System & CSS Injection for MachineMind AI.

Provides a high-density, dark industrial aesthetic inspired by observability platforms,
industrial control rooms, and telemetry monitoring software.
"""

import streamlit as st

CONTROL_ROOM_CSS = """
<style>
/* Industrial Control-Room Theme Palette */
:root {
    --bg-dark: #07090E;
    --bg-panel: #101520;
    --bg-panel-hover: #161D2B;
    --border-subtle: #1C2436;
    --border-strong: #2A364F;
    
    --accent-cyan: #00E5FF;
    --accent-blue: #2563EB;
    --accent-teal: #14B8A6;
    
    --status-live: #10B981;
    --status-warning: #F59E0B;
    --status-error: #EF4444;
    --status-muted: #64748B;
    
    --text-main: #F1F5F9;
    --text-sub: #94A3B8;
    --text-muted: #64748B;
}

/* Page Background & Reset */
.stApp {
    background-color: var(--bg-dark) !important;
    color: var(--text-main) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

.block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 1450px !important;
}

/* Hide Streamlit Default Chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #05070B !important;
    border-right: 1px solid var(--border-subtle) !important;
}

/* Industrial Control-Room Panels */
.ctrl-panel {
    background-color: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 12px 16px;
    margin-bottom: 12px;
}

.ctrl-panel-highlight {
    background-color: var(--bg-panel);
    border: 1px solid var(--border-strong);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 4px;
    padding: 12px 16px;
    margin-bottom: 12px;
}

/* Top Header Shell */
.ctrl-header-shell {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 10px 16px;
    margin-bottom: 14px;
}

.ctrl-header-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -0.01em;
}

.ctrl-header-subtitle {
    font-size: 0.78rem;
    color: var(--text-sub);
}

.ctrl-tag-live {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: var(--status-live);
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 3px;
    letter-spacing: 0.05em;
}

.ctrl-tag-proto {
    background: rgba(0, 229, 255, 0.08);
    border: 1px solid rgba(0, 229, 255, 0.25);
    color: var(--accent-cyan);
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 3px;
    text-transform: uppercase;
}

/* Operational Status Pills */
.status-dot-live { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: var(--status-live); box-shadow: 0 0 6px var(--status-live); margin-right: 6px; }
.status-dot-warning { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: var(--status-warning); box-shadow: 0 0 6px var(--status-warning); margin-right: 6px; }
.status-dot-error { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: var(--status-error); box-shadow: 0 0 6px var(--status-error); margin-right: 6px; }
.status-dot-muted { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: var(--status-muted); margin-right: 6px; }

.pill-live { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); color: var(--status-live); font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; }
.pill-warning { background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); color: var(--status-warning); font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; }
.pill-error { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: var(--status-error); font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; }

/* Section Header styling */
.section-hdr {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-main);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Metric Widget Customization */
div[data-testid="stMetric"] {
    background-color: var(--bg-panel) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
    padding: 10px 14px !important;
}

div[data-testid="stMetricLabel"] > label {
    color: var(--text-sub) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

div[data-testid="stMetricValue"] > div {
    color: var(--text-main) !important;
    font-size: 1.25rem !important;
    font-weight: 800 !important;
}

/* Dataframe & Table Overrides */
.stDataFrame {
    background-color: var(--bg-panel) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
}

/* Expander Overrides */
div[data-testid="stExpander"] {
    background-color: var(--bg-panel) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
}

/* Horizontal Rule */
hr {
    border-color: var(--border-subtle) !important;
    margin: 1rem 0 !important;
}
</style>
"""

def inject_theme():
    """Injects control-room global CSS into current page."""
    st.markdown(CONTROL_ROOM_CSS, unsafe_allow_html=True)
