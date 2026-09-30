"""
theme.py - Control-Room Industrial Visual System & CSS Injection for MachineMind AI.
"""

import streamlit as st

CONTROL_ROOM_CSS = """
<style>
/* Industrial Control-Room Theme System */
:root {
    --bg-dark: #07090E;
    --bg-panel: #101520;
    --bg-panel-hover: #161D2B;
    --border-subtle: #1C2436;
    --border-strong: #2A364F;
    
    --accent-cyan: #00E5FF;
    --accent-blue: #2563EB;
    
    --status-normal: #10B981;
    --status-watch: #F59E0B;
    --status-anomaly: #EF4444;
    --status-gray: #64748B;
    
    --text-main: #F1F5F9;
    --text-sub: #94A3B8;
    --text-muted: #64748B;
}

/* Page Reset */
.stApp {
    background-color: var(--bg-dark) !important;
    color: var(--text-main) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

.block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 1450px !important;
}

/* Hide Streamlit Default Chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #05070B !important;
    border-right: 1px solid var(--border-subtle) !important;
}

/* Control-Room Panels */
.ctrl-panel {
    background-color: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.ctrl-panel-highlight {
    background-color: var(--bg-panel);
    border: 1px solid var(--border-strong);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 4px;
    padding: 14px 16px;
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
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-main);
    letter-spacing: -0.01em;
}

.ctrl-header-subtitle {
    font-size: 0.75rem;
    color: var(--text-sub);
}

.ctrl-tag-proto {
    background: rgba(0, 229, 255, 0.08);
    border: 1px solid rgba(0, 229, 255, 0.25);
    color: var(--accent-cyan);
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 3px;
    text-transform: uppercase;
}

/* Operational Status Pills */
.status-dot-normal { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: var(--status-normal); box-shadow: 0 0 6px var(--status-normal); margin-right: 6px; }
.status-dot-watch { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: var(--status-watch); box-shadow: 0 0 6px var(--status-watch); margin-right: 6px; }
.status-dot-anomaly { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: var(--status-anomaly); box-shadow: 0 0 6px var(--status-anomaly); margin-right: 6px; }
.status-dot-gray { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: var(--status-gray); margin-right: 6px; }

.pill-normal { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); color: var(--status-normal); font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; }
.pill-watch { background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); color: var(--status-watch); font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; }
.pill-anomaly { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: var(--status-anomaly); font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; }
.pill-nodata { background: rgba(100, 116, 139, 0.1); border: 1px solid rgba(100, 116, 139, 0.3); color: var(--status-gray); font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; }

/* Horizontal Channel Condition Strip */
.channel-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 14px;
}

.channel-strip-item {
    background-color: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 10px 12px;
}

/* Streamlit Button & Form Overrides */
div.stButton > button {
    border-radius: 4px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 4px 14px !important;
}

div.stButton > button[kind="primary"] {
    background: var(--accent-cyan) !important;
    color: #000000 !important;
    border: none !important;
}

div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 10px rgba(0, 229, 255, 0.4) !important;
}

/* Dataframe & Table Overrides */
.stDataFrame {
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
