"""
formatting.py - Timestamp and number formatting utilities for Streamlit dashboard.
"""

from datetime import datetime


def format_timestamp(ts: str) -> str:
    """Formats ISO-8601 or NASA timestamp string into a clean UTC display string."""
    if not ts:
        return "N/A"
    clean_ts = str(ts).strip()
    if len(clean_ts) >= 19:
        clean_ts = clean_ts[:19].replace("T", " ")
    return f"{clean_ts} UTC"


def format_number(val: float, precision: int = 4) -> str:
    """Formats floating point numbers cleanly."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{precision}f}"
    except (ValueError, TypeError):
        return str(val)
