
"""
config.py - Environment variables and configuration settings for Streamlit frontend.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from frontend directory if present
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def get_config_val(key: str, default: str) -> str:
    # 1. Environment variable (upper and lower)
    val = os.getenv(key) or os.getenv(key.lower())
    if val is not None and val != "":
        return str(val)

    # 2. Streamlit Cloud Secrets (top-level or section)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            val = st.secrets.get(key)
            if val is None:
                val = st.secrets.get(key.lower())
            if val is None and hasattr(st.secrets, "get"):
                general_sec = st.secrets.get("general")
                if isinstance(general_sec, dict):
                    val = general_sec.get(key) or general_sec.get(key.lower())
    except Exception:
        val = None

    return str(val if val is not None and val != "" else default)


def get_backend_url() -> str:
    # Prefer environment variables, then Streamlit Cloud Secrets
    backend_url = get_config_val("BACKEND_URL", "http://localhost:5000")
    return backend_url.rstrip("/")


BACKEND_URL = get_backend_url()
READ_TIMEOUT = int(get_config_val("READ_TIMEOUT", "10"))
INFERENCE_TIMEOUT = int(get_config_val("INFERENCE_TIMEOUT", "60"))
TELEMETRY_TIMEOUT = int(get_config_val("TELEMETRY_TIMEOUT", "10"))
TELEMETRY_STALE_SECONDS = float(get_config_val("TELEMETRY_STALE_SECONDS", "30.0"))
TELEMETRY_REFRESH_SECONDS = int(get_config_val("TELEMETRY_REFRESH_SECONDS", "10"))

print(f"MachineMind backend configured: {BACKEND_URL}")