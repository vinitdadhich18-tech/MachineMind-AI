
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
    val = os.getenv(key)
    if val is None or val == "":
        try:
            import streamlit as st
            val = st.secrets.get(key)
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
TELEMETRY_TIMEOUT = int(get_config_val("TELEMETRY_TIMEOUT", "3"))
TELEMETRY_STALE_SECONDS = float(get_config_val("TELEMETRY_STALE_SECONDS", "30.0"))

print(f"MachineMind backend configured: {BACKEND_URL}")