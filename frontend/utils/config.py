
"""
config.py - Environment variables and configuration settings for Streamlit frontend.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from frontend directory if present
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def get_backend_url():
    # Prefer environment variables, then Streamlit Cloud Secrets
    backend_url = os.getenv("BACKEND_URL")

    if not backend_url:
        try:
            import streamlit as st
            backend_url = st.secrets.get("BACKEND_URL", None)
        except Exception:
            backend_url = None

    # Keep local development working by default
    return (backend_url or "http://localhost:5000").rstrip("/")


BACKEND_URL = get_backend_url()
READ_TIMEOUT = int(os.getenv("READ_TIMEOUT", "10"))
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", "60"))
