"""
config.py - Environment variables and configuration settings for Streamlit frontend.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from frontend directory if present
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000").rstrip("/")
READ_TIMEOUT = int(os.getenv("READ_TIMEOUT", "10"))
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", "60"))
