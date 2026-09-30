"""
config.py - Environment-driven configuration for MachineMind AI Backend.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from backend directory if present
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "machinemind")
    ML_SERVICE_PATH = os.getenv("ML_SERVICE_PATH", str(BASE_DIR / "ml-service"))
    MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "ml-service" / "models"))
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
        if origin.strip()
    ]
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    PORT = int(os.getenv("PORT", "5000"))
    API_KEY = os.getenv("API_KEY", "")
