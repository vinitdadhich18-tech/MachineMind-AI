"""
db.py - MongoClient management and health check utilities for MachineMind AI.
"""

import logging
from typing import Optional
from pymongo import MongoClient

logger = logging.getLogger(__name__)

mongo_client: Optional[MongoClient] = None


def get_db_client(mongo_uri: str) -> Optional[MongoClient]:
    """
    Returns an initialized MongoClient instance with a short timeout.
    Returns None if MongoDB connection fails.
    """
    global mongo_client
    if mongo_client is not None:
        return mongo_client

    try:
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        mongo_client.admin.command('ping')
        return mongo_client
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}")
        mongo_client = None
        return None


def check_db_health(mongo_uri: str, db_name: str) -> str:
    """Checks MongoDB connection availability and returns 'connected' or 'unavailable'."""
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1500)
        client.admin.command('ping')
        return "connected"
    except Exception:
        return "unavailable"
