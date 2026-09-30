"""
db.py - MongoClient management, database access, and index creation for MachineMind AI.
"""

import logging
from typing import Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database

logger = logging.getLogger(__name__)

mongo_client: Optional[MongoClient] = None
_test_db: Optional[Database] = None


def set_test_db(db: Optional[Database]) -> None:
    """Sets a test database instance (e.g. from mongomock) for unit testing."""
    global _test_db
    _test_db = db


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


def get_db(mongo_uri: str, db_name: str) -> Optional[Database]:
    """
    Returns the PyMongo Database object, or None if database is unavailable.
    If _test_db is set, returns _test_db directly.
    """
    global _test_db
    if _test_db is not None:
        return _test_db

    client = get_db_client(mongo_uri)
    if client is None:
        return None
    return client[db_name]


def check_db_health(mongo_uri: str, db_name: str) -> str:
    """Checks MongoDB connection availability and returns 'connected' or 'unavailable'."""
    global _test_db
    if _test_db is not None:
        return "connected"

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1500)
        client.admin.command('ping')
        return "connected"
    except Exception:
        return "unavailable"


def init_db_indexes(db: Database) -> None:
    """
    Creates required MongoDB indexes for machines, predictions, and alerts collections.
    """
    try:
        # machines: unique index on machine_id
        db.machines.create_index([("machine_id", ASCENDING)], unique=True)

        # predictions: index on machine_id + timestamp desc
        db.predictions.create_index([("machine_id", ASCENDING), ("timestamp", DESCENDING)])

        # alerts: index on machine_id + timestamp desc, status
        db.alerts.create_index([("machine_id", ASCENDING), ("timestamp", DESCENDING)])
        db.alerts.create_index([("status", ASCENDING)])

        logger.info("MongoDB database indexes successfully created/verified.")
    except Exception as e:
        logger.warning(f"Failed to create database indexes: {e}")
