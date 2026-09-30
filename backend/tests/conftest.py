"""
conftest.py - Pytest fixtures and mongomock DB setup for MachineMind AI Flask Backend.
"""

import pytest
import mongomock
from backend.app import create_app
from backend.config import Config
from backend.utils.db import set_test_db, init_db_indexes
from backend.services.ml_service import _machine_engines


class TestConfig(Config):
    TESTING = True
    MONGO_URI = "mongodb://localhost:27017"
    DATABASE_NAME = "machinemind_test"


@pytest.fixture
def app():
    """Creates a Flask test application instance with a fresh mongomock database."""
    client = mongomock.MongoClient()
    db = client[TestConfig.DATABASE_NAME]
    init_db_indexes(db)
    set_test_db(db)

    _machine_engines.clear()

    app_inst = create_app(TestConfig)
    yield app_inst

    set_test_db(None)


@pytest.fixture
def client(app):
    """Creates a test client for sending HTTP requests."""
    return app.test_client()
