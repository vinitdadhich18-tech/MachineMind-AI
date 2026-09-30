"""
conftest.py - Pytest fixtures for MachineMind AI Flask Backend.
"""

import pytest
from backend.app import create_app
from backend.config import Config


class TestConfig(Config):
    TESTING = True
    MONGO_URI = "mongodb://localhost:27017"
    DATABASE_NAME = "machinemind_test"


@pytest.fixture
def app():
    """Creates a Flask test application instance."""
    app_inst = create_app(TestConfig)
    yield app_inst


@pytest.fixture
def client(app):
    """Creates a test client for sending HTTP requests."""
    return app.test_client()
