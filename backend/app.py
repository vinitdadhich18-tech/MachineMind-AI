"""
app.py - Application factory and entrypoint for MachineMind AI Flask Backend.
"""

import logging
from flask import Flask
from flask_cors import CORS

from backend.config import Config
from backend.utils.errors import register_error_handlers
from backend.routes.health import health_bp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(config_class=Config) -> Flask:
    """Creates and configures the Flask application instance."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Enable CORS
    CORS(app, origins=config_class.CORS_ORIGINS)

    # Register global JSON error handlers
    register_error_handlers(app)

    # Register blueprints with /api prefix
    app.register_blueprint(health_bp, url_prefix="/api")

    # Initialize ML service
    from backend.services.ml_service import init_ml_service, get_model_version
    model_loaded = init_ml_service(app.config["MODEL_DIR"], app.config["ML_SERVICE_PATH"])
    app.model_loaded = model_loaded
    app.model_version = get_model_version() if model_loaded else "v1"

    return app


if __name__ == "__main__":
    app = create_app()
    logger.info(f"Starting MachineMind AI Backend on port {app.config['PORT']}...")
    app.run(host="0.0.0.0", port=app.config["PORT"], debug=app.config["FLASK_DEBUG"])
