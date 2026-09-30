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
    from backend.routes.health import health_bp
    from backend.routes.inference import inference_bp
    from backend.routes.machines import machines_bp
    from backend.routes.predictions import predictions_bp
    from backend.routes.alerts import alerts_bp

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(inference_bp, url_prefix="/api")
    app.register_blueprint(machines_bp, url_prefix="/api")
    app.register_blueprint(predictions_bp, url_prefix="/api")
    app.register_blueprint(alerts_bp, url_prefix="/api")

    # Initialize DB indexes if DB is available
    from backend.utils.db import get_db, init_db_indexes
    try:
        db = get_db(config_class.MONGO_URI, config_class.DATABASE_NAME)
        if db is not None:
            init_db_indexes(db)
    except Exception as e:
        logger.warning(f"Could not initialize DB indexes at startup: {e}")

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
