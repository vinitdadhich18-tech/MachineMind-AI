"""
app.py - Application factory and entrypoint for MachineMind AI Flask Backend.
"""

import os
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
    from backend.routes.data import data_bp
    from backend.routes.telemetry import telemetry_bp

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(inference_bp, url_prefix="/api")
    app.register_blueprint(machines_bp, url_prefix="/api")
    app.register_blueprint(predictions_bp, url_prefix="/api")
    app.register_blueprint(alerts_bp, url_prefix="/api")
    app.register_blueprint(data_bp, url_prefix="/api")
    app.register_blueprint(telemetry_bp, url_prefix="/api")

    # Initialize DB indexes and ensure canonical telemetry machine is registered
    from backend.utils.db import get_db, init_db_indexes
    try:
        db = get_db(config_class.MONGO_URI, config_class.DATABASE_NAME)
        if db is not None:
            init_db_indexes(db)
            # Ensure canonical NASA IMS Set 2 rig is registered in production/runtime (non-testing)
            if not getattr(config_class, "TESTING", False):
                from backend.services.machine_service import ensure_machine_registered
                with app.app_context():
                    canonical_machine_id = os.getenv("MQTT_MACHINE_ID", "ims_set2_rig").strip()
                    ensure_machine_registered(
                        machine_id=canonical_machine_id,
                        name="NASA IMS Bearing Test Rig",
                        description="NASA IMS Bearing Dataset (Set 2) 4-channel vibration test rig."
                    )
    except Exception as e:
        logger.warning(f"Could not initialize DB indexes or canonical machine at startup: {e}")

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
