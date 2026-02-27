"""
AI Email Classification System
Flask application factory with logging configuration.
"""

import logging
from flask import Flask


def create_app():
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    # --------------- Logging ---------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    app.logger.setLevel(logging.INFO)
    app.logger.info("Flask application initialised.")

    # --------------- Model ---------------
    from app.model_loader import load_model
    load_model(app)

    # --------------- Routes ---------------
    from app.routes import bp
    app.register_blueprint(bp)

    return app
