"""
Model loader — loads the trained classifier and TF-IDF vectorizer
into Flask app context once at startup.
"""

import os
import logging
import joblib
from flask import Flask

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "saved_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")


def load_model(app: Flask) -> None:
    """
    Load model and vectorizer into ``app.config`` so they are available
    across all request handlers.
    """
    if not os.path.exists(MODEL_PATH):
        logger.warning(
            "Model file not found at %s. Run `python model/train.py` first.", MODEL_PATH
        )
        app.config["MODEL"] = None
        app.config["VECTORIZER"] = None
        return

    app.config["MODEL"] = joblib.load(MODEL_PATH)
    app.config["VECTORIZER"] = joblib.load(VECTORIZER_PATH)
    logger.info("Model and vectorizer loaded successfully.")
