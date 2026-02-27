"""
API routes for the Email Classification System.
"""

import logging
from flask import Blueprint, request, jsonify, render_template, current_app
from app.preprocessing import clean_text

logger = logging.getLogger(__name__)
bp = Blueprint("main", __name__)


@bp.route("/", methods=["GET"])
def index():
    """Serve the web interface."""
    return render_template("index.html")


@bp.route("/predict", methods=["POST"])
def predict():
    """
    Classify email text.

    Request JSON:
        { "email_text": "..." }

    Response JSON:
        { "category": "Spam", "confidence": 0.94 }
    """
    model = current_app.config.get("MODEL")
    vectorizer = current_app.config.get("VECTORIZER")

    if model is None or vectorizer is None:
        logger.error("Model not loaded.")
        return jsonify({"error": "Model not loaded. Train the model first."}), 503

    # ── Input validation ──
    data = request.get_json(silent=True)
    if not data or "email_text" not in data:
        return jsonify({"error": "Missing 'email_text' in request body."}), 400

    email_text = data["email_text"].strip()
    if not email_text:
        return jsonify({"error": "'email_text' must not be empty."}), 400

    # ── Preprocess & predict ──
    try:
        cleaned = clean_text(email_text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]

        # Confidence via predict_proba (available on both LR & MNB)
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities.max())

        # Build probability distribution
        classes = model.classes_.tolist()
        prob_distribution = {
            cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)
        }

        logger.info(
            "Prediction: %s (%.2f%%) for input: '%.60s…'",
            prediction, confidence * 100, email_text,
        )

        return jsonify({
            "category": prediction,
            "confidence": round(confidence, 4),
            "probabilities": prob_distribution,
        }), 200

    except Exception as exc:
        logger.exception("Prediction failed.")
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500
