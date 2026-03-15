"""Flask REST API for Speech Emotion Recognition using fine-tuned DistilHuBERT."""

import logging
import os
import tempfile
from typing import Dict, List

from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_ID = "amnesiackid/distilhubert-finetuned-ravdess"

_classifier = None


def get_classifier():
    """Return the audio-classification pipeline, loading it on first call."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        logger.info("Loading model %s", MODEL_ID)
        _classifier = pipeline("audio-classification", model=MODEL_ID)
        logger.info("Model loaded successfully")
    return _classifier


@app.route("/predict", methods=["POST"])
def predict():
    """Predict emotion from an uploaded audio file.

    Expects multipart/form-data with an ``audio`` field.

    Returns:
        JSON with keys ``emotion`` (str), ``confidence`` (float 0–1),
        and ``probabilities`` (dict mapping each label to its score).
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    suffix = os.path.splitext(audio_file.filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        results = get_classifier()(tmp_path)
        top = results[0]
        return jsonify({
            "emotion": top["label"],
            "confidence": round(top["score"], 4),
            "probabilities": {r["label"]: round(r["score"], 4) for r in results},
        })
    except Exception:
        logger.exception("Prediction failed for %s", audio_file.filename)
        return jsonify({"error": "Prediction failed"}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/health", methods=["GET"])
def health():
    """Return server health status."""
    return jsonify({"status": "healthy", "model": MODEL_ID})


@app.route("/emotions", methods=["GET"])
def emotions():
    """Return the list of supported emotion labels in label-index order."""
    id2label: Dict[int, str] = get_classifier().model.config.id2label
    labels: List[str] = [id2label[i] for i in sorted(id2label.keys())]
    return jsonify({"emotions": labels})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
