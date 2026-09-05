"""Flask REST API for speech emotion recognition, plus static hosting of the web demo.

Endpoints:
    GET  /           the web demo (frontend/index.html)
    POST /predict    multipart form with an ``audio`` file -> emotion + probabilities
    GET  /health     liveness check
    GET  /emotions   the ordered list of supported labels

The model is loaded lazily on the first ``/predict`` call.
"""

import logging
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from ser.labels import LABELS
from ser.predict import MODEL_ID, classify, get_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


@app.get("/")
def index():
    """Serve the web demo."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.post("/predict")
def predict():
    """Predict the emotion of an uploaded audio file.

    Expects ``multipart/form-data`` with an ``audio`` field. Returns JSON with
    ``emotion`` (str), ``confidence`` (float in [0, 1]) and ``probabilities``
    (label -> score for every label).
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    suffix = os.path.splitext(audio_file.filename or "")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        return jsonify(classify(tmp_path, classifier=get_classifier()))
    except Exception:
        logger.exception("Prediction failed for %s", audio_file.filename)
        return jsonify({"error": "Prediction failed"}), 500
    finally:
        os.unlink(tmp_path)


@app.get("/health")
def health():
    return jsonify({"status": "healthy", "model": MODEL_ID})


@app.get("/emotions")
def emotions():
    return jsonify({"emotions": LABELS})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
