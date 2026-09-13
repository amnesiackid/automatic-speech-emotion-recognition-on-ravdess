"""Flask REST API for speech emotion recognition, plus static hosting of the web demo.

Endpoints:
    GET  /                 the web demo (frontend/index.html)
    POST /predict          multipart form with an ``audio`` file, or a ``sample`` id
                           from the local RAVDESS copy -> emotion + probabilities
    GET  /samples          the RAVDESS clips available for testing (see RAVDESS_DIR)
    GET  /samples/<id>     stream one of those clips
    GET  /health           liveness check
    GET  /emotions         the ordered list of supported labels

The model is loaded lazily on the first ``/predict`` call.

Set the ``RAVDESS_DIR`` environment variable to a folder containing RAVDESS
``.wav`` files (any layout, searched recursively) to enable the "Try testing on
RAVDESS" feature of the web demo. Without it the sample endpoints report that no
dataset is available and the rest of the API works as before.

Set ``SER_SAVE_UPLOADS`` to a folder to keep a copy of every uploaded recording,
named after the predicted emotion, for debugging what the model receives.
"""

import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import safe_join

from ser import predict as predict_module
from ser.labels import LABELS
from ser.predict import MODEL_ID, classify, get_classifier
from ser.ravdess import parse_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
RAVDESS_HINT = (
    "Set the RAVDESS_DIR environment variable to a folder with RAVDESS .wav files "
    "before starting the server."
)

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


def ravdess_dir() -> Path | None:
    """The local RAVDESS folder, or ``None`` when it is not configured or missing."""
    value = os.environ.get("RAVDESS_DIR", "").strip()
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.is_dir() else None


@app.get("/")
def index():
    """Serve the web demo."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.post("/predict")
def predict():
    """Predict the emotion of an uploaded audio file or of a local RAVDESS clip.

    Expects ``multipart/form-data`` with either an ``audio`` file field or a
    ``sample`` text field holding an id returned by ``GET /samples``. Returns JSON
    with ``emotion`` (str), ``confidence`` (float in [0, 1]) and ``probabilities``
    (label -> score for every label).
    """
    if "audio" in request.files:
        audio_file = request.files["audio"]
        suffix = os.path.splitext(audio_file.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        try:
            response = _classify(tmp_path, audio_file.filename)
            _save_upload(tmp_path, suffix, response)
            return response
        finally:
            os.unlink(tmp_path)

    sample_id = request.form.get("sample")
    if sample_id:
        root = ravdess_dir()
        if root is None:
            return jsonify({"error": "No RAVDESS dataset is configured", "hint": RAVDESS_HINT}), 404
        path = safe_join(str(root), sample_id)
        if path is None or not os.path.isfile(path):
            return jsonify({"error": "Unknown sample"}), 404
        return _classify(path, sample_id)

    return jsonify({"error": "No audio file provided"}), 400


def _classify(path: str, name: str | None):
    try:
        return jsonify(classify(path, classifier=get_classifier()))
    except Exception:
        logger.exception("Prediction failed for %s", name)
        return jsonify({"error": "Prediction failed"}), 500


def _save_upload(path: str, suffix: str, response) -> None:
    """Keep a copy of an uploaded recording when ``SER_SAVE_UPLOADS`` names a folder.

    Meant for debugging what the model actually receives from the web demo:
    files are named ``<timestamp>_<predicted emotion><suffix>`` so a batch of
    them can be listened to and re-run through ``ser-predict``.
    """
    target = os.environ.get("SER_SAVE_UPLOADS", "").strip()
    if not target:
        return
    try:
        Path(target).mkdir(parents=True, exist_ok=True)
        status = response[1] if isinstance(response, tuple) else 200
        body = (response[0] if isinstance(response, tuple) else response).get_json() or {}
        label = body.get("emotion", "error") if status == 200 else "error"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        shutil.copyfile(path, Path(target) / f"{stamp}_{label}{suffix}")
    except OSError:
        logger.exception("Could not save upload to %s", target)


@app.get("/samples")
def samples():
    """List the RAVDESS clips under ``RAVDESS_DIR`` with their decoded metadata.

    ``id`` is the path relative to ``RAVDESS_DIR`` and is what ``/samples/<id>``
    and the ``sample`` field of ``/predict`` accept. Files whose names do not
    follow the RAVDESS convention are skipped.
    """
    root = ravdess_dir()
    if root is None:
        return jsonify({"available": False, "samples": [], "hint": RAVDESS_HINT})

    clips = []
    for path in sorted(root.rglob("*.wav")):
        meta = parse_filename(path.name)
        if meta is not None:
            clips.append({"id": path.relative_to(root).as_posix(), "name": path.name, **meta})
    return jsonify({"available": True, "samples": clips})


@app.get("/samples/<path:sample_id>")
def sample_audio(sample_id: str):
    """Stream a clip from the local RAVDESS copy so the demo can play it back."""
    root = ravdess_dir()
    if root is None:
        return jsonify({"error": "No RAVDESS dataset is configured", "hint": RAVDESS_HINT}), 404
    return send_from_directory(root, sample_id)


@app.get("/health")
def health():
    """Liveness check; also reports which model revision is in memory once loaded."""
    info = {"status": "healthy", "model": MODEL_ID, "model_loaded": False}
    clf = predict_module._classifiers.get(MODEL_ID)
    if clf is not None:
        info["model_loaded"] = True
        info["model_revision"] = getattr(clf.model.config, "_commit_hash", None)
    return jsonify(info)


@app.get("/emotions")
def emotions():
    return jsonify({"emotions": LABELS})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
