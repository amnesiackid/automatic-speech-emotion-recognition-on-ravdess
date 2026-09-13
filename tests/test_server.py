"""Tests for the Flask API. The model pipeline is mocked: no GPU or internet needed."""

import struct
import wave
from unittest.mock import MagicMock, patch

import pytest

MOCK_RESULTS = [
    {"label": "happy", "score": 0.85},
    {"label": "neutral", "score": 0.10},
    {"label": "sad", "score": 0.05},
]


def _make_wav(path: str, duration_s: float = 0.5, sample_rate: int = 16000) -> None:
    """Write a minimal silent WAV file."""
    n_frames = int(duration_s * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))


@pytest.fixture
def mock_clf():
    return MagicMock(return_value=MOCK_RESULTS)


@pytest.fixture
def client(mock_clf):
    with patch("server.app.get_classifier", return_value=mock_clf):
        from server.app import app

        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


def test_index_serves_frontend(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"<html" in r.data.lower()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "healthy"
    assert "model" in data


def test_emotions(client):
    r = client.get("/emotions")
    assert r.status_code == 200
    labels = r.get_json()["emotions"]
    assert labels == [
        "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised",
    ]


def test_predict_no_audio(client):
    r = client.post("/predict")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_with_wav(client, mock_clf, tmp_path):
    wav_path = str(tmp_path / "test.wav")
    _make_wav(wav_path)

    with open(wav_path, "rb") as f:
        r = client.post(
            "/predict",
            data={"audio": (f, "test.wav")},
            content_type="multipart/form-data",
        )

    assert r.status_code == 200
    data = r.get_json()
    assert data["emotion"] == "happy"
    assert data["confidence"] == 0.85
    assert data["probabilities"] == {"happy": 0.85, "neutral": 0.10, "sad": 0.05}
    mock_clf.assert_called_once()


def test_predict_model_error_returns_500(client, mock_clf, tmp_path):
    mock_clf.side_effect = RuntimeError("boom")
    wav_path = str(tmp_path / "test.wav")
    _make_wav(wav_path)

    with open(wav_path, "rb") as f:
        r = client.post(
            "/predict",
            data={"audio": (f, "test.wav")},
            content_type="multipart/form-data",
        )

    assert r.status_code == 500
    assert r.get_json() == {"error": "Prediction failed"}


# ── RAVDESS sample endpoints ──────────────────────────────────────────────────


@pytest.fixture
def ravdess_dir(tmp_path, monkeypatch):
    """A tiny fake RAVDESS folder with the standard Actor_XX layout."""
    actor = tmp_path / "Actor_01"
    actor.mkdir()
    _make_wav(str(actor / "03-01-03-01-01-01-01.wav"))  # happy
    _make_wav(str(actor / "03-01-04-01-02-01-01.wav"))  # sad
    _make_wav(str(actor / "notes.wav"))  # not a RAVDESS name: must be skipped
    monkeypatch.setenv("RAVDESS_DIR", str(tmp_path))
    return tmp_path


def test_samples_unavailable_without_dataset(client, monkeypatch):
    monkeypatch.delenv("RAVDESS_DIR", raising=False)
    r = client.get("/samples")
    assert r.status_code == 200
    data = r.get_json()
    assert data["available"] is False
    assert data["samples"] == []
    assert "RAVDESS_DIR" in data["hint"]


def test_samples_unavailable_with_missing_folder(client, monkeypatch, tmp_path):
    monkeypatch.setenv("RAVDESS_DIR", str(tmp_path / "nope"))
    assert client.get("/samples").get_json()["available"] is False


def test_samples_lists_parsed_clips(client, ravdess_dir):
    data = client.get("/samples").get_json()
    assert data["available"] is True
    ids = [s["id"] for s in data["samples"]]
    assert ids == ["Actor_01/03-01-03-01-01-01-01.wav", "Actor_01/03-01-04-01-02-01-01.wav"]
    happy = data["samples"][0]
    assert happy["emotion"] == "happy"
    assert happy["actor"] == 1
    assert happy["statement"] == "Kids are talking by the door"


def test_sample_audio_is_served(client, ravdess_dir):
    r = client.get("/samples/Actor_01/03-01-03-01-01-01-01.wav")
    assert r.status_code == 200
    assert r.data[:4] == b"RIFF"
    assert client.get("/samples/Actor_01/missing.wav").status_code == 404
    assert client.get("/samples/../secret.wav").status_code == 404


def test_predict_sample(client, mock_clf, ravdess_dir):
    r = client.post(
        "/predict",
        data={"sample": "Actor_01/03-01-04-01-02-01-01.wav"},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert r.get_json()["emotion"] == "happy"
    called_path = mock_clf.call_args.args[0]
    assert called_path.endswith("03-01-04-01-02-01-01.wav")


def test_predict_unknown_sample(client, mock_clf, ravdess_dir):
    r = client.post(
        "/predict", data={"sample": "Actor_01/nope.wav"}, content_type="multipart/form-data"
    )
    assert r.status_code == 404
    mock_clf.assert_not_called()


def test_predict_sample_without_dataset(client, monkeypatch):
    monkeypatch.delenv("RAVDESS_DIR", raising=False)
    r = client.post(
        "/predict", data={"sample": "x.wav"}, content_type="multipart/form-data"
    )
    assert r.status_code == 404
    assert "RAVDESS_DIR" in r.get_json()["hint"]
