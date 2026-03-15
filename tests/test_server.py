"""Tests for the Flask emotion recognition API.

The HuggingFace pipeline is mocked so tests run without GPU or internet access.
"""

import struct
import wave
from unittest.mock import MagicMock, patch

import pytest

MOCK_RESULTS = [
    {"label": "happy", "score": 0.85},
    {"label": "neutral", "score": 0.10},
    {"label": "sad", "score": 0.05},
]

MOCK_ID2LABEL = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised",
}


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
    clf = MagicMock(return_value=MOCK_RESULTS)
    clf.model.config.id2label = MOCK_ID2LABEL
    return clf


@pytest.fixture
def client(mock_clf):
    with patch("server.app.get_classifier", return_value=mock_clf):
        from server.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "healthy"
    assert "model" in data


def test_emotions(client):
    r = client.get("/emotions")
    assert r.status_code == 200
    data = r.get_json()
    assert "emotions" in data
    assert len(data["emotions"]) == 8


def test_predict_no_audio(client):
    r = client.post("/predict")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_with_wav(client, tmp_path):
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
    assert "probabilities" in data
    assert len(data["probabilities"]) == 3
