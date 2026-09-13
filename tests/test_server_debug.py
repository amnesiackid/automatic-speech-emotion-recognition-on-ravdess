"""Debugging aids of the server: /health model revision and SER_SAVE_UPLOADS."""

import struct
import wave
from unittest.mock import MagicMock, patch

import pytest

from ser import predict as predict_module

MOCK_RESULTS = [{"label": "happy", "score": 0.9}, {"label": "sad", "score": 0.1}]


def _make_wav(path: str) -> None:
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<800h", *([0] * 800)))


@pytest.fixture
def mock_clf():
    clf = MagicMock(return_value=MOCK_RESULTS)
    clf.model.config._commit_hash = "abc1234"
    return clf


@pytest.fixture
def client(mock_clf):
    with patch("server.app.get_classifier", return_value=mock_clf):
        from server.app import app

        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


def test_health_reports_loaded_revision(client, mock_clf, monkeypatch):
    monkeypatch.setattr(predict_module, "_classifiers", {})
    assert client.get("/health").get_json()["model_loaded"] is False

    monkeypatch.setattr(predict_module, "_classifiers", {predict_module.MODEL_ID: mock_clf})
    data = client.get("/health").get_json()
    assert data["model_loaded"] is True
    assert data["model_revision"] == "abc1234"


def test_uploads_are_saved_when_configured(client, tmp_path, monkeypatch):
    saved = tmp_path / "uploads"
    monkeypatch.setenv("SER_SAVE_UPLOADS", str(saved))
    wav_path = str(tmp_path / "clip.wav")
    _make_wav(wav_path)

    with open(wav_path, "rb") as f:
        r = client.post("/predict", data={"audio": (f, "clip.wav")},
                        content_type="multipart/form-data")

    assert r.status_code == 200
    files = list(saved.iterdir())
    assert len(files) == 1
    assert files[0].name.endswith("_happy.wav")
    assert files[0].stat().st_size > 0


def test_uploads_are_not_saved_by_default(client, tmp_path, monkeypatch):
    monkeypatch.delenv("SER_SAVE_UPLOADS", raising=False)
    wav_path = str(tmp_path / "clip.wav")
    _make_wav(wav_path)
    with open(wav_path, "rb") as f:
        client.post("/predict", data={"audio": (f, "clip.wav")},
                    content_type="multipart/form-data")
    assert list(tmp_path.iterdir()) == [tmp_path / "clip.wav"]
