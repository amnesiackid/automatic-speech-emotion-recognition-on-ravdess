"""Tests for ser.predict with a mocked pipeline."""

from unittest.mock import MagicMock

from ser import predict
from ser.labels import NUM_LABELS

RAW = [
    {"label": "sad", "score": 0.2},
    {"label": "happy", "score": 0.7999999},
    {"label": "calm", "score": 0.0000001},
]


def test_classify_sorts_rounds_and_requests_all_labels():
    clf = MagicMock(return_value=RAW)

    result = predict.classify("clip.wav", classifier=clf)

    clf.assert_called_once_with("clip.wav", top_k=NUM_LABELS)
    assert result["emotion"] == "happy"
    assert result["confidence"] == 0.8
    assert list(result["probabilities"]) == ["happy", "sad", "calm"]
    assert result["probabilities"]["calm"] == 0.0


def test_classify_falls_back_to_default_classifier(monkeypatch):
    clf = MagicMock(return_value=RAW)
    monkeypatch.setattr(predict, "get_classifier", lambda: clf)

    assert predict.classify("clip.wav")["emotion"] == "happy"
    clf.assert_called_once()


def test_format_result_one_line_per_label():
    result = predict.classify("clip.wav", classifier=MagicMock(return_value=RAW))

    lines = predict.format_result(result).splitlines()

    assert len(lines) == 3
    assert lines[0].split()[0] == "happy"
    assert "80.0%" in lines[0]


def test_main_prints_prediction(monkeypatch, capsys):
    clf = MagicMock(return_value=RAW)
    monkeypatch.setattr(predict, "get_classifier", lambda model_id: clf)

    exit_code = predict.main(["a.wav", "b.wav"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out.count("-> happy") == 2
    assert clf.call_count == 2
