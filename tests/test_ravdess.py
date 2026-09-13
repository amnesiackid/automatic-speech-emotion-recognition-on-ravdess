"""Tests for the RAVDESS file-name parser."""

import pytest

from ser.labels import LABELS
from ser.ravdess import parse_filename


def test_parse_full_name():
    meta = parse_filename("Actor_12/03-01-05-02-02-01-12.wav")
    assert meta == {
        "emotion": "angry",
        "intensity": "strong",
        "statement": "Dogs are sitting by the door",
        "repetition": 1,
        "actor": 12,
        "gender": "female",
    }


def test_emotion_codes_follow_label_order():
    for code, label in enumerate(LABELS, start=1):
        meta = parse_filename(f"03-01-{code:02d}-01-01-01-01.wav")
        assert meta is not None
        assert meta["emotion"] == label
        assert meta["gender"] == "male"


@pytest.mark.parametrize(
    "name",
    [
        "speech.wav",
        "03-01-09-01-01-01-01.wav",  # emotion code out of range
        "03-01-01-03-01-01-01.wav",  # intensity code out of range
        "03-01-01-01-03-01-01.wav",  # statement code out of range
        "03-01-01-01-01-01.wav",  # too few fields
    ],
)
def test_non_ravdess_names_return_none(name):
    assert parse_filename(name) is None
