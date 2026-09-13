"""Helpers for the RAVDESS file-naming convention.

Every RAVDESS clip is named with seven two-digit fields separated by dashes:
``Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor``.
For example ``03-01-05-01-02-01-12.wav`` is audio-only (03) speech (01),
angry (05), normal intensity (01), "Dogs are sitting by the door" (02),
first repetition (01), spoken by actor 12 (even-numbered actors are female).

The emotion codes are the same ones ``ser.labels`` is built from, so the
label of a clip is simply ``ID2LABEL[code - 1]``.
"""

import re
from pathlib import Path

from ser.labels import ID2LABEL

STATEMENTS: dict[int, str] = {
    1: "Kids are talking by the door",
    2: "Dogs are sitting by the door",
}

INTENSITIES: dict[int, str] = {1: "normal", 2: "strong"}

_FILENAME = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$")


def parse_filename(name: str) -> dict | None:
    """Decode a RAVDESS file name into its metadata.

    Returns ``None`` when ``name`` does not follow the convention (so callers can
    mix RAVDESS clips with arbitrary audio files). Directory parts and the file
    extension are ignored.
    """
    match = _FILENAME.match(Path(name).stem)
    if match is None:
        return None

    _modality, _channel, emotion, intensity, statement, repetition, actor = (
        int(field) for field in match.groups()
    )
    if emotion - 1 not in ID2LABEL or statement not in STATEMENTS or intensity not in INTENSITIES:
        return None

    return {
        "emotion": ID2LABEL[emotion - 1],
        "intensity": INTENSITIES[intensity],
        "statement": STATEMENTS[statement],
        "repetition": repetition,
        "actor": actor,
        "gender": "female" if actor % 2 == 0 else "male",
    }
