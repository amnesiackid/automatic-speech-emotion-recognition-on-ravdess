"""Sanity checks on the shared label table."""

from ser.labels import ID2LABEL, LABEL2ID, LABELS, NUM_LABELS


def test_eight_ravdess_emotions_in_dataset_order():
    assert NUM_LABELS == 8
    assert LABELS == [
        "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised",
    ]


def test_mappings_are_inverse():
    assert ID2LABEL == {idx: label for label, idx in LABEL2ID.items()}
    assert sorted(ID2LABEL) == list(range(NUM_LABELS))
