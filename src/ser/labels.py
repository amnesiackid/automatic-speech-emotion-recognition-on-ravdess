"""Single source of truth for the emotion label set.

The integer ids follow the emotion coding of the RAVDESS file names
(01 = neutral ... 08 = surprised), shifted to start at zero. The same
mapping is baked into the published model's ``config.json``, so the
labels returned by the HuggingFace pipeline match ``LABELS`` exactly.
"""

ID2LABEL: dict[int, str] = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
}

LABEL2ID: dict[str, int] = {label: idx for idx, label in ID2LABEL.items()}

LABELS: list[str] = [ID2LABEL[i] for i in range(len(ID2LABEL))]

NUM_LABELS: int = len(LABELS)
