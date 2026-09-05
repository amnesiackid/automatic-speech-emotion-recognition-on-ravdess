"""Speech emotion recognition on RAVDESS with a fine-tuned DistilHuBERT model."""

from ser.labels import ID2LABEL, LABEL2ID, LABELS, NUM_LABELS
from ser.predict import MODEL_ID, classify, get_classifier

__all__ = [
    "ID2LABEL",
    "LABEL2ID",
    "LABELS",
    "NUM_LABELS",
    "MODEL_ID",
    "classify",
    "get_classifier",
]
