"""Predict emotions with the baseline 1D CNN from the committed Keras weights.

Usage (from this directory, with the ``baseline`` extra installed):

    pip install -e ".[baseline]"
    python predict.py path/to/clip.wav [more.wav ...]

The label set of this experiment is alphabetical (see emotion_categories.json)
and uses ``fear`` / ``surprise`` rather than the ``fearful`` / ``surprised``
of the main model, because that is how the training notebook named them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from features import features_for_file
from model import build_model

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "SER_model.weights.h5"
DEFAULT_CATEGORIES = HERE / "emotion_categories.json"


def load_baseline(weights: Path = DEFAULT_WEIGHTS, categories: Path = DEFAULT_CATEGORIES):
    """Return ``(model, category_list)`` with the trained weights loaded."""
    with open(categories, encoding="utf-8") as f:
        labels = json.load(f)
    model = build_model(num_classes=len(labels))
    model.load_weights(str(weights))
    return model, labels


def predict_file(model, labels: list[str], path: str) -> dict:
    x = features_for_file(path)[np.newaxis, :, np.newaxis]  # (1, 2376, 1)
    probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    return {
        "emotion": labels[order[0]],
        "confidence": float(probs[order[0]]),
        "probabilities": {labels[i]: float(probs[i]) for i in order},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("audio", nargs="+", help="audio file(s) to classify")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    args = parser.parse_args(argv)

    model, labels = load_baseline(args.weights, args.categories)
    for path in args.audio:
        result = predict_file(model, labels, path)
        print(f"\n{path}\n  -> {result['emotion']} ({result['confidence']:.1%})")
        for label, p in result["probabilities"].items():
            print(f"  {label:10} {p:6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
