"""Inference helpers for the fine-tuned DistilHuBERT emotion classifier.

Used by the CLI (``ser-predict`` / ``python -m ser.predict``), the Flask
server and the tests, so the model is loaded and post-processed in exactly
one place.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from ser.labels import NUM_LABELS

MODEL_ID = "amnesiackid/distilhubert-finetuned-ravdess"

logger = logging.getLogger(__name__)

_classifiers: dict[str, Any] = {}


def get_classifier(model_id: str = MODEL_ID):
    """Return a HuggingFace audio-classification pipeline, loading it on first use.

    ``transformers`` is imported lazily so that modules importing this one
    (the Flask app, the tests) stay cheap to import.
    """
    if model_id not in _classifiers:
        from transformers import pipeline

        logger.info("Loading model %s", model_id)
        _classifiers[model_id] = pipeline("audio-classification", model=model_id)
        logger.info("Model loaded")
    return _classifiers[model_id]


def classify(audio, classifier=None) -> dict[str, Any]:
    """Classify one audio input and return a JSON-serialisable result.

    Args:
        audio: a file path, or a dict ``{"array": np.ndarray, "sampling_rate": int}``
            as accepted by the HuggingFace pipeline.
        classifier: an already-loaded pipeline; defaults to :func:`get_classifier`.

    Returns:
        ``{"emotion": str, "confidence": float, "probabilities": {label: float}}``
        with a probability for every label, highest first.
    """
    clf = classifier if classifier is not None else get_classifier()
    results = clf(audio, top_k=NUM_LABELS)
    results = sorted(results, key=lambda r: r["score"], reverse=True)
    top = results[0]
    return {
        "emotion": top["label"],
        "confidence": round(float(top["score"]), 4),
        "probabilities": {r["label"]: round(float(r["score"]), 4) for r in results},
    }


def format_result(result: dict[str, Any], width: int = 30) -> str:
    """Render a result from :func:`classify` as text bars for the terminal."""
    lines = []
    for label, score in result["probabilities"].items():
        bar = "#" * int(score * width)
        lines.append(f"  {label:10}  {score:6.1%}  {bar}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ser-predict",
        description="Classify the emotion in one or more speech recordings.",
    )
    parser.add_argument("audio", nargs="+", help="path(s) to audio file(s): wav, mp3, flac, ...")
    parser.add_argument("--model", default=MODEL_ID,
                        help="HuggingFace model id or local checkpoint (default: %(default)s)")
    args = parser.parse_args(argv)

    print(f"Loading {args.model} ...")
    clf = get_classifier(args.model)
    for path in args.audio:
        result = classify(path, classifier=clf)
        print(f"\n{path}\n  -> {result['emotion']} ({result['confidence']:.1%})\n")
        print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
