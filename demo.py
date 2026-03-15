#!/usr/bin/env python3
"""Classify speech emotion from an audio file using the fine-tuned DistilHuBERT model.

Usage:
    python demo.py <path/to/audio.wav>

Requires:
    pip install transformers torch soundfile
"""

import sys


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python demo.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    print("Loading amnesiackid/distilhubert-finetuned-ravdess...")
    from transformers import pipeline
    classifier = pipeline(
        "audio-classification",
        model="amnesiackid/distilhubert-finetuned-ravdess",
    )

    print(f"Classifying: {audio_path}\n")
    results = classifier(audio_path)

    print("Results:")
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        bar = "█" * int(r["score"] * 30)
        print(f"  {r['label']:10}  {r['score']:.1%}  {bar}")


if __name__ == "__main__":
    main()
