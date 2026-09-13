"""Step 1: run every RAVDESS clip through frozen wav2vec2-base and cache the hidden states.

Writes ``features/<index>.pt`` (one (T, 768) tensor per clip) and
``features/labels.pt`` (list of integer labels in the same order).

    python extract_features.py [--output features]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from datasets import Audio, load_dataset

from model import SAMPLE_RATE, hidden_states, load_wav2vec2
from ser.labels import LABEL2ID

DATASET_ID = "amnesiackid/ravdess-emotion-intensity"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path("features"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    ds = load_dataset(DATASET_ID, split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    logger.info("Loaded %d clips", len(ds))

    processor, wav2vec2 = load_wav2vec2(device)

    labels = []
    for i, item in enumerate(ds):
        feats = hidden_states(item["audio"]["array"], processor, wav2vec2, device)
        torch.save(feats, args.output / f"{i}.pt")
        labels.append(LABEL2ID[item["emotion_labels"]])
        if (i + 1) % 100 == 0:
            logger.info("  %d / %d", i + 1, len(ds))

    torch.save(labels, args.output / "labels.pt")
    logger.info("Saved %d feature files to %s", len(labels), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
