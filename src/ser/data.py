"""
Data preparation for RAVDESS speech emotion recognition.

Downloads the RAVDESS dataset from the HuggingFace Hub, decodes and resamples
every clip to 16 kHz mono, encodes labels, extracts the actor id from the file
name, and writes a DatasetDict with ``train`` / ``validation`` / ``test``
splits to disk for train.py and evaluate.py.

Waveforms are stored raw (not as model features) so that train.py can apply
augmentation on the fly.

Splits:
  * default: the same seed-42 random 80/20 train/test split the published
    model was evaluated on, plus a stratified validation split carved out of
    the training portion for model selection
  * ``--split-by-actor``: hold out whole actors instead, so validation and
    test speakers are never seen in training (a more honest estimate of how
    the model behaves on new voices)

Extra corpora (``--extra-corpora crema-d tess savee`` or ``all``) are
downloaded from Kaggle and added to the *training* split only, so the RAVDESS
validation and test numbers stay comparable across runs. They add 97 speakers
with other accents and recording chains, which is what a RAVDESS-only model
lacks when it meets a new voice.

Usage:
    python -m ser.data
    python -m ser.data --output data/ravdess --val-size 0.1 --seed 42
    python -m ser.data --split-by-actor
    python -m ser.data --extra-corpora all --output ravdess_plus
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import re

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from datasets import (
    Audio,
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
)

from ser import corpora
from ser.labels import LABEL2ID, LABELS

try:  # datasets >= 4 calls the list feature ``List``; older versions ``Sequence``
    from datasets import List as ListFeature
except ImportError:  # pragma: no cover
    from datasets import Sequence as ListFeature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_ID = "amnesiackid/ravdess-emotion-intensity"
SAMPLE_RATE = 16_000
MAX_DURATION = 4.5  # seconds; train.py / evaluate.py crop clips to this length

_ACTOR_RE = re.compile(r"-(\d\d)\.wav$")

PROCESSED_FEATURES = Features(
    {
        "waveform": ListFeature(Value("float32")),
        "label": ClassLabel(names=LABELS),
        "actor": Value("int32"),
        "intensity": Value("string"),
        "corpus": Value("string"),
    }
)


# ── Steps ─────────────────────────────────────────────────────────────────────

def load_raw(dataset_id: str = DATASET_ID) -> Dataset:
    """Load the Hub dataset without decoding audio (decoding is done by :func:`decode`)."""
    logger.info("Loading dataset: %s", dataset_id)
    ds = load_dataset(dataset_id, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    logger.info("Loaded %d utterances", len(ds))
    return ds


MAX_RESAMPLE_PHASES = 1000  # above this, torchaudio's polyphase kernel gets very slow


def resample(x: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample a mono float32 array to ``target_sr``.

    torchaudio's resampler costs ``sr / gcd(sr, target_sr)`` filter phases; for
    friendly rates (48 000, 44 100) that is cheap, but for TESS's 24 414 Hz it
    is 12 207 phases and about 3 s per clip. Such ratios use FFT (sinc)
    resampling instead, which is exact for any ratio and takes milliseconds.
    """
    if sr == target_sr:
        return x.astype(np.float32)
    if sr // math.gcd(sr, target_sr) <= MAX_RESAMPLE_PHASES:
        y = AF.resample(torch.from_numpy(np.ascontiguousarray(x)), orig_freq=sr, new_freq=target_sr)
        return y.numpy().astype(np.float32)
    n_in = len(x)
    n_out = int(round(n_in * target_sr / sr))
    spec = np.fft.rfft(x)
    out = np.zeros(n_out // 2 + 1, dtype=spec.dtype)
    keep = min(len(out), len(spec))
    out[:keep] = spec[:keep]  # truncating the spectrum is an ideal low-pass
    return (np.fft.irfft(out, n=n_out) * (n_out / n_in)).astype(np.float32)


def decode_waveform(audio: dict, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode one ``{"bytes", "path"}`` audio entry to a 16 kHz mono float32 array."""
    if audio.get("bytes") is not None:
        array, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    else:
        array, sr = sf.read(audio["path"], dtype="float32", always_2d=True)
    return resample(array.mean(axis=1), sr, target_sr)


def actor_id(path: str | None) -> int:
    """RAVDESS file names end in the actor number: 03-01-06-01-02-02-10.wav -> 10."""
    match = _ACTOR_RE.search(path or "")
    return int(match.group(1)) if match else -1


def decode(ds: Dataset, num_proc: int = 1) -> Dataset:
    """Decode + resample every clip and encode labels and actors."""

    def _process(example):
        return {
            "waveform": decode_waveform(example["audio"]),
            "label": LABEL2ID[example["emotion_labels"]],
            "actor": actor_id(example["audio"].get("path")),
            "intensity": example["intensity"],
            "corpus": "ravdess",
        }

    logger.info("Decoding and resampling to %d Hz ...", SAMPLE_RATE)
    ds = ds.map(
        _process,
        remove_columns=ds.column_names,
        features=PROCESSED_FEATURES,
        num_proc=num_proc,
        desc="decode",
    )
    logger.info("Actors found: %s", sorted(set(ds["actor"])))
    return ds


def decode_extra_corpora(names: list[str], num_proc: int = 1) -> Dataset:
    """Download the requested extra corpora from Kaggle and decode them like RAVDESS."""
    parts = []
    for name in names:
        root = corpora.download(name)
        clips = list(corpora.iter_corpus(name, root))
        table = {
            "path": [str(p) for p, _ in clips],
            "label": [m["label"] for _, m in clips],
            "actor": [m["actor"] for _, m in clips],
            "intensity": [m["intensity"] for _, m in clips],
            "corpus": [name] * len(clips),
        }
        logger.info("%s: %d clips from %d speakers", name, len(clips), len(set(table["actor"])))

        def _process(example):
            return {"waveform": decode_waveform({"bytes": None, "path": example["path"]})}

        ds = Dataset.from_dict(table).map(
            _process,
            remove_columns=["path"],
            features=PROCESSED_FEATURES,
            num_proc=num_proc,
            desc=f"decode {name}",
        )
        parts.append(ds)
    return concatenate_datasets(parts)


def make_splits(
    ds: Dataset,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    by_actor: bool = False,
) -> DatasetDict:
    """Return a DatasetDict with train / validation / test.

    ``test_size`` and ``val_size`` are fractions of the whole dataset. With
    ``by_actor=False`` the test split is exactly the seed-42 split used for the
    published model; the validation split is then taken (stratified by label)
    from the remaining training clips. With ``by_actor=True`` whole actors are
    assigned to each split.
    """
    if by_actor:
        actors = sorted(set(ds["actor"]))
        rng = np.random.default_rng(seed)
        order = [actors[i] for i in rng.permutation(len(actors))]
        n_test = max(1, round(len(actors) * test_size))
        n_val = max(1, round(len(actors) * val_size)) if val_size > 0 else 0
        test_actors, val_actors = set(order[:n_test]), set(order[n_test : n_test + n_val])
        logger.info("Held-out actors — test: %s  validation: %s",
                    sorted(test_actors), sorted(val_actors))
        held_out = test_actors | val_actors
        splits = DatasetDict(
            {
                "train": ds.filter(lambda a: a not in held_out, input_columns="actor"),
                "validation": ds.filter(lambda a: a in val_actors, input_columns="actor"),
                "test": ds.filter(lambda a: a in test_actors, input_columns="actor"),
            }
        )
    else:
        first = ds.train_test_split(test_size=test_size, shuffle=True, seed=seed)
        train, test = first["train"], first["test"]
        if val_size > 0:
            val_frac = val_size / (1.0 - test_size)
            second = train.train_test_split(
                test_size=val_frac, shuffle=True, seed=seed, stratify_by_column="label"
            )
            train, validation = second["train"], second["test"]
        else:
            validation = None
        splits = DatasetDict({"train": train, "test": test})
        if validation is not None:
            splits["validation"] = validation

    for name, part in splits.items():
        counts = np.bincount(part["label"], minlength=len(LABELS))
        logger.info("%-10s %4d clips  per class: %s", name, len(part), counts.tolist())
    return splits


def validate(splits: DatasetDict) -> None:
    """Sanity-check the processed dataset."""
    sample = splits["train"][0]
    assert set(sample) == set(PROCESSED_FEATURES), sample.keys()
    wave = np.asarray(sample["waveform"], dtype=np.float32)
    assert wave.ndim == 1 and 1.0 <= len(wave) / SAMPLE_RATE <= 10.0, wave.shape
    assert 0 <= sample["label"] < len(LABELS)
    if "validation" in splits:
        train_actors = set(splits["train"]["actor"])
        for name in ("validation", "test"):
            overlap = train_actors & set(splits[name]["actor"])
            logger.info("Actors shared between train and %s: %d", name, len(overlap))
    logger.info("Validation passed — first clip: %.2f s, label %s",
                len(wave) / SAMPLE_RATE, LABELS[sample["label"]])


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="ravdess_encoded",
                        help="Directory to save the processed DatasetDict (default: %(default)s)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of clips (or actors) held out for testing")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Fraction held out for validation / model selection (0 disables)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-by-actor", action="store_true",
                        help="Speaker-independent splits: hold out whole actors")
    parser.add_argument("--num-proc", type=int, default=1, help="Workers for decoding")
    parser.add_argument("--extra-corpora", nargs="+", default=[],
                        choices=[*corpora.CORPORA, "all"], metavar="CORPUS",
                        help="Add CREMA-D / TESS / SAVEE (from Kaggle) to the training split; "
                             "'all' for the three of them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = decode(load_raw(), num_proc=args.num_proc)
    splits = make_splits(ds, args.test_size, args.val_size, args.seed, args.split_by_actor)

    extra = list(corpora.CORPORA) if "all" in args.extra_corpora else args.extra_corpora
    if extra:
        extra_ds = decode_extra_corpora(extra, num_proc=args.num_proc)
        splits["train"] = concatenate_datasets([splits["train"], extra_ds])
        counts = np.bincount(splits["train"]["label"], minlength=len(LABELS))
        logger.info("train with extra corpora: %d clips  per class: %s  corpora: %s",
                    len(splits["train"]), counts.tolist(),
                    sorted(set(splits["train"]["corpus"])))

    validate(splits)
    splits.save_to_disk(args.output)
    logger.info("Dataset saved to: %s", args.output)


if __name__ == "__main__":
    main()
