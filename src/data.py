"""
Data preparation for RAVDESS speech emotion recognition.

Downloads the RAVDESS dataset from HuggingFace, resamples audio to 16 kHz,
encodes string emotion labels to integers, generates a reproducible train/test
split, applies DistilHuBERT feature extraction, and saves the processed
DatasetDict to disk for use by train.py.

Usage:
    python src/data.py
    python src/data.py --output data/ravdess_encoded --test-size 0.2 --seed 42
"""

import argparse
import logging

from datasets import Audio, DatasetDict, load_dataset
from transformers import AutoFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_ID = "amnesiackid/ravdess-emotion-intensity"
MODEL_ID = "ntu-spml/distilhubert"
SAMPLE_RATE = 16_000
MAX_DURATION = 4.5  # seconds — clips are truncated / zero-padded to this length

ID2LABEL = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# ── Steps ─────────────────────────────────────────────────────────────────────

def load_and_resample(dataset_id: str) -> DatasetDict:
    """Load the raw RAVDESS dataset and resample all audio to SAMPLE_RATE."""
    logger.info("Loading dataset: %s", dataset_id)
    ds = load_dataset(dataset_id)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    logger.info("Loaded %d utterances, resampled to %d Hz", len(ds["train"]), SAMPLE_RATE)
    return ds


def encode_labels(ds: DatasetDict) -> DatasetDict:
    """Map string emotion labels to integer class indices."""
    def _encode(example):
        example["emotion_labels"] = LABEL2ID[example["emotion_labels"]]
        return example

    ds = ds.map(_encode)
    unique = sorted(set(ds["train"]["emotion_labels"]))
    logger.info("Label indices after encoding: %s", unique)
    return ds


def split(ds: DatasetDict, test_size: float, seed: int) -> DatasetDict:
    """Create a stratified train/test split from the single training split."""
    ds = ds["train"].train_test_split(test_size=test_size, shuffle=True, seed=seed)
    logger.info(
        "Split — train: %d  test: %d  (test_size=%.0f%%, seed=%d)",
        len(ds["train"]), len(ds["test"]), test_size * 100, seed,
    )
    return ds


def extract_features(ds: DatasetDict, model_id: str, max_duration: float) -> DatasetDict:
    """Apply DistilHuBERT feature extraction to every audio clip."""
    logger.info("Loading feature extractor: %s", model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id,
        do_normalize=True,
        return_attention_mask=True,
    )
    max_samples = int(feature_extractor.sampling_rate * max_duration)
    logger.info("Max input length: %d samples (%.1f s)", max_samples, max_duration)

    def _preprocess(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        return feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=max_samples,
            truncation=True,
            return_attention_mask=True,
            return_tensors="np",
        )

    logger.info("Extracting features (this may take a few minutes)...")
    ds = ds.map(
        _preprocess,
        remove_columns=["audio", "intensity"],
        batched=True,
        batch_size=100,
    )
    ds = ds.rename_column("emotion_labels", "label")
    logger.info("Feature extraction complete: %s", ds)
    return ds


def validate(ds: DatasetDict) -> None:
    """Sanity-check the processed dataset."""
    sample = ds["train"][0]
    assert set(sample.keys()) == {"label", "input_values", "attention_mask"}, (
        f"Unexpected keys: {sample.keys()}"
    )
    assert isinstance(sample["label"], int), "Label should be int"
    logger.info(
        "Validation passed — keys: %s, label: %s (%s)",
        list(sample.keys()),
        sample["label"],
        ID2LABEL[sample["label"]],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="ravdess_encoded",
                        help="Directory to save the processed dataset (default: ravdess_encoded)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data held out for testing (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splitting (default: 42)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ds = load_and_resample(DATASET_ID)
    ds = encode_labels(ds)
    ds = split(ds, test_size=args.test_size, seed=args.seed)
    ds = extract_features(ds, model_id=MODEL_ID, max_duration=MAX_DURATION)
    validate(ds)

    ds.save_to_disk(args.output)
    logger.info("Dataset saved to: %s", args.output)


if __name__ == "__main__":
    main()
