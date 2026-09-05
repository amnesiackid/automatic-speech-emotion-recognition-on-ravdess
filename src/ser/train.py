"""
Fine-tune DistilHuBERT for speech emotion recognition on RAVDESS.

Loads the DatasetDict produced by data.py and trains
AutoModelForAudioClassification from ntu-spml/distilhubert with the
HuggingFace Trainer.

Changes relative to the first (published) model, which was trained on clean
studio clips only and collapses to a few classes on noisy microphone audio:

  * waveform augmentation on the fly (noise, speed, reverb, low-pass; see
    ser.augment) — the single most important change
  * SpecAugment time masking inside the model (disabled in the base config)
  * the convolutional feature encoder is frozen
  * label smoothing and weight decay
  * model selection on a validation split, not on the test split

Usage:
    python -m ser.train --data ravdess_encoded
    python -m ser.train --data ravdess_encoded --epochs 20 --push-to-hub
    python -m ser.train --data ravdess_encoded --no-augment          # reproduce the old recipe
    python -m ser.train --data ravdess_encoded --no-fp16 --max-steps 3  # CPU smoke test
"""

from __future__ import annotations

import argparse
import inspect
import logging

import evaluate
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)

from ser.augment import AugmentConfig, WaveformAugmenter, crop
from ser.data import MAX_DURATION, SAMPLE_RATE
from ser.labels import ID2LABEL, LABEL2ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "ntu-spml/distilhubert"


# ── Data pipeline ─────────────────────────────────────────────────────────────

def make_transform(feature_extractor, augmenter: WaveformAugmenter | None, seed: int):
    """Return a ``set_transform`` function: waveform -> (augmented) model inputs.

    Cropping is random during training (``augmenter`` given) and from the start
    of the clip otherwise, matching how the published model was evaluated.
    """
    max_samples = int(SAMPLE_RATE * MAX_DURATION)
    crop_rng = np.random.default_rng(seed) if augmenter is not None else None

    def transform(batch):
        waves = []
        for wave in batch["waveform"]:
            wave = np.asarray(wave, dtype=np.float32)
            if augmenter is not None:
                wave = augmenter(wave)
            waves.append(crop(wave, max_samples, crop_rng))
        encoded = feature_extractor(
            waves, sampling_rate=SAMPLE_RATE, return_attention_mask=True
        )
        return {
            "input_values": encoded["input_values"],
            "attention_mask": encoded["attention_mask"],
            "labels": batch["label"],
        }

    return transform


class PadCollator:
    """Pad variable-length clips in a batch (zeros + attention mask)."""

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, features):
        batch = self.feature_extractor.pad(
            [{"input_values": f["input_values"], "attention_mask": f["attention_mask"]}
             for f in features],
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return batch


def build_compute_metrics():
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

    return compute_metrics


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    logger.info("Loading dataset from: %s", args.data)
    dataset = load_from_disk(args.data)
    if args.eval_split not in dataset:
        available = [s for s in ("validation", "test") if s in dataset]
        logger.warning("No '%s' split in %s; using '%s' for model selection. "
                       "Re-run ser.data with --val-size > 0 for a proper validation split.",
                       args.eval_split, args.data, available[0])
        args.eval_split = available[0]
    logger.info("Splits: %s", {k: len(v) for k, v in dataset.items()})

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        BASE_MODEL_ID, do_normalize=True, return_attention_mask=True
    )

    augmenter = None
    if args.augment:
        augmenter = WaveformAugmenter(AugmentConfig(), SAMPLE_RATE, seed=args.seed)
        logger.info("Waveform augmentation ON: %s", augmenter.config)
    else:
        logger.info("Waveform augmentation OFF")

    train_ds = dataset["train"]
    eval_ds = dataset[args.eval_split]
    train_ds.set_transform(make_transform(feature_extractor, augmenter, args.seed))
    eval_ds.set_transform(make_transform(feature_extractor, None, args.seed))

    logger.info("Loading base model: %s", BASE_MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=len(ID2LABEL),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        apply_spec_augment=args.spec_augment,
        mask_time_prob=args.mask_time_prob if args.spec_augment else 0.0,
        mask_time_length=10,
    )
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters — total: %s  trainable: %s  (feature encoder %s, SpecAugment %s)",
                f"{total:,}", f"{trainable:,}",
                "frozen" if args.freeze_feature_encoder else "trainable",
                "on" if args.spec_augment else "off")

    # transformers < 5 has `warmup_ratio`; 5.x folded it into `warmup_steps` (float = ratio)
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        warmup_kwargs = {"warmup_ratio": args.warmup_ratio}
    else:
        warmup_kwargs = {"warmup_steps": args.warmup_ratio}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        **warmup_kwargs,
        label_smoothing_factor=args.label_smoothing,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,  # required: inputs are built by set_transform
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(feature_extractor),
        processing_class=feature_extractor,
        compute_metrics=build_compute_metrics(),
    )

    logger.info("Starting training (%d epochs, batch %d x %d, lr %s, eval on '%s') ...",
                args.epochs, args.batch_size, args.grad_accum, args.lr, args.eval_split)
    result = trainer.train()
    logger.info("Training complete — loss: %.4f", result.training_loss)

    best = trainer.evaluate()
    logger.info("Best checkpoint on %s: accuracy %.4f", args.eval_split, best["eval_accuracy"])
    if args.eval_split != "test" and "test" in dataset:
        test_ds = dataset["test"]
        test_ds.set_transform(make_transform(feature_extractor, None, args.seed))
        test = trainer.evaluate(test_ds, metric_key_prefix="test")
        logger.info("Test accuracy (clean clips): %.4f — run ser.evaluate --robustness "
                    "for the noisy-condition numbers", test["test_accuracy"])

    trainer.save_model(args.output_dir)
    logger.info("Model saved to: %s", args.output_dir)
    if args.push_to_hub:
        logger.info("Pushing best checkpoint to the Hub ...")
        trainer.push_to_hub()
        logger.info("Pushed.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="ravdess_encoded",
                        help="Path to the DatasetDict written by ser.data")
    parser.add_argument("--output-dir", default="distilhubert-finetuned-ravdess",
                        help="Directory for checkpoints and the final model")
    parser.add_argument("--eval-split", default="validation", choices=["validation", "test"],
                        help="Split used for per-epoch evaluation and best-model selection")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Stop after this many optimiser steps (smoke tests)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mask-time-prob", type=float, default=0.05,
                        help="SpecAugment: fraction of time steps masked")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="Disable waveform augmentation (the old recipe)")
    parser.add_argument("--no-spec-augment", dest="spec_augment", action="store_false")
    parser.add_argument("--no-freeze-feature-encoder", dest="freeze_feature_encoder",
                        action="store_false", help="Also fine-tune the CNN feature encoder")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Mixed precision (GPU only)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Data-loader workers (augmentation runs there)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push-to-hub", action="store_true", default=False)
    parser.add_argument("--hub-model-id", default=None,
                        help="Hub repo to push to (default: derived from --output-dir)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.push_to_hub:
        from huggingface_hub import login

        login()
    train(args)


if __name__ == "__main__":
    main()
