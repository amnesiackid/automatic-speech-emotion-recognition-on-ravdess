"""
Fine-tune DistilHuBERT for speech emotion recognition on RAVDESS.

Loads the preprocessed DatasetDict produced by data.py, initialises
AutoModelForAudioClassification from ntu-spml/distilhubert, and trains
with the HuggingFace Trainer. The best checkpoint is optionally pushed
to the HuggingFace Hub.

Usage:
    python -m ser.train --data ravdess_encoded
    python -m ser.train --data ravdess_encoded --epochs 16 --push-to-hub
    python -m ser.train --data ravdess_encoded --no-fp16  # CPU / MPS
"""

import argparse
import logging

import evaluate
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)

from ser.labels import ID2LABEL, LABEL2ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "ntu-spml/distilhubert"


# ── Metric ────────────────────────────────────────────────────────────────────

def build_compute_metrics():
    """Return a compute_metrics function for HuggingFace Trainer."""
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy.compute(
            predictions=predictions,
            references=eval_pred.label_ids,
        )

    return compute_metrics


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    fp16: bool,
    push_to_hub: bool,
) -> None:
    logger.info("Loading dataset from: %s", data_dir)
    dataset = load_from_disk(data_dir)
    logger.info("Dataset: %s", dataset)

    logger.info("Loading feature extractor: %s", BASE_MODEL_ID)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        BASE_MODEL_ID,
        do_normalize=True,
        return_attention_mask=True,
    )

    logger.info("Loading base model: %s", BASE_MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=len(ID2LABEL),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters — total: %s  trainable: %s", f"{total:,}", f"{trainable:,}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=fp16,
        push_to_hub=push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=feature_extractor,
        compute_metrics=build_compute_metrics(),
    )

    logger.info("Starting training (%d epochs, batch_size=%d, lr=%s)...",
                epochs, batch_size, learning_rate)
    result = trainer.train()
    logger.info("Training complete — loss: %.4f", result.training_loss)

    if push_to_hub:
        logger.info("Pushing best checkpoint to HuggingFace Hub...")
        trainer.push_to_hub()
        logger.info("Model pushed to Hub.")
    else:
        trainer.save_model(output_dir)
        logger.info("Model saved to: %s", output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="ravdess_encoded",
                        help="Path to processed DatasetDict (output of data.py)")
    parser.add_argument("--output-dir", default="distilhubert-finetuned-ravdess",
                        help="Directory for checkpoints and final model")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use mixed-precision training (GPU only)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--push-to-hub", action="store_true", default=False,
                        help="Push best checkpoint to HuggingFace Hub after training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.push_to_hub:
        from huggingface_hub import login
        login()

    train(
        data_dir=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=args.fp16,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
