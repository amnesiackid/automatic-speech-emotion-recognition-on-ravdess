"""
Evaluate a fine-tuned model on the RAVDESS test split, optionally under noise.

Reads the DatasetDict written by ser.data (so the split is exactly the one the
model was not trained on), runs batched inference and reports:
  - overall accuracy
  - per-class precision / recall / F1
  - confusion matrix and per-class accuracy plots (PNG)
  - with ``--robustness``: accuracy and the predicted-label histogram again
    after adding white noise at several signal-to-noise ratios, which is the
    condition under which the first model collapsed to a few classes

Usage:
    python -m ser.evaluate --data ravdess_encoded
    python -m ser.evaluate --model ./distilhubert-finetuned-ravdess --robustness
    python -m ser.evaluate --split validation --no-plots
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from ser.augment import add_noise, crop
from ser.data import MAX_DURATION, SAMPLE_RATE
from ser.labels import LABELS
from ser.predict import MODEL_ID as DEFAULT_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model(model_id: str, device: torch.device):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id).to(device).eval()
    id2label = model.config.id2label
    labels = [id2label[i] for i in range(len(id2label))]
    return feature_extractor, model, labels


@torch.no_grad()
def predict_all(feature_extractor, model, labels, waveforms, device, batch_size=8):
    """Return predicted label strings for a list of 16 kHz waveforms."""
    max_samples = int(SAMPLE_RATE * MAX_DURATION)
    preds = []
    for start in range(0, len(waveforms), batch_size):
        batch = [crop(np.asarray(w, dtype=np.float32), max_samples)
                 for w in waveforms[start : start + batch_size]]
        inputs = feature_extractor(
            batch, sampling_rate=SAMPLE_RATE, padding=True,
            return_attention_mask=True, return_tensors="pt",
        ).to(device)
        logits = model(**inputs).logits
        preds.extend(labels[i] for i in logits.argmax(-1).tolist())
        if (start // batch_size) % 10 == 0:
            logger.info("  %d / %d", min(start + batch_size, len(waveforms)), len(waveforms))
    return preds


# ── Metrics ───────────────────────────────────────────────────────────────────

def print_metrics(y_true: list, y_pred: list, title: str = "") -> float:
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'─' * 50}")
    print(f"  {title + ' — ' if title else ''}Accuracy: {accuracy:.1%}")
    print(f"{'─' * 50}\n")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))
    return accuracy


def robustness_report(feature_extractor, model, labels, waveforms, y_true, device,
                      snrs=(20.0, 10.0), batch_size=8) -> dict[str, float]:
    """Accuracy and prediction histogram after adding white noise at each SNR."""
    results = {}
    print(f"\n{'─' * 50}\n  Robustness to additive white noise\n{'─' * 50}")
    for snr in snrs:
        rng = np.random.default_rng(0)
        noisy = [add_noise(np.asarray(w, dtype=np.float32), snr, rng) for w in waveforms]
        preds = predict_all(feature_extractor, model, labels, noisy, device, batch_size)
        acc = accuracy_score(y_true, preds)
        hist = Counter(preds)
        results[f"snr_{int(snr)}dB"] = acc
        print(f"  {snr:4.0f} dB SNR  accuracy {acc:.1%}   predicted: "
              + ", ".join(f"{lbl}={hist.get(lbl, 0)}" for lbl in LABELS))
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: list, y_pred: list, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix — RAVDESS test split", fontsize=14, pad=15)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Confusion matrix saved to: %s", output_path)
    plt.close(fig)


def plot_per_class_accuracy(y_true: list, y_pred: list, accuracy: float,
                            output_path: Path) -> None:
    import matplotlib.pyplot as plt

    class_correct: dict = defaultdict(int)
    class_total: dict = defaultdict(int)
    for true, pred in zip(y_true, y_pred, strict=True):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1
    class_acc = {lbl: class_correct[lbl] / max(class_total[lbl], 1) for lbl in LABELS}

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(LABELS, [class_acc[lbl] for lbl in LABELS],
                  color="steelblue", edgecolor="white")
    ax.axhline(accuracy, color="red", linestyle="--", label=f"Overall ({accuracy:.1%})")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-class accuracy — RAVDESS test split", fontsize=13)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.legend()
    for bar, lbl in zip(bars, LABELS, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{class_acc[lbl]:.0%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Per-class accuracy chart saved to: %s", output_path)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model id or local checkpoint directory")
    parser.add_argument("--data", default="ravdess_encoded",
                        help="DatasetDict written by ser.data")
    parser.add_argument("--split", default="test", help="Split to evaluate (default: test)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=".", help="Where to write the plots")
    parser.add_argument("--no-plots", action="store_true", help="Metrics only")
    parser.add_argument("--robustness", action="store_true",
                        help="Also evaluate with added noise at 20 and 10 dB SNR")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use the first N clips (quick checks)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model: %s", args.model)
    feature_extractor, model, labels = load_model(args.model, device)

    split = load_from_disk(args.data)[args.split]
    if args.limit:
        split = split.select(range(min(args.limit, len(split))))
    logger.info("Evaluating on '%s': %d clips", args.split, len(split))
    waveforms = split["waveform"]
    y_true = [LABELS[i] for i in split["label"]]

    y_pred = predict_all(feature_extractor, model, labels, waveforms, device, args.batch_size)
    accuracy = print_metrics(y_true, y_pred, f"{args.split} split, clean")

    if args.robustness:
        robustness_report(feature_extractor, model, labels, waveforms, y_true, device,
                          batch_size=args.batch_size)

    if not args.no_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
        plot_per_class_accuracy(y_true, y_pred, accuracy, output_dir / "per_class_accuracy.png")


if __name__ == "__main__":
    main()
