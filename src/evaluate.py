"""
Evaluate the fine-tuned DistilHuBERT model on the RAVDESS test set.

Loads amnesiackid/distilhubert-finetuned-ravdess from HuggingFace, runs
inference over the test split, and reports:
  - Overall accuracy
  - Per-class precision / recall / F1 (classification report)
  - Confusion matrix (saved to confusion_matrix.png)
  - Per-class accuracy bar chart (saved to per_class_accuracy.png)

Usage:
    python src/evaluate.py
    python src/evaluate.py --model amnesiackid/distilhubert-finetuned-ravdess
    python src/evaluate.py --model ./distilhubert-finetuned-ravdess  # local checkpoint
    python src/evaluate.py --no-plots  # metrics only, no matplotlib
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from datasets import Audio, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "amnesiackid/distilhubert-finetuned-ravdess"
DATASET_ID = "amnesiackid/ravdess-emotion-intensity"
LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
SAMPLE_RATE = 16_000


# ── Inference ─────────────────────────────────────────────────────────────────

def run_predictions(classifier, test_set) -> tuple[list, list]:
    """Return (y_true, y_pred) string lists for the full test set."""
    y_true, y_pred = [], []
    total = len(test_set)

    for i, sample in enumerate(test_set):
        audio_input = {
            "array": sample["audio"]["array"],
            "sampling_rate": sample["audio"]["sampling_rate"],
        }
        result = classifier(audio_input)
        y_true.append(sample["emotion_labels"])
        y_pred.append(result[0]["label"])

        if (i + 1) % 50 == 0:
            logger.info("  %d / %d", i + 1, total)

    logger.info("Predictions complete: %d samples", total)
    return y_true, y_pred


# ── Metrics ───────────────────────────────────────────────────────────────────

def print_metrics(y_true: list, y_pred: list) -> float:
    """Print accuracy and per-class classification report. Returns overall accuracy."""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'─' * 50}")
    print(f"  Overall Accuracy : {accuracy:.1%}")
    print(f"{'─' * 50}\n")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3))
    return accuracy


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: list, y_pred: list, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Confusion Matrix — DistilHuBERT on RAVDESS Test Set",
                 fontsize=14, pad=15)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Confusion matrix saved to: %s", output_path)
    plt.close(fig)


def plot_per_class_accuracy(y_true: list, y_pred: list,
                             accuracy: float, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    class_correct: dict = defaultdict(int)
    class_total: dict = defaultdict(int)
    for true, pred in zip(y_true, y_pred):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1
    class_acc = {lbl: class_correct[lbl] / class_total[lbl] for lbl in LABELS}

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(LABELS, [class_acc[l] for l in LABELS],
                  color="steelblue", edgecolor="white")
    ax.axhline(accuracy, color="red", linestyle="--",
               label=f"Overall ({accuracy:.1%})")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy — DistilHuBERT on RAVDESS Test Set",
                 fontsize=13)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.legend()
    for bar, lbl in zip(bars, LABELS):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{class_acc[lbl]:.0%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Per-class accuracy chart saved to: %s", output_path)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model ID or local checkpoint path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed — must match the seed used in data.py (default: 42)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save plot files (default: current directory)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib plots (metrics only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    logger.info("Loading model: %s", args.model)
    classifier = pipeline("audio-classification", model=args.model)

    logger.info("Loading test set (seed=%d, test_size=%.0f%%)",
                args.seed, args.test_size * 100)
    ds = load_dataset(DATASET_ID)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    test_set = ds["train"].train_test_split(
        test_size=args.test_size, shuffle=True, seed=args.seed
    )["test"]
    logger.info("Test set: %d samples", len(test_set))

    logger.info("Running predictions...")
    y_true, y_pred = run_predictions(classifier, test_set)

    accuracy = print_metrics(y_true, y_pred)

    if not args.no_plots:
        plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
        plot_per_class_accuracy(y_true, y_pred, accuracy,
                                output_dir / "per_class_accuracy.png")


if __name__ == "__main__":
    main()
