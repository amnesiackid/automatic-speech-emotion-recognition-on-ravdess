"""Step 2: train the CNN classifier on cached wav2vec2 features.

Same recipe as the notebook (Adam, lr 5e-4, linear warm-up then cosine decay,
batch size 8, 32 epochs, 80/20 split) but with a fixed seed and the best
epoch's weights saved to ``checkpoints/cnn_classifier.pt``.

    python train.py [--features features] [--epochs 32] [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split

from model import CNNClassifier
from ser.labels import NUM_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class CachedFeatures(Dataset):
    def __init__(self, root: Path):
        self.root = root
        self.labels = torch.load(root / "labels.pt")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        feats = torch.load(self.root / f"{idx}.pt")
        return feats, torch.tensor(self.labels[idx], dtype=torch.long)


def collate(batch):
    feats = pad_sequence([b[0] for b in batch], batch_first=True)  # (B, T_max, D)
    labels = torch.stack([b[1] for b in batch])
    return feats, labels


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * feats.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += feats.size(0)
    return total_loss / n, correct / n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--features", type=Path, default=Path("features"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/cnn_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    dataset = CachedFeatures(args.features)
    n_val = int(len(dataset) * args.val_fraction)
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate)
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    feat_dim = dataset[0][0].shape[-1]
    model = CNNClassifier(feat_dim, NUM_LABELS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        logger.info(
            "Epoch %2d/%d  train loss %.4f acc %.4f | val loss %.4f acc %.4f",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)

    logger.info("Best validation accuracy: %.4f  (saved to %s)", best_acc, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
