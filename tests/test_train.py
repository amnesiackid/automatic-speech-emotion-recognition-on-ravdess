"""Unit tests for the pure helpers in ser.train (skipped without the training extras)."""

import numpy as np
import pytest

pytest.importorskip("evaluate")
pytest.importorskip("transformers")

from ser import train  # noqa: E402


def test_class_weights_upweight_rare_classes_and_average_one():
    labels = [0] * 100 + [1] * 25 + [2] * 4  # 3 classes, very imbalanced
    w = train.class_weights_from(labels, num_classes=3)
    assert w.shape == (3,)
    assert w[2] > w[1] > w[0]
    assert float(w.mean()) == pytest.approx(1.0)
    # square-root damping: a 25x count ratio gives a 5x weight ratio
    assert float(w[2] / w[0]) == pytest.approx((100 / 4) ** 0.5)


def test_class_weights_handle_missing_classes():
    w = train.class_weights_from([0, 0, 1], num_classes=4)
    assert np.isfinite(w.numpy()).all() and (w > 0).all()


def test_parse_args_defaults_match_the_documented_recipe():
    args = train.parse_args([])
    assert args.augment and args.spec_augment and args.freeze_feature_encoder
    assert args.eval_split == "validation" and args.class_weights == "auto"
    assert args.label_smoothing == 0.1 and args.epochs == 20
    old = train.parse_args(["--no-augment", "--no-spec-augment", "--no-freeze-feature-encoder",
                            "--label-smoothing", "0", "--eval-split", "test"])
    assert not (old.augment or old.spec_augment or old.freeze_feature_encoder)
