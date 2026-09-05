"""Tests for the split logic and helpers in ser.data (skipped if `datasets` is absent)."""

import io
import wave

import numpy as np
import pytest

datasets = pytest.importorskip("datasets")

from ser import data  # noqa: E402
from ser.labels import LABELS, NUM_LABELS  # noqa: E402


def _fake_dataset(n_actors: int = 24, clips_per_actor_and_label: int = 1):
    rows = {"waveform": [], "label": [], "actor": [], "intensity": []}
    for actor in range(1, n_actors + 1):
        for label in range(NUM_LABELS):
            for _ in range(clips_per_actor_and_label):
                rows["waveform"].append(np.zeros(1600, dtype=np.float32))
                rows["label"].append(label)
                rows["actor"].append(actor)
                rows["intensity"].append("normal")
    return datasets.Dataset.from_dict(rows, features=data.PROCESSED_FEATURES)


def test_actor_id_from_filename():
    assert data.actor_id("03-01-06-01-02-02-10.wav") == 10
    assert data.actor_id("/some/dir/03-01-01-01-02-01-23.wav") == 23
    assert data.actor_id(None) == -1
    assert data.actor_id("unknown.wav") == -1


def test_decode_waveform_resamples_to_16k_mono():
    sr, seconds = 48_000, 1
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.zeros(sr * seconds * 2, dtype="<i2").tobytes())
    wave_ = data.decode_waveform({"bytes": buf.getvalue(), "path": "x.wav"})
    assert wave_.dtype == np.float32 and wave_.ndim == 1
    assert len(wave_) == pytest.approx(data.SAMPLE_RATE * seconds, rel=0.01)


def test_random_split_sizes_and_stratified_validation():
    ds = _fake_dataset(n_actors=24, clips_per_actor_and_label=2)  # 384 clips
    splits = data.make_splits(ds, test_size=0.25, val_size=0.125, seed=1)
    assert set(splits) == {"train", "validation", "test"}
    assert len(splits["test"]) == 96
    assert len(splits["validation"]) == 48
    assert len(splits["train"]) == 240
    # stratified: every class present in validation with (near) equal counts
    counts = np.bincount(splits["validation"]["label"], minlength=NUM_LABELS)
    assert counts.min() >= 5 and counts.max() <= 7 and counts.sum() == 48


def test_random_split_test_set_is_reproducible():
    ds = _fake_dataset()
    a = data.make_splits(ds, test_size=0.2, val_size=0.1, seed=42)["test"]
    b = data.make_splits(ds, test_size=0.2, val_size=0.0, seed=42)["test"]
    assert a["actor"] == b["actor"] and a["label"] == b["label"]


def test_actor_split_keeps_speakers_disjoint():
    ds = _fake_dataset(n_actors=24)
    splits = data.make_splits(ds, test_size=0.2, val_size=0.1, seed=7, by_actor=True)
    actors = {name: set(part["actor"]) for name, part in splits.items()}
    assert len(actors["test"]) == 5 and len(actors["validation"]) == 2
    assert not actors["train"] & actors["test"]
    assert not actors["train"] & actors["validation"]
    assert not actors["validation"] & actors["test"]
    assert sum(len(p) for p in splits.values()) == len(ds)


def test_label_feature_matches_shared_labels():
    assert data.PROCESSED_FEATURES["label"].names == LABELS
