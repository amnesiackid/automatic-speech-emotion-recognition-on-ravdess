"""Tests for the waveform augmentations (numpy + torchaudio only, no model)."""

import numpy as np
import pytest

from ser import augment

SR = 16_000


@pytest.fixture
def tone():
    t = np.arange(SR * 2) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _snr_db(clean, noisy):
    noise = noisy - clean
    return 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))


def test_add_noise_hits_requested_snr(tone):
    rng = np.random.default_rng(0)
    for color in ("white", "pink"):
        noisy = augment.add_noise(tone, 15.0, rng, color)
        assert noisy.dtype == np.float32 and noisy.shape == tone.shape
        assert abs(_snr_db(tone, noisy) - 15.0) < 0.5


def test_speed_perturb_changes_length(tone):
    faster = augment.speed_perturb(tone, 1.1, SR)
    slower = augment.speed_perturb(tone, 0.9, SR)
    assert len(faster) == pytest.approx(len(tone) / 1.1, rel=0.01)
    assert len(slower) == pytest.approx(len(tone) / 0.9, rel=0.01)
    assert augment.speed_perturb(tone, 1.0, SR) is tone


def test_reverb_keeps_length_and_level(tone):
    out = augment.add_reverb(tone, 0.4, np.random.default_rng(0), SR)
    assert out.shape == tone.shape and out.dtype == np.float32
    assert np.sqrt(np.mean(out**2)) == pytest.approx(np.sqrt(np.mean(tone**2)), rel=0.05)
    assert not np.allclose(out, tone)


def test_low_pass_removes_high_frequencies():
    t = np.arange(SR) / SR
    low = np.sin(2 * np.pi * 500 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 6000 * t).astype(np.float32)
    out = augment.low_pass(low + high, cutoff_hz=3000.0, sr=SR)
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    assert spec[np.argmin(abs(freqs - 6000))] < 0.01 * spec[np.argmin(abs(freqs - 500))]


def test_crop_random_and_deterministic(tone):
    assert augment.crop(tone, len(tone) + 1) is tone
    head = augment.crop(tone, SR)
    assert len(head) == SR and np.array_equal(head, tone[:SR])
    rng = np.random.default_rng(3)
    window = augment.crop(tone, SR, rng)
    assert len(window) == SR


def test_augmenter_is_seeded_and_always_augments_something(tone):
    cfg = augment.AugmentConfig(p_noise=1.0, p_speed=0.0, p_reverb=0.0, p_lowpass=0.0)
    a = augment.WaveformAugmenter(cfg, SR, seed=1)
    b = augment.WaveformAugmenter(cfg, SR, seed=1)
    out_a, out_b = a(tone), b(tone)
    assert np.array_equal(out_a, out_b)
    assert out_a.dtype == np.float32 and not np.allclose(out_a, tone)


def test_augmenter_can_be_a_no_op(tone):
    cfg = augment.AugmentConfig(p_noise=0.0, p_speed=0.0, p_reverb=0.0, p_lowpass=0.0)
    assert np.array_equal(augment.WaveformAugmenter(cfg, SR, seed=0)(tone), tone)
