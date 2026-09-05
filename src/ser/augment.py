"""Waveform augmentations for training on 16 kHz mono float32 audio.

Why this exists: the first model was fine-tuned on clean studio recordings only.
On the RAVDESS test split it scores 86 %, but adding background noise at
20 dB SNR halves its accuracy and the predictions collapse onto a couple of
classes (calm, disgust, fearful), which is what users see with microphone
recordings. Every function here simulates a real-world degradation and is
applied on the fly during training so the model sees a different version of
each clip every epoch.

All functions take and return 1-D ``np.float32`` arrays. The augmenter is
deterministic for a given seed (per data-loader worker).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio.functional as AF

SAMPLE_RATE = 16_000


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def add_noise(
    x: np.ndarray, snr_db: float, rng: np.random.Generator, color: str = "white"
) -> np.ndarray:
    """Add white or pink noise scaled to the requested signal-to-noise ratio."""
    n = rng.standard_normal(len(x)).astype(np.float32)
    if color == "pink":
        spec = np.fft.rfft(n)
        freqs = np.fft.rfftfreq(len(n))
        spec[1:] /= np.sqrt(freqs[1:] / freqs[1])  # 1/f power spectrum
        n = np.fft.irfft(spec, n=len(n)).astype(np.float32)
    scale = _rms(x) / (_rms(n) * 10 ** (snr_db / 20))
    return (x + n * scale).astype(np.float32)


def speed_perturb(x: np.ndarray, factor: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Speed up (factor > 1) or slow down the clip; pitch shifts with it, as in Kaldi."""
    new_sr = int(round(sr / factor))
    if new_sr == sr:
        return x
    y = AF.resample(torch.from_numpy(np.ascontiguousarray(x)), orig_freq=sr, new_freq=new_sr)
    return y.numpy().astype(np.float32)


def add_reverb(
    x: np.ndarray, rt60: float, rng: np.random.Generator, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Convolve with a synthetic room impulse response (exponentially decaying noise)."""
    n = max(int(rt60 * sr), 2)
    t = np.arange(n, dtype=np.float32) / sr
    rir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.908 * t / rt60)  # -60 dB at rt60
    rir[0] = 1.0  # direct path
    rir /= np.sqrt(np.sum(rir**2))
    size = len(x) + n - 1
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(rir, size), size)[: len(x)]
    y = y * (_rms(x) / _rms(y))
    return y.astype(np.float32)


def low_pass(
    x: np.ndarray, cutoff_hz: float, sr: int = SAMPLE_RATE, transition_hz: float = 500.0
) -> np.ndarray:
    """Remove content above ``cutoff_hz`` (cheap microphone / codec simulation)."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    gain = np.clip((cutoff_hz + transition_hz - freqs) / transition_hz, 0.0, 1.0)
    return np.fft.irfft(spec * gain, len(x)).astype(np.float32)


def crop(x: np.ndarray, max_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Cut a clip to ``max_samples``: a random window when ``rng`` is given, else the start."""
    if len(x) <= max_samples:
        return x
    start = int(rng.integers(0, len(x) - max_samples + 1)) if rng is not None else 0
    return x[start : start + max_samples]


@dataclass
class AugmentConfig:
    p_noise: float = 0.5
    snr_db: tuple[float, float] = (5.0, 30.0)
    p_pink: float = 0.5  # share of noise draws that are pink rather than white
    p_speed: float = 0.3
    speed: tuple[float, float] = (0.9, 1.1)
    p_reverb: float = 0.3
    rt60: tuple[float, float] = (0.2, 0.6)
    p_lowpass: float = 0.3
    cutoff_hz: tuple[float, float] = (3000.0, 7000.0)


class WaveformAugmenter:
    """Apply a random subset of the augmentations above to one waveform."""

    def __init__(
        self,
        config: AugmentConfig | None = None,
        sample_rate: int = SAMPLE_RATE,
        seed: int | None = None,
    ):
        self.config = config or AugmentConfig()
        self.sample_rate = sample_rate
        self.seed = seed
        self._rng: np.random.Generator | None = None

    @property
    def rng(self) -> np.random.Generator:
        """Lazily created so each data-loader worker gets its own stream."""
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            offset = worker.id if worker is not None else 0
            self._rng = np.random.default_rng(None if self.seed is None else self.seed + offset)
        return self._rng

    def __call__(self, x: np.ndarray) -> np.ndarray:
        c, rng = self.config, self.rng
        x = np.asarray(x, dtype=np.float32)
        if rng.random() < c.p_speed:
            x = speed_perturb(x, rng.uniform(*c.speed), self.sample_rate)
        if rng.random() < c.p_reverb:
            x = add_reverb(x, rng.uniform(*c.rt60), rng, self.sample_rate)
        if rng.random() < c.p_lowpass:
            x = low_pass(x, rng.uniform(*c.cutoff_hz), self.sample_rate)
        if rng.random() < c.p_noise:
            color = "pink" if rng.random() < c.p_pink else "white"
            x = add_noise(x, rng.uniform(*c.snr_db), rng, color)
        return x
