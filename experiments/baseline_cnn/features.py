"""Hand-crafted features for the baseline 1D CNN: zero-crossing rate, RMS energy and MFCCs.

This reproduces ``extract_features`` from ``notebooks/01_baseline_cnn.ipynb`` so
that the committed weights can be used outside the notebook. A 2.5 s clip at
22 050 Hz with hop length 512 gives 108 frames, and

    108 (ZCR) + 108 (RMS) + 20 * 108 (MFCC) = 2376

values, which is the input length the network was trained with.
"""

from __future__ import annotations

import librosa
import numpy as np

SAMPLE_RATE = 22_050  # librosa's default; the notebook never overrode it
DURATION = 2.5  # seconds of audio used per clip
OFFSET = 0.6  # seconds skipped at the start of each clip
FRAME_LENGTH = 2048
HOP_LENGTH = 512
FEATURE_LENGTH = 2376


def extract_features(
    data: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Stack ZCR, RMS and flattened MFCC frames into one 1-D vector."""
    zcr = np.squeeze(
        librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    )
    rms = np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))
    # The notebook called librosa.feature.mfcc with its defaults (n_mfcc=20, n_fft=2048, hop=512).
    mfcc = np.ravel(librosa.feature.mfcc(y=data, sr=sr).T)
    return np.hstack((zcr, rms, mfcc))


def load_clip(path: str, duration: float = DURATION, offset: float = OFFSET):
    """Load a clip exactly as the training notebook did (mono, 22 050 Hz, 2.5 s from 0.6 s)."""
    return librosa.load(path, duration=duration, offset=offset)


def features_for_file(path: str) -> np.ndarray:
    """Return the 2376-dimensional feature vector for one audio file.

    Clips shorter than 2.5 s produce fewer frames; they are zero-padded to the
    expected length so the model always receives a valid input.
    """
    data, sr = load_clip(path)
    feats = extract_features(data, sr)
    return librosa.util.fix_length(feats, size=FEATURE_LENGTH).astype(np.float32)
