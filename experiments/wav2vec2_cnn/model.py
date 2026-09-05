"""Small 1D CNN classifier over frozen wav2vec2 hidden states.

Identical to the ``CNNClassifier`` in ``notebooks/02_wav2vec2_features_cnn.ipynb``.
"""

from __future__ import annotations

import torch
from torch import nn

WAV2VEC2_ID = "facebook/wav2vec2-base"
SAMPLE_RATE = 16_000


class CNNClassifier(nn.Module):
    def __init__(self, feat_dim: int = 768, n_classes: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        return self.net(x.transpose(1, 2))


def load_wav2vec2(device: torch.device):
    """Return ``(processor, model)`` for the frozen wav2vec2 feature extractor."""
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_ID)
    model = Wav2Vec2Model.from_pretrained(WAV2VEC2_ID).to(device).eval()
    return processor, model


@torch.no_grad()
def hidden_states(waveform, processor, wav2vec2, device: torch.device) -> torch.Tensor:
    """Last hidden state of wav2vec2 for one 16 kHz mono waveform: shape (T, 768)."""
    inputs = processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    return wav2vec2(inputs.input_values.to(device)).last_hidden_state.squeeze(0).cpu()
