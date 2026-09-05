"""Step 3: classify audio files with a trained wav2vec2 + CNN checkpoint.

    python predict.py --checkpoint checkpoints/cnn_classifier.pt clip.wav [more.wav ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF

from model import SAMPLE_RATE, CNNClassifier, hidden_states, load_wav2vec2
from ser.labels import LABELS, NUM_LABELS


def load_waveform(path: str) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.mean(axis=1))
    if sr != SAMPLE_RATE:
        waveform = AF.resample(waveform, sr, SAMPLE_RATE)
    return waveform


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("audio", nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cnn_classifier.pt"))
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, wav2vec2 = load_wav2vec2(device)
    model = CNNClassifier(wav2vec2.config.hidden_size, NUM_LABELS)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device).eval()

    for path in args.audio:
        feats = hidden_states(load_waveform(path).numpy(), processor, wav2vec2, device)
        with torch.no_grad():
            probs = torch.softmax(model(feats.unsqueeze(0).to(device)), dim=-1)[0].cpu()
        order = torch.argsort(probs, descending=True)
        print(f"\n{path}\n  -> {LABELS[order[0]]} ({probs[order[0]]:.1%})")
        for i in order:
            print(f"  {LABELS[i]:10} {probs[i]:6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
