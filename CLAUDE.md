# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech Emotion Recognition (SER) on the RAVDESS dataset. The headline model is a fine-tuned
DistilHuBERT published on the HuggingFace Hub as `amnesiackid/distilhubert-finetuned-ravdess`
(8 emotions, 86.8% accuracy on a 20% held-out split). Two earlier models (a Keras 1D CNN baseline
and a frozen-wav2vec2 + CNN) are kept under `experiments/` for the write-up's comparison; they are
not used by the CLI, server or web demo.

## Environment Setup

```bash
pip install -e ".[server,dev]"      # inference + Flask API + tests
pip install -e ".[train]"           # add this to run src/ser/data.py, train.py, evaluate.py
pip install -e ".[baseline]"        # add this to run experiments/baseline_cnn
```

Dependencies are declared in `pyproject.toml` only (no requirements.txt). `requires-python >= 3.10`,
tested on 3.11.

## Common Commands

```bash
ser-predict path/to/audio.wav        # CLI (same as: python -m ser.predict ...)
python server/app.py                 # Flask API + web demo on http://localhost:5000
pytest                               # tests; the model is mocked, no GPU or internet needed
ruff check .                         # lint
docker build -t ser-api . && docker run -p 5000:5000 ser-api

# Reproduce the fine-tuning pipeline (GPU recommended)
python -m ser.data --output ravdess_encoded
python -m ser.train --data ravdess_encoded --epochs 20 --push-to-hub
python -m ser.evaluate --data ravdess_encoded --robustness --output-dir results/
```

## Architecture

**`src/ser/labels.py`** is the single source of truth for the label table (`ID2LABEL`, `LABEL2ID`,
`LABELS`, `NUM_LABELS`). Never re-declare the label list elsewhere; import it. Label names are the
RAVDESS ones: `fearful` and `surprised`, not `fear` / `surprise`.

**`src/ser/predict.py`** owns model loading (`get_classifier`, lazy, cached per model id) and
`classify(audio, classifier=None)`, which returns `{"emotion", "confidence", "probabilities"}`.
The CLI `main()` lives here too. `transformers` is imported inside `get_classifier` so importing
the package stays cheap.

**`src/ser/data.py`** downloads the Hub dataset, decodes with soundfile + torchaudio (no torchcodec
dependency, works on Windows), and writes raw 16 kHz waveforms with `train` / `validation` / `test`
splits. The test split is the seed-42 20 % the published model was scored on; validation is carved
out of train (stratified). `--split-by-actor` gives speaker-independent splits (actor id is parsed
from the file name).

**`src/ser/augment.py`** holds the waveform augmentations (noise, speed, reverb, low-pass, crop) and
`WaveformAugmenter`. **`src/ser/train.py`** applies them on the fly through `Dataset.set_transform`
(hence `remove_unused_columns=False`), enables SpecAugment, freezes the feature encoder, uses label
smoothing, and selects the best epoch on the validation split. It supports transformers 4.46+ and
5.x (`warmup_ratio` vs float `warmup_steps`). `--max-steps 3 --no-fp16` is the CPU smoke test.
**`src/ser/evaluate.py`** runs batched inference on a split from `ser.data` and `--robustness`
re-evaluates under added noise at 20 / 10 dB SNR.

Background: the published model collapses to calm / disgust / fearful on noisy microphone audio
(86 % clean, ~50 % at 20 dB SNR, ~34 % at 10 dB). That is a training-data problem, not an inference
bug; the augmentation recipe above is the fix, and any new checkpoint should be judged by the
`--robustness` table, not the clean accuracy alone.

**`server/app.py`** is the Flask app. It imports `classify` and `get_classifier` from `ser.predict`
and serves `frontend/` as static files at `/`. Tests patch `server.app.get_classifier`.

**`frontend/index.html`** is a self-contained page (all CSS/JS inline). `API_BASE` is empty when
served by Flask and falls back to `http://127.0.0.1:5000` when opened from disk. Images live in
`frontend/images/<label>.png` with the canonical label names.

**`experiments/baseline_cnn/`** has the Keras baseline: `features.py` (ZCR + RMS + MFCC, 2376-dim),
`model.py`, `predict.py`, the committed weights and `emotion_categories.json` (alphabetical label
order; reconstructed from the notebook's OneHotEncoder). **`experiments/wav2vec2_cnn/`** has the
script version of the wav2vec2 experiment; no checkpoint is committed.

**`notebooks/`** are the original training notebooks, numbered in project order, with outputs
stripped except for the DistilHuBERT one. Treat them as historical records, not as maintained code.

## Testing

`tests/` uses pytest with `pythonpath = ["src", "."]` from `pyproject.toml`, so no install is
strictly required to run them. The HF pipeline is always mocked (`MagicMock` returning a list of
`{"label", "score"}` dicts). Keep it that way: tests must not download models.

## Conventions

- Large binaries do not go in git. The fine-tuned model lives on the HF Hub; anything over ~10 MB
  should too. `*.pt`, `*.pth`, `*.safetensors`, `*.wav` are git-ignored.
- Docs and figures go in `docs/`; the report is `docs/report.pdf`.
- No LICENSE file yet by the author's decision; RAVDESS itself is CC BY-NC-SA 4.0.
