# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech Emotion Recognition (SER) project using a fine-tuned DistilHuBERT model (`amnesiackid/distilhubert-finetuned-ravdess`) trained on the RAVDESS dataset. Detects 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised. Achieves 86.8% accuracy.

## Environment Setup

```bash
pip install -r requirements.txt
```

## Common Commands

```bash
# Run the CLI demo (downloads model from HuggingFace on first run)
python demo.py path/to/audio.wav

# Run the Flask API server (model loads on first request)
python server/app.py

# Run tests (model is mocked — no GPU or internet required)
pytest tests/

# Build and run with Docker
docker build -t ser-api . && docker run -p 5000:5000 ser-api

# Reproduce training pipeline
python src/data.py --output ravdess_encoded
python src/train.py --data ravdess_encoded --epochs 16 --push-to-hub
python src/evaluate.py --output-dir results/
```

## Architecture

Single-model stack — everything runs through `amnesiackid/distilhubert-finetuned-ravdess` via the HuggingFace `pipeline("audio-classification")` API.

**`src/data.py`** — Downloads RAVDESS, resamples to 16 kHz, encodes labels, splits 80/20, applies DistilHuBERT feature extraction, saves to disk. CLI: `--output`, `--test-size`, `--seed`.

**`src/train.py`** — Loads the processed dataset, initialises `AutoModelForAudioClassification` from `ntu-spml/distilhubert`, trains with `Trainer`. CLI: `--data`, `--epochs`, `--batch-size`, `--lr`, `--fp16`/`--no-fp16`, `--push-to-hub`.

**`src/evaluate.py`** — Loads fine-tuned model, runs inference on the test split, prints accuracy + classification report, saves confusion matrix and per-class accuracy plots. CLI: `--model`, `--no-plots`, `--output-dir`.

**`server/app.py`** — Flask REST API with three endpoints (`/predict`, `/health`, `/emotions`). The pipeline is lazily loaded via `get_classifier()` on first request. Audio is saved to a temp file, classified, then cleaned up.

**`demo.py`** — Standalone CLI that loads the same pipeline and prints confidence bars. No server needed.

**`frontend/index.html`** — Self-contained HTML/CSS/JS web UI. Posts audio to `http://127.0.0.1:5000/predict` and highlights the returned emotion. All CSS is inline; `frontend/css/master.css` only sets body background.

## Testing

Tests live in `tests/test_server.py` and use `unittest.mock` to patch `server.app.get_classifier` — no model download needed. `conftest.py` at root ensures the project root is on `sys.path` for imports.

## API

| Method | Endpoint   | Notes                                              |
|--------|------------|----------------------------------------------------|
| POST   | /predict   | multipart `audio` field; returns emotion + probs   |
| GET    | /health    | always 200 if server is up                         |
| GET    | /emotions  | returns ordered list of 8 label strings            |
