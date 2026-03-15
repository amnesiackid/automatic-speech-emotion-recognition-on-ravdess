# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech Emotion Recognition (SER) student project for "Computational Linguistics Team Laboratory: Phonetics" at the Institute of Natural Language Processing, University of Stuttgart. Detects 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from audio using models trained on the [RAVDESS dataset](https://zenodo.org/records/1188976).

## Environment Setup

```bash
conda env create -f ser_model.yml
conda activate ser_model
```

The DistilHuBERT notebook (`Fine_tune_distill_hubert_on_ravdess_data.ipynb`) is designed for Google Colab — its first cell installs dependencies automatically.

## Running the Server

```bash
python python_model_server.py
```

Flask API starts on `http://localhost:5000`. Endpoints:
- `POST /predict` — upload audio file (multipart `audio` field), returns emotion + confidence + per-class probabilities
- `GET /health` — server and model status
- `GET /emotions` — list supported emotion labels

## Local Inference / Debug Testing

```bash
python emotion_inference_debug.py
```

Interactive menu for single-file or batch testing against custom or local Wav2Vec2 model checkpoints.

## Baseline Model

```bash
python "Baseline model/Predict emotion/predict_emotion.py"
```

Uses the saved weights at `Baseline model/Predict emotion/SER_model.weights.h5`.

## Architecture

Three separate model implementations, each independent:

| Model | Entry Point | Notes |
|---|---|---|
| CNN Baseline | `Baseline model/Baseline_Model.ipynb` + `predict_emotion.py` | Keras/TF CNN, 1D Conv over raw features, input shape (2376, 1) |
| Wav2Vec2 (disabled) | `wav2vec2_model.ipynb` | Requires pre-extracting Wav2Vec2 last hidden states as input features |
| DistilHuBERT (fine-tuned) | `Fine_tune_distill_hubert_on_ravdess_data.ipynb` | Published to HuggingFace: `amnesiackid/distilhubert-finetuned-ravdess` |

**`python_model_server.py`** exposes the Wav2Vec2 model (`superb/wav2vec2-base-superb-er`) as a Flask REST API with CORS enabled. Audio is resampled to 16kHz via librosa, processed through `Wav2Vec2FeatureExtractor`, and classified by `Wav2Vec2ForSequenceClassification`.

**`emotion_inference_debug.py`** contains a custom `Wav2Vec2ForSpeechClassification` class and `EmotionPredictor` wrapper for local fine-tuned model checkpoints, with batch processing and multi-format support (.wav, .mp3, .m4a, .flac, .ogg).

**Front-end** (`front end/homepage.html` + `css/master.css` + `js/main.js`) is a static web UI that targets the Flask backend. `js/main.js` is currently empty.

## Using the Fine-Tuned HuBERT Model Directly

```python
from transformers import pipeline
classifier = pipeline("audio-classification", "amnesiackid/distilhubert-finetuned-ravdess")
result = classifier("/path/to/audio.wav")
```

## Notable Constraints

- RAVDESS audio data and model weights (`.h5`, `.pt`, `.safetensors`) are excluded from the repo via `.gitignore` — they must be obtained separately.
- The Wav2Vec2 integrated model in `wav2vec2_model.ipynb` is described as "disabled" — it requires running Wav2Vec2 feature extraction as a preprocessing step before model input.
