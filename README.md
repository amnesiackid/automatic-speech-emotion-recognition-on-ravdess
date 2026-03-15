# Speech Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-distilhubert--finetuned--ravdess-yellow)](https://huggingface.co/amnesiackid/distilhubert-finetuned-ravdess)
[![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey.svg)](https://flask.palletsprojects.com)
[![Dataset](https://img.shields.io/badge/Dataset-RAVDESS-green.svg)](https://zenodo.org/records/1188976)

Automatically detect emotion in speech using a fine-tuned [DistilHuBERT](https://huggingface.co/ntu-spml/distilhubert) model trained on the [RAVDESS](https://zenodo.org/records/1188976) dataset. Achieves **86.8% accuracy** across 8 emotion classes.

> Academic project — Computational Linguistics Team Laboratory: Phonetics
> Institute of Natural Language Processing, University of Stuttgart

---

## Emotions

`neutral` · `calm` · `happy` · `sad` · `angry` · `fearful` · `disgust` · `surprised`

---

## Architecture

```
Audio Input (.wav / .mp3 / ...)
        │
        ▼
┌───────────────────────────────┐
│  DistilHuBERT Feature Encoder │  — pre-trained, frozen
│  (7 transformer layers)       │
└───────────────┬───────────────┘
                │  hidden states (768-dim sequence)
                ▼
┌───────────────────────────────┐
│  Mean Pooling                 │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Classification Head          │  — fine-tuned on RAVDESS
│  Linear(768 → 8)              │
└───────────────┬───────────────┘
                │
                ▼
        Emotion Label + Confidence
```

---

## Results

| Metric   | Value                              |
|----------|------------------------------------|
| Accuracy | **86.8%**                          |
| Dataset  | RAVDESS (1440 utterances, 8 classes) |
| Model    | [amnesiackid/distilhubert-finetuned-ravdess](https://huggingface.co/amnesiackid/distilhubert-finetuned-ravdess) |

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run the demo

```bash
python demo.py path/to/audio.wav
```

Example output:
```
Loading amnesiackid/distilhubert-finetuned-ravdess...
Classifying: speech.wav

Results:
  happy       85.0%  █████████████████████████
  neutral     10.0%  ███
  sad          5.0%  █
```

### 3. Start the API server

```bash
python server/app.py
```

Server starts on `http://localhost:5000`. The model is loaded on first request.

#### API Endpoints

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| `POST` | `/predict` | Classify emotion in an audio file  |
| `GET`  | `/health`  | Server health check                |
| `GET`  | `/emotions`| List supported emotion labels      |

**Predict example:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "audio=@speech.wav"
```

Response:
```json
{
  "emotion": "happy",
  "confidence": 0.85,
  "probabilities": {
    "happy": 0.85,
    "neutral": 0.10,
    "sad": 0.05,
    ...
  }
}
```

### 4. Deploy with Docker

```bash
docker build -t ser-api .
docker run -p 5000:5000 ser-api
```

---

## Use the Model Directly

```python
from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="amnesiackid/distilhubert-finetuned-ravdess",
)

result = classifier("speech.wav")
# [{'label': 'happy', 'score': 0.85}, ...]
```

---

## Reproducing the Model

```bash
# 1 — Download RAVDESS, extract features, save to disk
python src/data.py --output ravdess_encoded

# 2 — Fine-tune (GPU recommended; omit --push-to-hub to save locally)
python src/train.py --data ravdess_encoded --epochs 16 --push-to-hub

# 3 — Evaluate and generate plots
python src/evaluate.py --output-dir results/
```

---

## Project Structure

```
├── demo.py                # Standalone CLI demo
├── requirements.txt
├── Dockerfile
├── src/
│   ├── data.py            # Data prep: download, preprocess, save
│   ├── train.py           # Fine-tuning with HuggingFace Trainer
│   └── evaluate.py        # Metrics, confusion matrix, per-class accuracy
├── server/
│   └── app.py             # Flask REST API
├── frontend/
│   └── index.html         # Web interface
├── tests/
│   └── test_server.py
└── docs/
    └── Automatic Speech Emotion Recognition with Machine Learning Methods.pdf
```

---

## Development

```bash
# Run tests (no GPU or internet required — model is mocked)
pytest tests/

# Run server in debug mode
FLASK_DEBUG=1 python server/app.py
```
