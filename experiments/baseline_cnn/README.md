# Experiment 1: baseline 1D CNN on hand-crafted features

The first model of the project, kept for comparison with the DistilHuBERT
model in `src/ser`. It is **not** what the CLI, server or web demo use.

| | |
|---|---|
| Input | 2376-dim vector: zero-crossing rate + RMS energy + 20 MFCCs, 2.5 s of audio at 22 050 Hz |
| Training data | RAVDESS + CREMA-D + TESS + SAVEE (Kaggle mirrors), with noise / pitch augmentation |
| Architecture | 3 × (Conv1D → BatchNorm → MaxPool → Dropout) → GlobalAveragePooling → Dense(32) → Dense(8) |
| Framework | TensorFlow / Keras |
| Reported accuracy | 56.7 % in the write-up; 56.5 % in the notebook's final evaluation |
| Training code | [`notebooks/01_baseline_cnn.ipynb`](../../notebooks/01_baseline_cnn.ipynb) |

## Files

- `SER_model.weights.h5`: the trained weights (665 KB), saved by the notebook.
- `emotion_categories.json`: label order of the output layer. The notebook never
  saved this file; it was reconstructed from the notebook's `OneHotEncoder`, which
  orders categories alphabetically: `angry, calm, disgust, fear, happy, neutral, sad, surprise`.
- `features.py`: the feature extraction copied from the notebook.
- `model.py`: the architecture copied from the notebook.
- `predict.py`: command-line prediction using the files above.

## Run it

```bash
pip install -e ".[baseline]"
cd experiments/baseline_cnn
python predict.py path/to/clip.wav
```

## Notes

- The original `predict_emotion.py` fed 2376 raw audio samples (about 0.1 s) into
  the network instead of the 2376 engineered features, so its predictions were
  meaningless. `predict.py` here uses the same features as training.
- This experiment names two classes `fear` and `surprise`; the main model uses the
  RAVDESS names `fearful` and `surprised`.
- The datasets are downloaded through `kagglehub` inside the notebook and need a
  Kaggle account.
