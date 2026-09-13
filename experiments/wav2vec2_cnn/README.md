# Experiment 2: frozen wav2vec2 features + 1D CNN

The intermediate model of the project. A pre-trained `facebook/wav2vec2-base` is
used as a frozen feature extractor; its last hidden state (a `(T, 768)` sequence
per clip) is fed to a small CNN classifier trained from scratch. The write-up
calls this the "disabled" model because it cannot run without the separate
wav2vec2 forward pass.

| | |
|---|---|
| Input | last hidden state of frozen wav2vec2-base, 16 kHz audio |
| Training data | RAVDESS (1440 clips, `amnesiackid/ravdess-emotion-intensity`), 80/20 split |
| Architecture | Conv1d(768→128) → MaxPool → Conv1d(128→256) → AdaptiveAvgPool → Linear(256→8) |
| Reported accuracy | 67.2 % in the write-up |
| Training code | [`notebooks/02_wav2vec2_features_cnn.ipynb`](../../notebooks/02_wav2vec2_features_cnn.ipynb) |

## Status

**No trained checkpoint is included.** The notebook saved
`checkpoints/cnn_classifier.pth` on a university server and it was never
committed. The stored notebook output only covers the first 8 epochs
(best validation accuracy 49.3 %), so the 67.2 % figure cannot be verified
from this repository until the experiment is re-run.

The scripts below are a faithful script version of the notebook, with a fixed
seed and the best epoch saved, so the experiment can be reproduced on a GPU
machine or Colab:

```bash
pip install -e ".[train]"
cd experiments/wav2vec2_cnn
python extract_features.py            # ~1440 x (T, 768) tensors -> features/
python train.py --epochs 32 --seed 42 # -> checkpoints/cnn_classifier.pt
python predict.py clip.wav            # uses the checkpoint
```

`features/` and `checkpoints/` are git-ignored. If a checkpoint is produced it
is a few MB and can be committed here or uploaded to the HuggingFace Hub.
