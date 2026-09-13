# Notebooks

The original training notebooks, in the order the project developed. They are
kept as the record of how each model in the write-up was produced; the
maintained, script-based pipeline for the final model lives in `src/ser/`.

| Notebook | Model | Where it ran | Outputs |
|---|---|---|---|
| `01_baseline_cnn.ipynb` | 1D CNN on ZCR + RMS + MFCC features, four datasets | Kaggle / Colab GPU, Python 3.12 | stripped (was 2.4 MB) |
| `02_wav2vec2_features_cnn.ipynb` | frozen wav2vec2-base features + CNN | university GPU server, Python 3.13 | stripped |
| `03_distilhubert_finetune.ipynb` | fine-tuned DistilHuBERT (the published model) | Google Colab, Python 3.11 | kept: training log, 86.8 % eval accuracy |

Notes:

- `01` downloads its data with `kagglehub` and needs a Kaggle account.
- `02` was run on a machine where audio decoding went through `torchcodec`; the
  script version in `experiments/wav2vec2_cnn/` uses the standard `datasets`
  audio decoding instead.
- `03` installs `transformers` from the GitHub main branch in its first cell,
  which is not reproducible; `src/ser/train.py` pins a released version.
- The `pip` cells at the top of each notebook are Colab conveniences. For local
  runs install the `train` (or `baseline`) extra from `pyproject.toml` instead.
