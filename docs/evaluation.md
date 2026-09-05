# Evaluation log: problems found and what was done about them

This is the running record of how the DistilHuBERT model was diagnosed and improved after the
course project ended. Every number below was measured with the scripts in this repository and can
be reproduced with the commands given. Dates are when the measurement was taken.

## Setup used for all measurements

- **Test split**: the 288-clip, seed-42, 20 % split of RAVDESS that the original notebook
  evaluated on (`python -m ser.data` recreates it exactly; `--val-size` only carves the validation
  split out of the remaining 80 %).
- **Clean accuracy**: `python -m ser.evaluate --data ravdess_encoded`.
- **Noise robustness**: the same clips with white noise added at a fixed signal-to-noise ratio,
  `python -m ser.evaluate --robustness` (20 dB and 10 dB SNR; 20 dB is quieter than a typical
  laptop microphone in a normal room).
- **Browser path**: clips encoded to opus/webm at 48 kHz, as the web demo's `MediaRecorder` does,
  and decoded by ffmpeg inside the Hugging Face pipeline (96-clip subset, 12 per class).
- Hardware: all evaluations ran on CPU; training runs were done on a Colab T4.

## Problem 1: the demo answered almost only "disgust" and "calm" (2026-09-05)

**Symptom.** On microphone recordings the published model
(`amnesiackid/distilhubert-finetuned-ravdess`, Hub revision `d48dddc`) returned calm or disgust
for nearly everything, regardless of the emotion acted.

**Ruled out first.**

| Hypothesis | Check | Result |
|---|---|---|
| Wrong or corrupted weights on the Hub | Evaluated the current Hub weights and the epoch-16 training checkpoint on the test split | Both 86.1 %; the August re-upload is byte-identical to epoch 16 |
| Bad audio path in the web demo (linear-interpolation resampling) | Same clips through that resampling | 89.6 % on the subset, no collapse |
| Silence at the start/end, or long inputs | Trimmed clips; clips repeated to 10 s | 87.5 % and 86.5 % |
| Codec (opus/webm via ffmpeg) | Same clips through the browser path | 87.5 % vs 89.6 % raw; about 2 points |

**Found.** The model is extremely sensitive to background noise:

| Condition (test split, 288 clips) | Accuracy | What it predicts |
|---|---|---|
| clean | 85.4 % | all eight classes, balanced |
| + white noise, 20 dB SNR | 44.4 % | fearful, disgust, sad, calm; never neutral |
| + white noise, 10 dB SNR | 35.1 % | fearful 124, disgust 87, calm 56 of 288 |

The training log stored on the Hub explains why: the model was fine-tuned on the clean studio
recordings only, with no augmentation and SpecAugment disabled in the base config, and its training
loss reached 0.002. It memorised the corpus. Any real microphone recording sits in the noisy regime
where it collapses onto a couple of classes.

Side finding: the Hub carries the *last* epoch (16, 85.8 % in the log) rather than the best one
(11, 86.8 %), because the notebook never called `push_to_hub()` after `load_best_model_at_end`.

**Solution.** Train for robustness instead of for the clean split (`src/ser/train.py`):

- waveform augmentation on the fly (`src/ser/augment.py`): white or pink noise at 5–30 dB SNR,
  speed perturbation 0.9–1.1, synthetic reverb, random low-pass filter, random 4.5 s crop;
- SpecAugment time masking enabled in the model;
- the CNN feature encoder frozen;
- label smoothing 0.1 and weight decay 0.01;
- best epoch chosen on a new validation split (10 %, stratified, carved out of train), test reported
  separately; `push_to_hub()` called after training.

The old recipe remains available as flags (`--no-augment --no-spec-augment
--no-freeze-feature-encoder --label-smoothing 0 --eval-split test`).

**Result after retraining** (20 epochs on a T4, Hub revision `93da8aa`):

| Condition (test split) | old model `d48dddc` | retrained `93da8aa` |
|---|---|---|
| clean | 85.4 % | 85.8 % |
| + white noise, 20 dB SNR | 44.4 % | **80.6 %** |
| + white noise, 10 dB SNR | 35.1 % | **77.1 %** |
| browser path, clean (subset) | 87.5 % | 83.3 % |
| browser path + 15 dB noise (subset) | 40.6 % | **79.2 %** |

Predictions under noise are spread across all eight classes again. Noise is solved.

## Problem 2: training took 12 s per step (2026-09-05)

**Symptom.** The first Colab run with the new recipe showed 11.8 s per optimiser step, 8 hours for
20 epochs; the original notebook managed 1.6 steps per second.

**Found.** `torchaudio.functional.resample` builds a filter kernel with `sr / gcd(sr, new_sr)`
phases. The speed perturbation picked a random target rate; whenever it was coprime with 16 000
(e.g. 17 391) one call took 5 s instead of 10 ms.

**Solution.** Target rates are rounded to a multiple of 160 (`RESAMPLE_STEP` in `augment.py`).
The whole augmenter now costs about 11 ms per clip; a regression test guards it. A 20-epoch run
takes 30–40 minutes on a T4.

## Problem 3: still calm/disgust on the user's own voice after retraining (2026-09-06)

**Symptom.** With the retrained model the "Game" page still reported disgust (88 %) for a sentence
spoken as fearful, and calm/disgust dominated across attempts.

**Ruled out.** Noise (see the table above), the codec path (about 2 points), and the server
possibly still holding the old weights (`GET /health` now reports the loaded revision so this can be
checked directly).

**Found.** What remains is a speaker gap. RAVDESS is 24 North American actors reading two sentences.
A voice with a different accent, pitch range or speaking style is outside everything the model has
seen, and an out-of-distribution input tends to fall onto whichever classes have the widest
acoustic footprint (calm, disgust). More epochs on RAVDESS cannot fix this; more speakers can.

**Solution (opt-in).** `python -m ser.data --extra-corpora all` mixes in the three corpora the
project's baseline experiment already used, downloaded from Kaggle (`pip install -e ".[train,corpora]"`
plus Kaggle credentials):

| Corpus | Clips | Speakers | Labels covered |
|---|---|---|---|
| CREMA-D | 7 439 | 91, diverse ages and ethnicities | angry disgust fearful happy neutral sad |
| TESS | 2 818 | 2 actresses | all but calm |
| SAVEE | 480 | 4 male British speakers | all but calm |

The extra clips go into the training split only; validation and test remain pure RAVDESS so the
tables above stay comparable. The mixed training set is 11 745 clips with calm at 1.2 % and
surprised at 5 %, so `ser.train` switches on square-root-damped inverse-frequency class weights
automatically (`--class-weights auto|on|off`; measured weights: calm 2.56, surprised 1.24, others
about 0.7). Building the mixed set takes under three minutes; a CPU smoke test of training on it
passes. **The full multi-speaker training run has not been done yet**; its numbers belong in the
table below once it exists.

Side finding fixed on the way: TESS is sampled at 24 414 Hz, which triggers the same resampling
slowdown as problem 2 (3 s per clip). `ser.data.resample` now falls back to FFT resampling for
awkward ratios.

## Problem 4: the browser asked for the microphone on every recording (2026-09-06)

**Found.** The page opened a new `MediaStream` per recording and stopped its tracks afterwards.
Browsers do not persist microphone permission for `file://` pages, so each new stream prompted
again.

**Solution.** One stream is opened on the first recording and reused for the rest of the page's
life (`getMicStream` in `frontend/index.html`); it is released on `pagehide` or if the device goes
away. The demo should be opened through the server (`http://localhost:5000`), where the permission
is remembered across page loads. Capture is now raw (no browser noise suppression, echo cancellation
or automatic gain), because that processing strips low-energy cues and the model handles noise
itself.

## Debugging aids added

- `python -m ser.evaluate --robustness`: the noise table for any checkpoint.
- `GET /health`: `model_loaded` and the loaded Hub `model_revision`.
- `SER_SAVE_UPLOADS=<dir>`: keeps every uploaded recording, named after its prediction, so real
  user recordings can be listened to and re-run with `ser-predict`.
- `python -m ser.train --max-steps 3 --no-fp16`: CPU smoke test of the whole training loop.

## Scoreboard

| Model | clean | 20 dB | 10 dB | notes |
|---|---|---|---|---|
| original notebook, Hub `d48dddc` | 85.4 % | 44.4 % | 35.1 % | collapses under noise |
| RAVDESS + augmentation, Hub `93da8aa` | 85.8 % | 80.6 % | 77.1 % | current published model |
| RAVDESS + CREMA-D + TESS + SAVEE | — | — | — | not trained yet |

Open question for the next round: how the multi-speaker model behaves on the user's own
recordings, which is the only test that matters for the demo. `SER_SAVE_UPLOADS` exists to make
that measurable.
