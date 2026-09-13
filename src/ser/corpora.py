"""Extra training corpora: CREMA-D, TESS and SAVEE, mapped onto the RAVDESS label set.

RAVDESS has 24 North American actors. A model trained on it alone learns those
24 voices; on a new speaker with a different accent it tends to fall back to a
couple of classes. These three corpora add 97 speakers, more recording chains
and more speaking styles. They were already used by the project's baseline
notebook via the same Kaggle mirrors.

Each corpus has its own file-name convention; the ``parse_*`` functions turn a
file name into ``{"label": int, "actor": int, "intensity": str}`` or ``None``
for files that are not labelled clips. Actor ids are offset per corpus so they
never collide with RAVDESS actors 1-24.

Label coverage (RAVDESS names):
    CREMA-D  angry disgust fearful happy neutral sad            (no calm, surprised)
    TESS     angry disgust fearful happy neutral sad surprised  (no calm)
    SAVEE    angry disgust fearful happy neutral sad surprised  (no calm)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

from ser.labels import LABEL2ID

logger = logging.getLogger(__name__)

KAGGLE_DATASETS = {
    "crema-d": "ejlok1/cremad",
    "tess": "ejlok1/toronto-emotional-speech-set-tess",
    "savee": "ejlok1/surrey-audiovisual-expressed-emotion-savee",
}
CORPORA = tuple(KAGGLE_DATASETS)

_CREMA_EMOTIONS = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fearful",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}
_CREMA_RE = re.compile(r"^(\d{4})_[A-Z]{3}_([A-Z]{3})_([A-Z]{2})\.wav$", re.IGNORECASE)

_TESS_EMOTIONS = {
    "angry": "angry", "disgust": "disgust", "fear": "fearful", "happy": "happy",
    "neutral": "neutral", "sad": "sad", "ps": "surprised",
}
_TESS_RE = re.compile(r"^(OAF|YAF)_[A-Za-z]+_([a-z]+)\.wav$", re.IGNORECASE)
_TESS_ACTORS = {"OAF": 2001, "YAF": 2002}

_SAVEE_EMOTIONS = {
    "a": "angry", "d": "disgust", "f": "fearful", "h": "happy",
    "n": "neutral", "sa": "sad", "su": "surprised",
}
_SAVEE_RE = re.compile(r"^(DC|JE|JK|KL)_(sa|su|a|d|f|h|n)\d{2}\.wav$", re.IGNORECASE)
_SAVEE_ACTORS = {"DC": 3001, "JE": 3002, "JK": 3003, "KL": 3004}


def parse_crema(name: str) -> dict | None:
    """``1001_DFA_ANG_XX.wav`` -> actor 1001, angry, intensity XX/LO/MD/HI."""
    m = _CREMA_RE.match(name)
    if not m or m.group(2).upper() not in _CREMA_EMOTIONS:
        return None
    return {
        "label": LABEL2ID[_CREMA_EMOTIONS[m.group(2).upper()]],
        "actor": 1000 + int(m.group(1)) - 1000,  # CREMA-D actors are numbered 1001-1091
        "intensity": m.group(3).upper(),
    }


def parse_tess(name: str) -> dict | None:
    """``OAF_back_angry.wav`` / ``YAF_back_ps.wav`` -> two actresses, seven emotions."""
    m = _TESS_RE.match(name)
    if not m or m.group(2).lower() not in _TESS_EMOTIONS:
        return None
    return {
        "label": LABEL2ID[_TESS_EMOTIONS[m.group(2).lower()]],
        "actor": _TESS_ACTORS[m.group(1).upper()],
        "intensity": "normal",
    }


def parse_savee(name: str) -> dict | None:
    """``DC_a01.wav`` / ``JK_sa12.wav`` -> four speakers, seven emotions."""
    m = _SAVEE_RE.match(name)
    if not m:
        return None
    return {
        "label": LABEL2ID[_SAVEE_EMOTIONS[m.group(2).lower()]],
        "actor": _SAVEE_ACTORS[m.group(1).upper()],
        "intensity": "normal",
    }


PARSERS = {"crema-d": parse_crema, "tess": parse_tess, "savee": parse_savee}


def iter_corpus(name: str, root: Path) -> Iterator[tuple[Path, dict]]:
    """Yield ``(path, metadata)`` for every labelled clip of ``name`` under ``root``."""
    parse = PARSERS[name]
    seen = 0
    for path in sorted(root.rglob("*.wav")):
        meta = parse(path.name)
        if meta is not None:
            seen += 1
            yield path, meta
    if seen == 0:
        raise FileNotFoundError(f"No {name} clips found under {root}")


def download(name: str) -> Path:
    """Download (or reuse the cached copy of) a corpus from Kaggle via kagglehub.

    Needs Kaggle credentials: a ``~/.kaggle/kaggle.json`` or the
    ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables. On Colab,
    ``kagglehub.login()`` prompts for them.
    """
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The extra corpora are optional and need kagglehub: "
            'pip install -e ".[corpora]" (and Kaggle credentials) to use --extra-corpora'
        ) from exc
    logger.info("Fetching %s from Kaggle (%s) ...", name, KAGGLE_DATASETS[name])
    return Path(kagglehub.dataset_download(KAGGLE_DATASETS[name]))
