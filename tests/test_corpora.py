"""File-name parsing for the extra corpora (no downloads)."""

from pathlib import Path

import pytest

from ser import corpora
from ser.labels import LABEL2ID


@pytest.mark.parametrize(
    "name, label, actor, intensity",
    [
        ("1001_DFA_ANG_XX.wav", "angry", 1001, "XX"),
        ("1091_WSI_SAD_HI.wav", "sad", 1091, "HI"),
        ("1042_IEO_FEA_LO.wav", "fearful", 1042, "LO"),
        ("1042_IEO_NEU_XX.wav", "neutral", 1042, "XX"),
    ],
)
def test_parse_crema(name, label, actor, intensity):
    meta = corpora.parse_crema(name)
    assert meta == {"label": LABEL2ID[label], "actor": actor, "intensity": intensity}


@pytest.mark.parametrize(
    "name, label, actor",
    [
        ("OAF_back_angry.wav", "angry", 2001),
        ("YAF_back_ps.wav", "surprised", 2002),
        ("OAF_youth_fear.wav", "fearful", 2001),
        ("YAF_bar_neutral.wav", "neutral", 2002),
    ],
)
def test_parse_tess(name, label, actor):
    meta = corpora.parse_tess(name)
    assert meta["label"] == LABEL2ID[label] and meta["actor"] == actor


@pytest.mark.parametrize(
    "name, label, actor",
    [
        ("DC_a01.wav", "angry", 3001),
        ("JE_sa12.wav", "sad", 3002),
        ("JK_su05.wav", "surprised", 3003),
        ("KL_n30.wav", "neutral", 3004),
        ("KL_h07.wav", "happy", 3004),
    ],
)
def test_parse_savee(name, label, actor):
    meta = corpora.parse_savee(name)
    assert meta["label"] == LABEL2ID[label] and meta["actor"] == actor


@pytest.mark.parametrize("parse", [corpora.parse_crema, corpora.parse_tess, corpora.parse_savee])
def test_unrelated_files_are_skipped(parse):
    for name in ("README.wav", "03-01-06-01-02-02-10.wav", "notes.txt", "1001_DFA_XYZ_XX.wav"):
        assert parse(name) is None


def test_iter_corpus_walks_recursively_and_skips_junk(tmp_path):
    (tmp_path / "AudioWAV").mkdir()
    for name in ("1001_DFA_ANG_XX.wav", "1002_DFA_HAP_XX.wav", "junk.wav"):
        (tmp_path / "AudioWAV" / name).write_bytes(b"")
    found = list(corpora.iter_corpus("crema-d", tmp_path))
    assert [p.name for p, _ in found] == ["1001_DFA_ANG_XX.wav", "1002_DFA_HAP_XX.wav"]
    assert found[0][1]["label"] == LABEL2ID["angry"]


def test_iter_corpus_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(corpora.iter_corpus("tess", Path(tmp_path)))


def test_actor_ids_never_collide_with_ravdess():
    ids = [corpora.parse_crema("1001_DFA_ANG_XX.wav")["actor"],
           corpora.parse_tess("OAF_back_angry.wav")["actor"],
           corpora.parse_savee("DC_a01.wav")["actor"]]
    assert all(i > 24 for i in ids) and len(set(i // 1000 for i in ids)) == 3
