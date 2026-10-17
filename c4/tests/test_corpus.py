"""
Tests for reading the shared corpus.

C2's output format differs between recordings; every shape below occurs in
the corpus, and C4 must read all of them (FR1).
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus import load_real, load_texts, summary_texts, translation_records  # noqa: E402

RECORD = {"utt_id": "X_u001", "clean_en": "hello", "clean_si": "ආයුබෝවන්"}


@pytest.mark.parametrize("data", [
    {"utterance_records": [RECORD]},
    {"recording_id": "X", "utterance_records": [RECORD]},
    {"recording": "X", "output_version": "2", "utterance_records": [RECORD]},
    {"recording_id": "X", "utterances": [RECORD]},
    [RECORD],
])
def test_every_translation_shape(data):
    assert translation_records(data) == [RECORD]


@pytest.mark.parametrize("data", [
    {"summaries": {"en": {"text": "E"}, "si": {"text": "S"}}},
    {"summaries": {"en": {"text": "E", "status": "ok"}, "si": {"text": "S"}}},
    {"summaries": {"en": "E", "si": "S"}},
    {"summaries": [{"lang": "en", "text": "E"}, {"lang": "si", "text": "S"}]},
    {"recording_id": "X", "summary": "E", "summary_si": "S", "action_items": []},
])
def test_every_summary_shape(data):
    assert summary_texts(data) == {"en": "E", "si": "S"}


def test_missing_summary_language_is_omitted_not_empty():
    assert summary_texts({"summaries": {"en": {"text": "E"}}}) == {"en": "E"}


def _write(root, rel, obj, jsonl=False):
    path = root / "annotations" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if jsonl:
            f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in obj)
        else:
            json.dump(obj, f, ensure_ascii=False)


def test_doc_ids_follow_the_contract(tmp_path):
    _write(tmp_path, "c1/X.transcript.json", {"utterances": [{"utt_id": "X_u001", "text": "t"}]})
    _write(tmp_path, "c2/X.translation.json", {"utterances": [RECORD]})
    _write(tmp_path, "c2/X.summary.json", {"summary": "E"})
    assert set(load_texts("X", tmp_path)) == {
        "X_transcript_u001_v1", "X_c2_u001_en_v1", "X_c2_u001_si_v1", "X_c2_summary_en_v1"}


def test_rows_without_character_offsets_are_dropped(tmp_path):
    """R0061 annotates token offsets only; such rows cannot be scored."""
    good = {"doc_id": "X_transcript_u001_v1", "start_char": 0, "end_char": 1, "label": "PERSON",
            "surface": "t", "entity_id": "X_e001"}
    bad = {"utt_id": "X_u001", "tok_start": 0, "tok_end": 1, "surface": "t", "entity_id": "X_e001"}
    _write(tmp_path, "c4/X.pii.jsonl", [good, bad], jsonl=True)
    assert load_real("X", tmp_path).rows == [good]
