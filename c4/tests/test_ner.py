"""
Tests for the learned-detector plumbing (SO4).

Trained models are not committed (models/ is git-ignored), so these tests check
the data preparation and the hybrid combination with a stand-in model; the
model's accuracy is measured by the evaluation harness, not here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ner  # noqa: E402
from corpus import Recording, load_synthetic  # noqa: E402
from ner import MODEL_LABELS, HybridDetector, span_examples  # noqa: E402
from rules import Detection  # noqa: E402


def recording(text, rows):
    return Recording("R", {"d": text}, [{"doc_id": "d", **r} for r in rows], None)


def test_only_model_labels_become_training_targets():
    rec = recording("Nimal 0771234567", [
        {"label": "PERSON", "start_char": 0, "end_char": 5, "surface": "Nimal"},
        {"label": "PHONE", "start_char": 6, "end_char": 16, "surface": "0771234567"}])
    examples, _ = span_examples([rec])
    assert examples == [("Nimal 0771234567", [(0, 5, "PERSON")])]


def test_stale_offsets_are_skipped_not_trained_on():
    """R0008 has offsets that no longer match its text; training on them teaches noise."""
    rec = recording("Nimal Perera", [
        {"label": "PERSON", "start_char": 2, "end_char": 7, "surface": "Nimal"}])
    examples, skipped = span_examples([rec])
    assert examples == [("Nimal Perera", [])] and skipped["offset"] == 1


def test_overlapping_spans_keep_the_longer():
    rec = recording("Mr. Nimal Perera", [
        {"label": "PERSON", "start_char": 4, "end_char": 9, "surface": "Nimal"},
        {"label": "PERSON", "start_char": 0, "end_char": 16, "surface": "Mr. Nimal Perera"}])
    examples, skipped = span_examples([rec])
    assert examples[0][1] == [(0, 16, "PERSON")] and skipped["overlap"] == 1


def test_synthetic_train_examples_are_clean():
    examples, skipped = span_examples(load_synthetic("train"))
    assert skipped == {"offset": 0, "overlap": 0}
    labels = {label for _, spans in examples for _, _, label in spans}
    assert labels <= set(MODEL_LABELS) and "PERSON" in labels


class _StubModel:
    def __init__(self, dets):
        self.dets = dets

    def __call__(self, text, preceding=""):
        return self.dets


def test_hybrid_lets_rules_win_where_they_overlap(monkeypatch):
    text = "Nimal, NIC eka 953201456V"
    stub = _StubModel([Detection(0, 5, "PERSON", "Nimal", role="PRIVATE_INDIVIDUAL"),
                       Detection(15, 25, "ORG", "953201456V", role="ORGANISATION")])
    monkeypatch.setattr(ner, "ModelDetector", lambda name: stub)
    found = HybridDetector("x")(text)
    assert [(d.label, d.surface) for d in found] == [("PERSON", "Nimal"), ("NIC", "953201456V")]
