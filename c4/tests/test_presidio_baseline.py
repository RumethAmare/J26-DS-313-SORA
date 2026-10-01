"""
Tests for the Presidio wrapper (SO2): label mapping, boundary trimming and
overlap handling. A stand-in analyzer is used, so the test needs neither
Presidio's large model nor network access.
"""
import sys
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import presidio_baseline  # noqa: E402
from presidio_baseline import PRESIDIO_TO_C4, PresidioDetector  # noqa: E402

R = namedtuple("R", "entity_type start end score")


class _StubAnalyzer:
    def __init__(self, results):
        self.results = results

    def analyze(self, text, language, score_threshold):
        assert language == "en", "Presidio is run as English: one language per request"
        return [r for r in self.results if r.score >= score_threshold]


def run(text, results, monkeypatch):
    monkeypatch.setattr(presidio_baseline, "_analyzer", lambda model: _StubAnalyzer(results))
    return [(d.label, d.surface) for d in PresidioDetector()(text)]


def test_entity_types_are_mapped_and_unmapped_ones_dropped(monkeypatch):
    text = "Nimal called 0771234567 from Colombo, NRP x"
    found = run(text, [R("PERSON", 0, 5, 0.85), R("PHONE_NUMBER", 13, 23, 0.75),
                       R("LOCATION", 29, 36, 0.85), R("NRP", 38, 41, 0.85)], monkeypatch)
    assert found == [("PERSON", "Nimal"), ("PHONE", "0771234567"), ("ADDRESS", "Colombo")]


def test_trailing_punctuation_is_trimmed(monkeypatch):
    assert run("call Nimal.", [R("PERSON", 5, 11, 0.85)], monkeypatch) == [("PERSON", "Nimal")]


def test_overlaps_keep_the_most_confident(monkeypatch):
    text = "born 12 August 1995"
    found = run(text, [R("DATE_TIME", 5, 19, 0.85), R("US_SSN", 5, 7, 0.3)], monkeypatch)
    assert found == [("DOB", "12 August 1995")]


def test_mapping_targets_are_c4_labels():
    assert set(PRESIDIO_TO_C4.values()) <= {"PERSON", "PHONE", "EMAIL", "ADDRESS", "ORG",
                                            "DOB", "ACCOUNT", "NIC"}
