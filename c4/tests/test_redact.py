"""
Tests for redaction and re-identification.

The guarantees that matter: nothing known survives redaction, one entity has
one placeholder everywhere, the map reverses redaction exactly, and the map is
never written anywhere git could commit it.
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus import C4_ROOT, load_synthetic  # noqa: E402
from redact import (REID_DIR, Redaction, leak_check, redact_recording,  # noqa: E402
                    restore, write_reid_map)
from rules import Detection  # noqa: E402

SYNTHETIC_TEST = load_synthetic("test")


def test_spoken_and_written_forms_share_one_placeholder():
    texts = {"a_transcript_u001_v1": "number eka 0 7 7 1 2 3 4 5 6 7.",
             "a_c2_summary_en_v1": "The contact number is 0771234567."}
    out = redact_recording(texts)
    assert out.texts["a_transcript_u001_v1"] == "number eka [PHONE_1]."
    assert out.texts["a_c2_summary_en_v1"] == "The contact number is [PHONE_1]."


def test_distinct_entities_get_distinct_placeholders():
    out = redact_recording({"d": "mobile 0771234567, office 0112345678 ta call karanna."})
    assert "[PHONE_1]" in out.texts["d"] and "[PHONE_2]" in out.texts["d"]


def test_nic_with_and_without_its_letter_is_one_entity():
    texts = {"d1": "NIC eka 953201456V.", "d2": "NIC eka 9 5 3 2 0 1 4 5 6."}
    out = redact_recording(texts)
    assert out.texts["d1"] == "NIC eka [NIC_1]." and out.texts["d2"] == "NIC eka [NIC_1]."


def test_organisation_hotline_stays_visible():
    out = redact_recording({"d": "Dialog hotline eka 0117654321."})
    assert out.texts["d"] == "Dialog hotline eka 0117654321."
    assert out.entities == []


def test_propagation_redacts_a_mention_the_detector_missed():
    """The second document has no keyword and a detector that sees nothing."""
    texts = {"d1": "account number eka 123456789012.", "d2": "ref 123456789012 hari."}

    def detect_first_only(text, preceding):
        return [Detection(19, 31, "ACCOUNT", "123456789012", "123456789012")] \
            if text.startswith("account") else []

    without = redact_recording(texts, detect=detect_first_only, propagate=False)
    assert "123456789012" in without.texts["d2"] and without.leaks
    with_prop = redact_recording(texts, detect=detect_first_only)
    assert with_prop.texts["d2"] == "ref [ACCOUNT_1] hari." and not with_prop.leaks


def test_known_name_overrides_a_keep_visible_label_elsewhere():
    """The model calls the customer's name ORG in one sentence; it must still be redacted."""
    texts = {"d1": "mage nama Nimal Perera.", "d2": "Thank you Nimal Perera, goodbye."}

    def detect(text, preceding):
        i = text.find("Nimal Perera")
        label, role = ("PERSON", "PRIVATE_INDIVIDUAL") if text.startswith("mage") else ("ORG", "ORGANISATION")
        return [Detection(i, i + 12, label, "Nimal Perera", role=role)]

    out = redact_recording(texts, detect=detect, classify_roles=False)
    assert out.texts["d2"] == "Thank you [PERSON_1], goodbye." and not out.leaks


def test_sinhala_name_with_a_case_ending_is_still_redacted():
    """නිමල්ට is 'to Nimal': the name is redacted, the grammatical ending kept."""
    texts = {"d1": "මගේ නම නිමල් පෙරේරා.", "d2": "මම නිමල් පෙරේරාට කතා කළා."}

    def detect(text, preceding):
        if text.startswith("මගේ"):
            return [Detection(7, 19, "PERSON", "නිමල් පෙරේරා", role="PRIVATE_INDIVIDUAL")]
        return []

    out = redact_recording(texts, detect=detect, classify_roles=False)
    assert out.texts["d2"] == "මම [PERSON_1]ට කතා කළා." and not out.leaks


def test_propagation_does_not_match_inside_a_longer_number():
    texts = {"d1": "account number eka 123456789012.", "d2": "9123456789012345"}

    def detect_first_only(text, preceding):
        return [Detection(19, 31, "ACCOUNT", "123456789012", "123456789012")] \
            if text.startswith("account") else []

    out = redact_recording(texts, detect=detect_first_only)
    assert out.texts["d2"] == "9123456789012345"


def test_leak_check_catches_a_respaced_value():
    result = Redaction("r", {"d": "call 077 123 4567"}, [
        {"placeholder": "[PHONE_1]", "label": "PHONE", "role": "", "value": "0771234567",
         "mentions": [{"surface": "0771234567"}]}], {})
    assert leak_check(result) == [{"placeholder": "[PHONE_1]", "doc_id": "d"}]


@pytest.mark.parametrize("rec", SYNTHETIC_TEST[:25], ids=lambda r: r.rid)
def test_restore_is_exact_and_nothing_known_leaks(rec):
    out = redact_recording(rec.texts, rec.rid)
    assert restore(out.texts, out.reid) == rec.texts
    assert out.leaks == []


def test_every_placeholder_in_the_text_is_in_the_map():
    rec = SYNTHETIC_TEST[0]
    out = redact_recording(rec.texts, rec.rid)
    placeholders = {e["placeholder"] for e in out.entities}
    for doc_id, text in out.texts.items():
        for r in out.reid.get(doc_id, []):
            assert text[r["start"]:r["end"]] == r["placeholder"] in placeholders


def test_shareable_summary_contains_no_values():
    out = redact_recording({"d": "NIC eka 953201456V"})
    assert "953201456" not in str(out.summary())


# --- NFR2: the map is personal data ------------------------------------------

def test_reid_map_refuses_a_location_git_does_not_ignore(tmp_path):
    """Outside the repository nothing is git-ignored, so nothing is protected."""
    out = redact_recording({"d": "NIC eka 953201456V"}, "SYN_X")
    with pytest.raises(PermissionError):
        write_reid_map(out, tmp_path)
    assert not list(tmp_path.iterdir())


def test_reid_file_pattern_is_ignored_anywhere_in_the_component():
    """.gitignore blocks *.reid.json everywhere, not only under reid_map/."""
    from redact import is_git_ignored
    assert is_git_ignored(C4_ROOT / "src" / "anything.reid.json")
    assert not is_git_ignored(C4_ROOT / "src" / "redact.py")


def test_reid_map_is_written_only_under_the_ignored_directory():
    out = redact_recording({"d": "NIC eka 953201456V"}, "SYN_X")
    target = REID_DIR / "pytest_tmp"
    try:
        path = write_reid_map(out, target)
        assert path.exists() and "953201456V" in path.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(target, ignore_errors=True)
