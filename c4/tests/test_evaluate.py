"""
Tests for the evaluation harness.

The scorer produces the numbers that get reported, so it is checked against
hand-worked examples rather than against itself.
"""
import sys
from collections import namedtuple
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus import Recording  # noqa: E402
from evaluate import (LABEL_SETS, doc_kind, evaluate, mask, prf,  # noqa: E402
                      span_script)

P = namedtuple("P", "start end label")

DOC = "SYN_T0001_transcript_u001_v1"
TEXT = "Nimal 0771234567 NIC 953201456V mail a@b.com and 199512345678 ok"
#       0     6          17  21         32   37      45  49           62


def row(start, end, label, script="LATIN", **kw):
    return {"ann_id": f"a{start}", "doc_id": DOC, "entity_id": f"e{start}", "label": label,
            "start_char": start, "end_char": end, "surface": TEXT[start:end],
            "script": script, "role": "PRIVATE_INDIVIDUAL", "redact": True, **kw}


def recording(rows):
    return Recording("SYN_T0001", {DOC: TEXT}, rows, None)


def fixed(*preds):
    return lambda text, preceding: list(preds)


def test_offsets_in_this_file_are_right():
    assert TEXT[6:16] == "0771234567"
    assert TEXT[21:31] == "953201456V"
    assert TEXT[37:44] == "a@b.com"
    assert TEXT[49:61] == "199512345678"


def test_hand_worked_example():
    """
    gold: PHONE, NIC, EMAIL, NIC          pred: PHONE, NIC, EMAIL(wrong end), ACCOUNT
    tp = PHONE, NIC(21:31)                -> 2
    fp = EMAIL(37:43), ACCOUNT(49:61)     -> 2
    fn = EMAIL(37:44), NIC(49:61)         -> 2
    P = 2/4 = 0.5, R = 2/4 = 0.5
    """
    rec = recording([row(6, 16, "PHONE"), row(21, 31, "NIC"),
                     row(37, 44, "EMAIL"), row(49, 61, "NIC")])
    pred = fixed(P(6, 16, "PHONE"), P(21, 31, "NIC"), P(37, 43, "EMAIL"),
                 P(49, 61, "ACCOUNT"))
    out = evaluate([rec], pred, LABEL_SETS["structured"])
    m = out["micro"]
    assert (m["tp"], m["fp"], m["fn"]) == (2, 2, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert out["per_label"]["NIC"] == prf(1, 0, 1)
    assert out["per_label"]["ACCOUNT"] == prf(0, 1, 0)
    # One error of each kind.
    assert len(out["errors"]["boundary"]) == 1      # EMAIL off by one
    assert len(out["errors"]["label"]) == 1         # NIC called ACCOUNT


def test_strict_matching_gives_no_partial_credit():
    rec = recording([row(6, 16, "PHONE")])
    out = evaluate([rec], fixed(P(6, 15, "PHONE")), ["PHONE"])
    assert out["micro"]["tp"] == 0 and out["micro"]["recall"] == 0.0


def test_labels_outside_the_set_are_ignored_on_both_sides():
    rec = recording([row(0, 5, "PERSON"), row(6, 16, "PHONE")])
    out = evaluate([rec], fixed(P(0, 5, "PERSON"), P(6, 16, "PHONE")), ["PHONE"])
    assert out["micro"]["tp"] == 1 and out["micro"]["fp"] == 0


def test_perfect_prediction():
    rows = [row(6, 16, "PHONE"), row(21, 31, "NIC")]
    out = evaluate([recording(rows)], fixed(P(6, 16, "PHONE"), P(21, 31, "NIC")),
                   ["PHONE", "NIC"])
    assert out["micro"]["f1"] == 1.0 and out["macro"]["f1"] == 1.0


def test_macro_averages_only_labels_present_in_gold():
    """DOB has no gold spans here, so it must not pull the macro mean to 0."""
    rec = recording([row(6, 16, "PHONE")])
    out = evaluate([rec], fixed(P(6, 16, "PHONE")), ["PHONE", "DOB"])
    assert out["macro"]["f1"] == 1.0


def test_per_script_uses_gold_script_for_misses():
    rec = recording([row(6, 16, "PHONE", script="SINHALA")])
    out = evaluate([rec], fixed(), ["PHONE"])
    assert out["per_script"]["SINHALA"]["fn"] == 1


def test_lenient_mode_strips_gold_trailing_punctuation_only():
    text = "call 0771234567."
    doc_row = {**row(5, 16, "PHONE"), "surface": "0771234567."}
    rec = Recording("SYN_T0001", {DOC: text}, [doc_row], None)
    pred = fixed(P(5, 15, "PHONE"))
    assert evaluate([rec], pred, ["PHONE"])["micro"]["tp"] == 0
    assert evaluate([rec], pred, ["PHONE"], lenient=True)["micro"]["tp"] == 1


def test_real_surfaces_are_masked_in_error_lists():
    rec = recording([row(0, 5, "PERSON"), row(6, 16, "PHONE")])
    out = evaluate([rec], fixed(), ["PERSON", "PHONE"], masked=True)
    shown = {e["gold"] for e in out["errors"]["missed"]}
    assert shown == {"xxxxx", "9999999999"}


def test_mask_hides_every_script():
    assert mask("Nimal නිමල් 077") == "xxxxx සසසසස 999"


@pytest.mark.parametrize("doc_id,kind", [
    ("J26DS313_R0002_transcript_u001_v1", "transcript"),
    ("J26DS313_R0002_c2_u001_en_v1", "clean_en"),
    ("J26DS313_R0002_c2_u001_si_v1", "clean_si"),
    ("J26DS313_R0002_c2_summary_en_v1", "summary"),
])
def test_doc_kind(doc_id, kind):
    assert doc_kind(doc_id) == kind


def test_digits_in_clean_si_count_as_sinhala_script():
    assert span_script("0771234567", "X_c2_u001_si_v1") == "SINHALA"
    assert span_script("0771234567", "X_c2_u001_en_v1") == "LATIN"


def test_noise_recall_is_reported_for_synthetic_noise():
    rec = recording([row(6, 16, "PHONE", noise=["digit_repeated"]), row(21, 31, "NIC")])
    out = evaluate([rec], fixed(P(21, 31, "NIC")), ["PHONE", "NIC"])
    assert out["recall_by_noise"]["digit_repeated"]["recall"] == 0.0
    assert out["recall_by_noise"]["clean"]["recall"] == 1.0
