"""
Tests for the synthetic corpus generator.

Synthetic data is only useful as gold data if it is exactly right, so these
check the generator against the schema validator, against the numeral decoder,
and against the NIC issuing rules.
"""
import datetime as dt
import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numerals import decode_number_words  # noqa: E402
from synthetic import LEXICONS, generate, make_nic, nic_day_of_year  # noqa: E402
from validate import validate_recording  # noqa: E402

CORPUS = list(generate(40, seed=13))
TEST_CORPUS = list(generate(40, seed=29, split="test"))
HAS_WORDS = re.compile(r"[a-z඀-෿]{3}")


@pytest.mark.parametrize("rid,docs,rows,registry", CORPUS + TEST_CORPUS,
                         ids=[c[0] for c in CORPUS + TEST_CORPUS])
def test_every_recording_passes_the_validator(rid, docs, rows, registry):
    rep = validate_recording(rows, {d["doc_id"]: d["text"] for d in docs}, registry)
    assert rep.ok, rep.errors
    assert not rep.warnings, "synthetic data must use the current schema only"


@pytest.mark.parametrize("split", ["train", "test"])
def test_generation_is_deterministic(split):
    assert list(generate(3, 13, split)) == list(generate(3, 13, split))
    assert list(generate(3, 13, split)) != list(generate(3, 14, split))


# --- Test split: held out from train ------------------------------------------

def test_test_split_lexicon_is_disjoint_from_train():
    """No train name, place or organisation may reach the test split."""
    for key in LEXICONS["train"]:
        train = {x[0] for x in LEXICONS["train"][key]} | {x[1] for x in LEXICONS["train"][key]}
        test = {x[0] for x in LEXICONS["test"][key]} | {x[1] for x in LEXICONS["test"][key]}
        assert not train & test, (key, train & test)


def test_test_split_never_uses_train_templates():
    train_texts = {d["text"] for _, docs, _, _ in CORPUS for d in docs}
    test_texts = {d["text"] for _, docs, _, _ in TEST_CORPUS for d in docs}
    assert not train_texts & test_texts


def test_test_split_has_keywordless_answer_turns():
    """Question in one utterance, identifier in the next with no keyword."""
    answers = [d["text"] for _, docs, _, _ in TEST_CORPUS for d in docs
               if d["kind"] == "transcript" and d["topic"] == "dob"
               and not re.search(r"upan|birth|ipadune", d["text"])]
    assert answers


def test_test_split_noise_is_recorded_on_the_span():
    kinds = {n for _, _, rows, _ in TEST_CORPUS for r in rows for n in r.get("noise", [])}
    assert {"nic_letter_dropped", "digit_repeated", "restatement", "ordinal_error"} <= kinds
    assert not any("noise" in r for _, _, rows, _ in CORPUS for r in rows)


def test_every_row_is_marked_synthetic():
    assert all(r["annotation_source"] == "synthetic"
               for _, _, rows, _ in CORPUS for r in rows)


def test_one_person_one_entity_id_across_documents_and_scripts():
    """The customer is one entity everywhere, in both scripts."""
    for _, _, rows, registry in CORPUS:
        customer = next(e for e in registry["entities"]
                        if e["label"] == "PERSON" and e["role"] == "PRIVATE_INDIVIDUAL")
        mentions = [r for r in rows if r["entity_id"] == customer["entity_id"]]
        assert {r["script"] for r in mentions} == {"LATIN", "SINHALA"}
        docs = {r["doc_id"] for r in mentions}
        for kind in ("_transcript_", "_en_v1", "_si_v1", "_summary_"):
            assert any(kind in d for d in docs), kind


def test_spoken_identifiers_decode_to_their_registry_value():
    """Ties the generator to the decoder: a spoken span means its value."""
    checked = 0
    for _, _, rows, registry in CORPUS:
        values = {e["entity_id"]: e.get("value") for e in registry["entities"]}
        for r in rows:
            if r["label"] in ("NIC", "PHONE", "ACCOUNT") and HAS_WORDS.search(r["surface"]):
                digits = re.sub(r"\D", "", values[r["entity_id"]])
                assert decode_number_words(r["surface"])[0] == digits, r["surface"]
                checked += 1
    assert checked > 20, "corpus should exercise the number-word forms"


def test_hard_negatives_are_present_and_unredacted():
    seen = {(r["label"], r["role"], r["redact"]) for _, _, rows, _ in CORPUS for r in rows}
    assert ("LOCATION", "PUBLIC_PLACE", False) in seen
    assert ("PHONE", "ORGANISATION", False) in seen
    assert ("ORG", "ORGANISATION", False) in seen


def test_distractor_amounts_are_never_annotated():
    amount = re.compile(r"(?:Rs\. |rupiyal |රුපියල් )([\d,]+)")
    for _, docs, rows, _ in CORPUS:
        for d in docs:
            spans = [(r["start_char"], r["end_char"])
                     for r in rows if r["doc_id"] == d["doc_id"]]
            for m in amount.finditer(d["text"]):
                assert not any(s <= m.start(1) < e for s, e in spans)


def test_distractors_agree_across_one_utterance():
    """The three renderings of an utterance state the same amount."""
    for _, docs, _, _ in CORPUS:
        by_utt = {}
        for d in docs:
            if d["topic"] == "payment":
                nums = re.findall(r"\d[\d,]{3,}", d["text"])
                by_utt.setdefault(d["utterance"], set()).update(
                    n.replace(",", "") for n in nums)
        for amounts in by_utt.values():
            assert len(amounts) == 1, amounts


# --- NIC issuing rules --------------------------------------------------------

def test_nic_day_counts_february_as_29_days():
    """1 March is day 61 on an NIC in every year, leap or not."""
    assert nic_day_of_year(dt.date(1995, 3, 1)) == 61
    assert nic_day_of_year(dt.date(1996, 3, 1)) == 61


@pytest.mark.parametrize("old", [True, False])
@pytest.mark.parametrize("sex", ["M", "F"])
def test_nic_encodes_birth_date_and_sex(old, sex):
    dob = dt.date(1995, 8, 12)
    nic = make_nic(dob, sex, random.Random(0), old)
    day = nic_day_of_year(dob) + (500 if sex == "F" else 0)
    if old:
        assert re.fullmatch(r"\d{9}[VX]", nic)
        assert nic[:2] == "95" and int(nic[2:5]) == day
    else:
        assert re.fullmatch(r"\d{12}", nic)
        assert nic[:4] == "1995" and int(nic[4:7]) == day
