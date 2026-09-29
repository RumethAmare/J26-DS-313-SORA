"""
Tests for the structured-identifier rule layer.

Every surface form here mirrors a form observed in the real corpus, but every
VALUE is invented -- no real identifier appears in this repository (NFR2).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rules import classify_digits, detect, is_new_nic, normalise_phone  # noqa: E402
from synthetic import generate  # noqa: E402


def only(text, preceding=""):
    """The single detection in `text`, with the offset contract checked."""
    found = detect(text, preceding)
    assert len(found) == 1, found
    d = found[0]
    assert text[d.start:d.end] == d.surface
    return d


# --- PHONE ------------------------------------------------------------------

@pytest.mark.parametrize("text,surface", [
    ("mage number eka 0771234567.", "0771234567"),
    ("mage number eka 0 7 7 1 2 3 4 5 6 7 ta call karanna.", "0 7 7 1 2 3 4 5 6 7"),
    ("You can reach me on 077 123 4567.", "077 123 4567"),
    ("You can reach me on +94 77 123 4567.", "+94 77 123 4567"),
    ("contact number එක බිංදුවයි හතයි එකයි විසිතුනයි හතළිස්පහයි හයයි හතයි අට.",
     "බිංදුවයි හතයි එකයි විසිතුනයි හතළිස්පහයි හයයි හතයි අට"),
    ("mobile eka zero seven six, eight nine zero, one two three four.",
     "zero seven six, eight nine zero, one two three four"),
])
def test_phone_in_every_spoken_form(text, surface):
    d = only(text)
    assert (d.label, d.surface) == ("PHONE", surface)


def test_phone_value_is_normalised():
    assert only("call me on +94 77 123 4567").value == "0771234567"


def test_extension_is_part_of_the_phone_span():
    d = only("Call Dilini at 0112345678 extension 42.")
    assert d.surface == "0112345678 extension 42"


def test_standalone_extension_is_a_phone():
    d = only("Please ask for extension number 305.")
    assert (d.label, d.surface) == ("PHONE", "extension number 305")


def test_organisation_hotline_is_detected_but_not_redacted():
    d = only("Dialog hotline eka 0117654321.")
    assert d.label == "PHONE" and d.role == "ORGANISATION" and not d.redact


def test_misspoken_eleven_digit_mobile_is_flagged():
    d = only("07712345678")
    assert d.label == "PHONE" and d.ambiguous


# --- NIC ---------------------------------------------------------------------

@pytest.mark.parametrize("text,surface", [
    ("mage NIC eka 953201456V.", "953201456V"),
    ("NIC number eka 9 5 3 2 0 1 4 5 6 v.", "9 5 3 2 0 1 4 5 6 v"),
    ("My NIC number is 199512345678.", "199512345678"),
    ("My national identity card number is 199512345678.", "199512345678"),
    ("මගේ ජාතික හැඳුනුම්පත් අංකය 199512345678.", "199512345678"),
])
def test_nic_forms(text, surface):
    d = only(text)
    assert (d.label, d.surface) == ("NIC", surface)


def test_new_nic_structure():
    assert is_new_nic("199512345678")          # day 123
    assert is_new_nic("199562345678")          # day 623 = woman, day 123
    assert not is_new_nic("199540045678")      # day 400 is never issued


def test_keyword_decides_between_nic_and_account():
    """Twelve digits are a valid NIC shape either way; the keyword decides."""
    assert only("account number eka 199512345678").label == "ACCOUNT"
    assert only("NIC eka 199512345678").label == "NIC"


def test_nic_with_an_extra_spoken_digit_is_still_the_nic():
    d = only("The NIC is 1995123456789.")
    assert d.label == "NIC" and d.ambiguous


# --- ACCOUNT -----------------------------------------------------------------

@pytest.mark.parametrize("text,surface", [
    ("reference eka LN1234NG56789.", "LN1234NG56789"),
    ("My account is CEB/KW/1234/567890.", "CEB/KW/1234/567890"),
    ("Policy number is POL-2024-123456.", "POL-2024-123456"),
    ("customer number eka D L G 1 2 3 4 5 6 7 8.", "D L G 1 2 3 4 5 6 7 8"),
    ("ID eka N අනූ අටයි හැත්තෑ හයයි පන්සිය හතළිස් තුන.",
     "N අනූ අටයි හැත්තෑ හයයි පන්සිය හතළිස් තුන"),
    ("මගේ ගිණුම් අංකය 123456789012.", "123456789012"),
])
def test_account_and_reference_forms(text, surface):
    d = only(text)
    assert (d.label, d.surface) == ("ACCOUNT", surface)


@pytest.mark.parametrize("text", [
    "I paid Rs. 25,000 on the 3rd of May.",
    "mama rupiyal 150000 gewwa.",
    "last month 1500 rupees debit una, deka parak.",
    "මම රුපියල් 150000 ක් ගෙව්වා.",
])
def test_money_amounts_are_not_identifiers(text):
    assert detect(text) == []


# --- DOB ---------------------------------------------------------------------

@pytest.mark.parametrize("text,surface", [
    ("My date of birth is 12th of August 1995.", "12th of August 1995"),
    ("I was born on August 12, 1995.", "August 12, 1995"),
    ("date of birth eka June 3rd 1990.", "June 3rd 1990"),
    ("date of birth eka June 9rd 1990.", "June 9rd 1990"),     # ASR ordinal error
    ("DOB 12/08/1995", "12/08/1995"),
    ("මගේ උපන් දිනය 1995 අගෝස්තු 12.", "1995 අගෝස්තු 12"),
    ("උපන් දිනය 1995 පෙබරවාරි 21 වැනිදා.", "1995 පෙබරවාරි 21 වැනිදා"),
])
def test_birth_date_forms(text, surface):
    d = only(text)
    assert (d.label, d.surface) == ("DOB", surface)


def test_answer_in_the_next_utterance_uses_the_question_as_context():
    """'Date of birth?' / '2013 මාර්තු 12' -- the keyword is in the prior turn."""
    assert detect("2013 මාර්තු 12.") == []
    d = only("2013 මාර්තු 12.", preceding="Date of birth?")
    assert (d.label, d.value) == ("DOB", "2013-03-12")


def test_answer_with_a_second_sentence_is_still_an_answer():
    d = only("17th of August. I need to be home on that day.",
             preceding="When is your mother's birthday?")
    assert d.surface == "17th of August"


def test_previous_utterance_is_ignored_when_this_is_not_an_answer():
    """A long turn about something else must not inherit the last keyword."""
    text = "On 7th of June I transferred the money to my brother."
    assert detect(text, preceding="My birthday is in June.") == []


def test_keyword_after_the_number_sinhala_verb_final():
    """Sinhala is verb-final: 'call' comes after the number."""
    assert only("ගැටලුවක් ඇත්නම් 0112345678 අමතන්න.").label == "PHONE"
    assert only("prashnayak thiyenam 0112345678 ta call karanna.").label == "PHONE"


def test_amount_keyword_does_not_carry_into_the_next_turn():
    d = only("0112345678.", preceding="mama rupiyal 1500 gewwa")
    assert d.label == "PHONE"


def test_misspoken_eleven_digit_landline_is_a_phone():
    d = only("prashnayak thiyenam 01123456789 ta call karanna.")
    assert d.label == "PHONE" and d.ambiguous


def test_spoken_sinhala_date():
    text = "දෙදාස් දහතුනේ මාර්තු දොළහ."
    d = only(text, preceding="උපන් දිනය?")
    assert d.surface == "දෙදාස් දහතුනේ මාර්තු දොළහ" and d.value == "2013-03-12"


def test_transaction_date_is_not_a_birth_date():
    assert detect("mama 3rd of May rupiyal 1500 gewwa.") == []
    assert detect("I paid it on 3 May 2024.") == []


# --- EMAIL -------------------------------------------------------------------

def test_email_excludes_sentence_final_full_stop():
    d = only("Email eka sir, kasun.perera99@gmail.com.")
    assert (d.label, d.surface) == ("EMAIL", "kasun.perera99@gmail.com")


# --- Contract ----------------------------------------------------------------

def test_phone_normalisation():
    assert normalise_phone("94771234567") == "0771234567"
    assert normalise_phone("0771234567") == "0771234567"
    assert normalise_phone("0071234567") is None


def test_amount_context_suppresses_long_numbers():
    assert classify_digits("1500000", "AMOUNT") is None


def test_detections_never_overlap_and_match_their_text():
    text = ("Mage NIC eka 953201456V, number eka 0771234567, email "
            "kasun@gmail.com, date of birth 12th of August 1995.")
    found = detect(text)
    assert [d.label for d in found] == ["NIC", "PHONE", "EMAIL", "DOB"]
    for a, b in zip(found, found[1:]):
        assert a.end <= b.start
    assert all(text[d.start:d.end] == d.surface for d in found)


def test_full_recall_on_synthetic_structured_identifiers():
    """
    Regression guard, not a result: the rules must keep finding every
    structured identifier the generator produces. Real-data performance is
    measured by the evaluation harness on held-out recordings.
    """
    labels = {"NIC", "PHONE", "ACCOUNT", "DOB", "EMAIL"}
    missed = []
    for _, docs, rows, _ in generate(40, seed=13):
        for doc in docs:
            pred = {(d.start, d.end, d.label) for d in detect(doc["text"])}
            for r in rows:
                if r["doc_id"] == doc["doc_id"] and r["label"] in labels:
                    if (r["start_char"], r["end_char"], r["label"]) not in pred:
                        missed.append((r["label"], r["surface"]))
    assert not missed, missed[:10]
