"""
Tests for spoken-numeral decoding.

Every case marked GOLD is a real surface form taken from the shared corpus,
paired with the compact digit form recorded for the same entity_id in
annotations/c4/<recording>.entities.json. Those are ground truth, not
invented examples -- if one of them regresses, the decoder has broken against
data the component actually has to handle.
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numerals import (decode_number_words, encode_number_words,  # noqa: E402
                      find_number_word_runs)


# --- GOLD: spoken form -> compact form, both from the entity registry --------
GOLD = [
    ("බිංදුවයි හතයි හතයි පන්සිය පනස් පහයි හතළිස් හතරයි තිස් තුන", "0775554433"),
    ("දහනවයයි අනූඑකයි හැටහතයි අසූවයි තිස්හතරයි විසිඑක", "199167803421"),
    ("දහනවයයි අනූපහයි පනස් හයයි තිහයි විසි එකයි හතළිස් පහ", "199556302145"),
    ("බිංදුවයි හත්සිය හැත්තෑ හතයි දෙසිය තිස් හතරයි අටසිය අනූව", "0777234890"),
    ("බිංදුවයි බිංදුවයි නවයයි හතළිස් පහයි විසි තුනයි හැට හතයි දහය", "00945236710"),
    ("බිංදුවයි හැත්තෑ හයයි පනස් හතරයි තිස් දෙකයි එකසිය නමය", "0765432109"),
    ("අටසියයි පනස් හයයි හැත්තෑ දෙකයි විසි තුනයි හාරසිය අනූව", "800567223490"),
    ("බිංදුවයි හතයි එකයි විසිතුනයි හතළිස්පහයි හයයි හතයි අට", "0712345678"),
]


@pytest.mark.parametrize("spoken,expected", GOLD)
def test_gold_spoken_forms_reproduce_registry(spoken, expected):
    result = decode_number_words(spoken)
    assert result is not None, f"failed to decode: {spoken}"
    assert result[0] == expected


def test_romanized_sinhala_numerals():
    """Number words also appear romanized, mid-English. Real corpus surface."""
    spoken = ("zero binduwai hatai hayai, atai namayai binduwai, "
              "ekai dekai thunai hathara")
    digits, ambiguous = decode_number_words(spoken)
    assert digits == "0768901234"
    assert ambiguous, "romanized 'hata' is 7/60 ambiguous and must be flagged"


def test_english_digit_words():
    """Real corpus form: a phone number read out in English words."""
    spoken = "zero seven six, eight nine zero, one two three four"
    assert decode_number_words(spoken)[0] == "0768901234"


def test_cross_language_restatement_counts_once():
    """'zero binduwai' is one zero restated; 'binduwai binduwai' is two."""
    digits, ambiguous = decode_number_words("zero binduwai hatai")
    assert digits == "07" and ambiguous
    assert decode_number_words("binduwai binduwai hata")[0] == "007"


def test_english_digits_each_close_their_own_group():
    """'seven six' is 76. Summing it as an additive group would give 13."""
    assert decode_number_words("seven six")[0] == "76"


# --- Numeral system rules ---------------------------------------------------

def test_hundreds_are_multiplicative_not_additive():
    """එකසිය is 1x100, not එක + සිය = 101."""
    assert decode_number_words("එකසිය")[0] == "100"
    assert decode_number_words("එකසිය විසිතුන")[0] == "123"


def test_group_breaks_on_ascending_magnitude():
    """
    An additive group descends (500, 50, 5). An ascent means the previous
    group closed without a conjunctive suffix to mark it.
    """
    assert decode_number_words("පන්සිය පනස් පහ")[0] == "555"
    # 1 then 123 -- NOT 124
    assert decode_number_words("එක එකසිය විසිතුන")[0] == "1123"


def test_conjunctive_suffix_closes_a_group():
    assert decode_number_words("හැත්තෑ එකයි")[0] == "71"
    assert decode_number_words("අනූඑකයි")[0] == "91"


def test_constituents_may_be_written_without_spaces():
    """අනූඑකයි = අනූ + එක + යි, matched by longest prefix within the token."""
    assert decode_number_words("අනූඑකයි")[0] == decode_number_words("අනූ එකයි")[0]


def test_non_numeral_text_returns_none():
    assert decode_number_words("hello sampath bank") is None
    assert decode_number_words("මගේ නම") is None
    assert decode_number_words("") is None


# --- Span location ----------------------------------------------------------

def test_find_runs_returns_exact_offsets():
    text = "Contact number එක බිංදුවයි හතයි එකයි විසිතුනයි හතළිස්පහයි හයයි හතයි අට."
    runs = list(find_number_word_runs(text))
    assert len(runs) == 1
    start, end, digits, _ = runs[0]
    assert digits == "0712345678"
    # The offset contract the whole pipeline depends on.
    assert text[start:end].startswith("බිංදුවයි")
    assert "අට" in text[start:end]


def test_find_runs_excludes_sentence_final_punctuation():
    """Span boundary rule: the full stop ending the sentence is not the number."""
    text = "Contact number එක බිංදුවයි හතයි එකයි විසිතුනයි හතළිස්පහයි හයයි හතයි අට."
    start, end, _, _ = next(find_number_word_runs(text))
    assert text[start:end].endswith("අට")
    assert end == len(text) - 1


def test_case_suffix_on_final_numeral_is_excluded():
    """Real corpus form: 'නමයට' is nine + dative -ට ('call ... nine')."""
    text = "හරි බිංදුවයි හැත්තෑ හයයි පනස් හතරයි තිස් දෙකයි එකසිය නමයට කෝල් කරන්න."
    start, end, digits, _ = next(find_number_word_runs(text))
    assert digits == "0765432109"
    assert text[start:end].endswith("එකසිය නමය")


def test_short_counting_words_are_not_identifiers():
    """'deka' meaning 'two things' must not surface as an identifier."""
    assert list(find_number_word_runs("mata deka one")) == []


@pytest.mark.parametrize("prefix", [
    "Contact number එක ",
    "mobile number eka ",
    "account එක ",
    "NIC එක ",
])
def test_classifier_eka_is_not_the_digit_one(prefix):
    """
    එක/eka as a classifier means "the", not 1. Reading it as a digit silently
    corrupts the identifier -- 0712345678 becomes 1712345678 -- which is worse
    than missing it, because a corrupted number still looks plausible.
    """
    text = prefix + "බිංදුවයි හතයි එකයි විසිතුනයි හතළිස්පහයි හයයි හතයි අට"
    runs = list(find_number_word_runs(text))
    assert len(runs) == 1
    assert runs[0][2] == "0712345678"


def test_conjunctive_eka_is_still_a_digit():
    """එකයි (suffixed) is never the classifier -- it must stay a digit."""
    assert decode_number_words("හැත්තෑ එකයි")[0] == "71"


def test_run_offsets_never_exceed_text():
    text = "මගේ නම්බර් එක බිංදුවයි හතයි හතයි පන්සිය පනස් පහයි හතළිස් හතරයි තිස් තුන ය."
    for start, end, _, _ in find_number_word_runs(text):
        assert 0 <= start < end <= len(text)


# --- Documented limitations -------------------------------------------------

@pytest.mark.xfail(reason="speaker self-correction; annotator normalised it away",
                   strict=True)
def test_known_limitation_false_start_repetition():
    """
    'එක එකසිය විසිතුනයි' is a false start -- the speaker began a digit, then
    restated it as a hundreds group. The decoder reads what was said (1, 123);
    the registry records what was meant (123). Collapsing repeats would corrupt
    genuinely repeated digits such as the 77 in 0771234567, so this is accepted
    and reported rather than patched.
    """
    spoken = "දෙදාස් එකයි බිංදුවයි තුනයි හයසිය හැත්තෑවයි එක එකසිය විසිතුනයි පහ"
    assert decode_number_words(spoken)[0] == "2001036701235"


# --- Encoding: the inverse of the decoder -----------------------------------

@pytest.mark.parametrize("spoken,expected", GOLD)
def test_encoder_output_decodes_to_gold_digits(spoken, expected):
    assert decode_number_words(encode_number_words(expected))[0] == expected


def test_encoder_reproduces_a_real_corpus_rendering():
    assert encode_number_words("0775554433", [1, 1, 1, 3, 2, 2]) == GOLD[0][0]


def test_encode_decode_round_trip_property():
    """
    Every valid grouping of every digit string must decode back unchanged.
    This is what lets synthetic spoken identifiers be trusted as gold data.
    """
    rng = random.Random(7)
    for _ in range(3000):
        digits = "".join(rng.choices("0123456789", k=rng.randint(6, 13)))
        groups, i = [], 0
        while i < len(digits):
            size = 1 if digits[i] == "0" else rng.randint(1, min(3, len(digits) - i))
            groups.append(size)
            i += size
        for kw in ({"groups": groups}, {"groups": groups, "joined": True},
                   {"romanized": True}):
            spoken = encode_number_words(digits, **kw)
            assert decode_number_words(spoken)[0] == digits, (digits, kw, spoken)


@pytest.mark.parametrize("digits,groups", [
    ("0712", [2, 2]),      # "07" is not a spoken number
    ("123", [1, 1]),       # does not cover every digit
    ("1234", [4]),         # groups are at most three digits
])
def test_encoder_rejects_invalid_groupings(digits, groups):
    with pytest.raises(ValueError):
        encode_number_words(digits, groups)
