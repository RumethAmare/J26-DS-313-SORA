import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from normalize import (fix_numbers, fix_sinhala, normalize, prepare, remove_fillers,  # noqa: E402
                       romanize, split_mixed_words)


def test_hyphen_clitic_is_split():
    assert split_mixed_words("meeting-eka") == "meeting eka"
    assert split_mixed_words("file-ekata") == "file ekata"


def test_latin_glued_to_sinhala_is_split():
    assert split_mixed_words("bankඑකට") == "bank එකට"


def test_fillers_removed():
    assert remove_fillers("uh mama, umm call karannam") == "mama, call karannam"


def test_reduplication_kept():
    assert normalize("podi podi wada") == "podi podi wada"


def test_full_example():
    assert normalize("uh meeting-eka Friday ekata reschedule karamu ,") == \
        "meeting eka Friday ekata reschedule karamu"


def test_spoken_digits_become_numbers():
    assert prepare("number eka binduwai hatai hayai, atai namayai") == "number eka 0 7 6, 8 9"


def test_spelling_variant():
    assert prepare("van ekak onne neda?") == "van ekak one neda?"


def test_broken_conjunct_rejoined():
    assert fix_sinhala("නයින් ආර්ක් ට් රාන්ස්පෝට්") == "නයින් ආර්ක් ට්‍රාන්ස්පෝට්"
    assert fix_sinhala("මෝටර් රථ") == "මෝටර් රථ"   # a real word break is left alone


def test_romanize():
    assert romanize("මචං කොහොමද") == "machan kohomada"
    assert romanize("ඕනේ කීයද දන්නවා") == "one kiyada dannawa"
    assert romanize("meeting එක") == "meeting eka"


def test_split_conjunct_rejoined_only_when_known():
    vocab = {"අවශ්‍ය"}
    assert fix_sinhala("අවශ් යතාවය", vocab) == "අවශ්‍යතාවය"
    assert fix_sinhala("නිවේදනයක් යවමුද?", vocab) == "නිවේදනයක් යවමුද?"


def test_numbers_copied_from_source():
    assert fix_numbers("1995 123 089 34 madam", "1995123809346, Madam.") == "199512308934, Madam."
    assert fix_numbers("PC 20 23 COL 00 89 2", "PC-2023 COL-00892.") == "PC-2023 COL-00892."   # same digits: kept
    assert fix_numbers("1 2 3 4 and 5", "1234") == "1234"   # can't pair the numbers: left alone
