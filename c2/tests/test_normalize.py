import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from normalize import normalize, remove_fillers, split_mixed_words  # noqa: E402


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
