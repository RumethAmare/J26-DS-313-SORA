"""
Tests for cross-script address linking. Every address here is invented.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from addresses import cluster, part_of, same_address  # noqa: E402
from redact import redact_recording  # noqa: E402
from rules import Detection, detect, written_email  # noqa: E402


@pytest.mark.parametrize("a,b", [
    ("24, Galle Road, Dehiwala", "24, ගාලු පාර, දෙහිවල"),                      # transliterated
    ("No. 7, Temple Road, Kandy", "අංක 7, පන්සල පාර, මහනුවර"),                 # translated
    ("No. 88, Galle Road, Dehiwala", "Number අසූ අට, Galle Road, දෙහිවල"),     # spoken number
    ("number 14th Baseline Road, Colombo 8", "අංක 14, බේස්ලයින් පාර, කොළඹ 8"),  # ordinal
    ("88/3, Temple Lane, Colombo 5", "88/3, ටෙම්පල් පාර, කොළඹ 05"),              # zero-padding
    ("No. 82/1, Wijerama Road, Nugegoda", "number 82න් 1 විජේරාම පාර නුගේගොඩ"),  # "82න් 1" = 82/1
])
def test_same_address_in_both_scripts(a, b):
    assert same_address(a, b)


@pytest.mark.parametrize("a,b", [
    ("24, Galle Road, Dehiwala", "26, Galle Road, Dehiwala"),     # different house
    ("24, Galle Road, Dehiwala", "24, Kandy Road, Kadawatha"),    # different street and town
    ("24, Galle Road, Dehiwala", "Galle Road, Dehiwala"),         # number vs none: not the same
])
def test_different_addresses_stay_apart(a, b):
    assert not same_address(a, b)


def test_a_name_particle_is_not_a_spoken_number():
    """'De' in 'R. A. De Mel Mawatha' is a name, not 'deka' (2)."""
    assert same_address("410A, R. A. De Mel Mawatha, Colombo 3", "410A, ආර්. ඒ. ද මෙල් මාවත, කොළඹ 3")


def test_shortened_mention_joins_its_full_address():
    assert part_of("Nugegoda", "No. 82/1, Wijerama Road, Nugegoda")
    assert part_of("කොළඹ 7", "48/2, Horton Place, Colombo 7")
    assert not part_of("Colombo 3", "48/2, Horton Place, Colombo 7")


def test_ambiguous_shortened_mention_stays_apart():
    """Two different Colombo 7 addresses: 'Colombo 7' alone could be either."""
    c = cluster(["48/2, Horton Place, Colombo 7", "12, Ward Place, Colombo 7", "Colombo 7"])
    assert c[0] != c[1] and c[2] not in (c[0], c[1])


def test_written_email_from_spelling():
    assert written_email("n-i-m-a-l dot p-e-r-e-r-a nine five at example dot l-k") \
        == "nimal.perera95@example.lk"


def _address_detector(addresses):
    def detect_with_addresses(text, preceding=""):
        found = detect(text, preceding)
        for a in addresses:
            i = text.find(a)
            if i >= 0:
                found.append(Detection(i, i + len(a), "ADDRESS", a, a))
        return sorted(found, key=lambda d: d.start)
    return detect_with_addresses


def test_one_placeholder_per_address_and_email_across_documents():
    texts = {
        "en": "I live at 24, Galle Road, Dehiwala. Email eka nimal.perera95@example.lk.",
        "si": "මම ඉන්නේ 24, ගාලු පාර, දෙහිවල. Spell karoth "
              "n-i-m-a-l dot p-e-r-e-r-a nine five at example dot l-k.",
    }
    det = _address_detector(["24, Galle Road, Dehiwala", "24, ගාලු පාර, දෙහිවල"])
    red = redact_recording(texts, detect=det, classify_roles=False)
    out = red.texts
    assert "[ADDRESS_1]" in out["en"] and "[ADDRESS_1]" in out["si"]
    assert "[EMAIL_1]" in out["en"] and "[EMAIL_1]" in out["si"]
    assert "ADDRESS_2" not in out["si"] and "EMAIL_2" not in out["si"]
