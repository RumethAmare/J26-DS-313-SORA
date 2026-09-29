"""
Tests for cross-script entity resolution (Contribution 2).

Names here are common Sri Lankan names, not taken from the corpus.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus import load_synthetic  # noqa: E402
from redact import redact_recording  # noqa: E402
from resolve import (DEFAULT_THRESHOLD, Mention, mention_similarity,  # noqa: E402
                     name_tokens, phonetic_key, resolve, romanise, score_linking,
                     tune_threshold)
from rules import Detection  # noqa: E402
from rules import detect as rule_detect  # noqa: E402

# --- Romanisation --------------------------------------------------------------

@pytest.mark.parametrize("sinhala,latin", [
    ("නිමල්", "nimal"),            # al-lakuna: no final vowel
    ("චමරි", "chamari"),           # inherent vowel, vowel sign
    ("ප්‍රනාන්දු", "pranaandu"),     # rakaransaya conjunct (් + ZWJ + ර)
    ("වික්‍රමසිංහ", "wikramasingha"),  # conjunct and anusvara
    ("පෙරේරා", "pereeraa"),         # long vowel signs
    ("අමල්", "amal"),              # independent vowel
])
def test_romanise(sinhala, latin):
    assert romanise(sinhala) == latin


def test_sinhala_names_are_single_tokens():
    """\\w does not cover Sinhala vowel signs; the tokeniser must not split on them."""
    assert name_tokens("කසුන් වික්‍රමසිංහ") == ["කසුන්", "වික්‍රමසිංහ"]


def test_honorifics_are_not_name_tokens():
    assert name_tokens("Mr. Perera") == ["perera"]
    assert name_tokens("පෙරේරා මහත්මයා") == ["පෙරේරා"]


# --- Similarity ------------------------------------------------------------------

@pytest.mark.parametrize("latin,sinhala", [
    ("Fernando", "ප්‍රනාන්දු"),          # no shared characters at all
    ("Wickramasinghe", "වික්‍රමසිංහ"),    # -singhe / -singha
    ("Gunawardena", "ගුණවර්ධන"),
    ("Dissanayake", "දිසානායක"),
    ("Kasun", "කසුන් වික්‍රමසිංහ"),      # first name alone vs full name
    ("Mr. Perera", "නිමල් පෙරේරා"),      # honorific + surname vs full name
])
def test_same_person_across_scripts(latin, sinhala):
    assert mention_similarity(latin, sinhala) >= DEFAULT_THRESHOLD


@pytest.mark.parametrize("a,b", [
    ("Nimal", "කමල්"),
    ("Kamal", "කුමාර"),
    ("Kamal Perera", "නිමල් පෙරේරා"),   # shared surname, different person
    ("Mahesh", "මහේෂි"),                 # man / woman: -i forms the woman's name
    ("Nimal", "නිමලි"),
])
def test_different_people(a, b):
    assert mention_similarity(a, b) < DEFAULT_THRESHOLD


def test_sinhala_case_endings_are_ignored():
    """අමාශිට 'to Amashi', සෙල්වීගෙන් 'from Selvy' -- the span covers the ending."""
    assert mention_similarity("Amashi", "අමාශිට") >= DEFAULT_THRESHOLD
    assert mention_similarity("Selvy", "සෙල්වීගෙන්") >= DEFAULT_THRESHOLD


def test_final_i_is_kept_in_the_key():
    assert phonetic_key("Nimali") != phonetic_key("Nimal")
    assert phonetic_key("Wickramasinghe") == phonetic_key("වික්‍රමසිංහ")


# --- Clustering --------------------------------------------------------------------

def test_resolve_links_across_scripts_and_keeps_people_apart():
    mentions = [Mention("a", "Nimal Perera"), Mention("b", "නිමල් පෙරේරා"),
                Mention("c", "Kamal Perera"), Mention("d", "කමල් පෙරේරා"),
                Mention("e", "Nimal")]
    c = resolve(mentions)
    assert c["a"] == c["b"] == c["e"]
    assert c["c"] == c["d"]
    assert c["a"] != c["c"]


def test_exact_baseline_cannot_link_scripts():
    c = resolve([Mention("a", "Nimal"), Mention("b", "නිමල්")], system="exact")
    assert c["a"] != c["b"]


# --- Method discipline ---------------------------------------------------------------

def test_frozen_threshold_is_what_train_tuning_selects():
    """The threshold must come from the train split, never from test data."""
    assert tune_threshold(load_synthetic("train")) == DEFAULT_THRESHOLD


def test_linking_on_held_out_synthetic_names():
    """
    On unseen names the proposal target (0.85) is met, and every remaining
    failure is an inherent ambiguity: two people in the call share a first
    name, so a bare "Buddhika" cannot be attributed from the name alone.
    """
    from collections import defaultdict

    recordings = {r.rid: r for r in load_synthetic("test")}
    result = score_linking(recordings.values())
    assert result["cross_script_accuracy"] >= 0.85

    from resolve import person_mentions
    for failure in result["failures"]:
        mentions, gold = person_mentions(recordings[failure["recording"]])
        first_names = defaultdict(set)
        for m in mentions:
            tokens = name_tokens(m.surface)
            if tokens:
                first_names[phonetic_key(tokens[0])].add(gold[m.mid])
        assert any(len(ids) > 1 for ids in first_names.values()), failure


# --- Integration with redaction ------------------------------------------------------

def test_one_person_gets_one_placeholder_in_both_scripts():
    texts = {"r_transcript_u001_v1": "mage nama Nimal Perera.",
             "r_c2_u001_si_v1": "මගේ නම නිමල් පෙරේරා.",
             "r_c2_summary_en_v1": "Mr. Perera's brother Kamal Perera called."}
    names = ["Nimal Perera", "නිමල් පෙරේරා", "Mr. Perera", "Kamal Perera"]

    def detect(text, preceding):
        found = list(rule_detect(text, preceding))
        for n in names:
            i = text.find(n)
            if i >= 0:
                found.append(Detection(i, i + len(n), "PERSON", n))
        return sorted(found, key=lambda d: d.start)

    linked = redact_recording(texts, detect=detect)
    assert linked.texts["r_transcript_u001_v1"] == "mage nama [PERSON_1]."
    assert linked.texts["r_c2_u001_si_v1"] == "මගේ නම [PERSON_1]."
    assert linked.texts["r_c2_summary_en_v1"] == "[PERSON_1]'s brother [PERSON_2] called."

    unlinked = redact_recording(texts, detect=detect, resolve_names=False)
    assert unlinked.texts["r_c2_u001_si_v1"] == "මගේ නම [PERSON_2]."
