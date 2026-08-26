#!/usr/bin/env python3
"""
script_normalize.py -- Task 1 script-normalization pass for C1 WER scoring.

WHY THIS EXISTS
---------------
Gold C1 transcripts are written in *two different scripts* depending on the
recording: 9 of 27 are 100% romanized Latin ("mama", "thiyenawa", "gedara"),
while most of the rest are majority Sinhala Unicode ("මම", "තියෙනවා").
faster-whisper, by contrast, auto-detects one language per recording and
decodes almost everything in Sinhala Unicode.

Scoring those directly means a romanized-gold recording is compared against
Sinhala-Unicode hypothesis text, so WER is pinned near 1.0 no matter how well
the model actually heard the words. That is a *scoring artifact* stacked on
top of the genuine ASR failure, and quoting a single WER number conflates the
two. This module puts both sides into one representation so the two failure
modes can be reported separately.

APPROACH
--------
Both sides are mapped to a deliberately lossy "phonetic skeleton":

  1. Sinhala Unicode -> Latin, via the standard abugida rule (consonant
     carries an implicit /a/ unless a vowel sign or the virama follows).
  2. Latin -> collapsed skeleton: strip diacritics, fold aspirated digraphs
     onto their plain consonant (th->t, dh->d, ...), fold w->v, collapse
     doubled letters and long vowels.

The output is NOT meant to be readable or linguistically correct
romanization. It is meant to be *consistent*: the same spoken word reaches
the same skeleton whether it arrived as Sinhala Unicode, as a human
annotator's ad-hoc romanization ("thiyenawa"), or as a scholarly one
("tiyenavā"). Because the identical transform is applied to reference and
hypothesis alike, it is a fair (if lossy) comparison space.

English words are folded too ("the" -> "te"). That is intentional and
harmless: both sides get the same treatment, so English tokens still match
each other exactly.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It cannot recover a word the model never heard. Normalized WER is still a
real error rate; it just no longer charges the model for choosing a script.
Report BOTH numbers -- raw-script WER and normalized WER -- and the gap
between them is exactly the size of the script artifact.
"""
import re
import unicodedata

# --- Sinhala Unicode -> collapsed Latin -------------------------------------
# Values are already folded (aspirates -> plain, long vowels -> short,
# retroflex -> dental) so that output lands in the same space fuzzy_latin()
# maps romanized input into.

_VIRAMA = "්"  # hal kirima: kills the implicit vowel

_CONSONANTS = {
    "ක": "k",  "ඛ": "k",  "ග": "g",  "ඝ": "g",
    "ඞ": "n",  "ඟ": "ng",
    "ච": "c",  "ඡ": "c",  "ජ": "j",  "ඣ": "j",
    "ඤ": "n",  "ඥ": "n",  "ඦ": "nj",
    "ට": "t",  "ඨ": "t",  "ඩ": "d",  "ඪ": "d",
    "ණ": "n",  "ඬ": "nd",
    "ත": "t",  "ථ": "t",  "ද": "d",  "ධ": "d",
    "න": "n",  "ඳ": "nd",
    "ප": "p",  "ඵ": "p",  "බ": "b",  "භ": "b",
    "ම": "m",  "ඹ": "mb",
    "ය": "y",  "ර": "r",  "ල": "l",
    "ව": "v",  "ශ": "s",  "ෂ": "s",  "ස": "s",
    "හ": "h",  "ළ": "l",  "ෆ": "f",
}

_INDEPENDENT_VOWELS = {
    "අ": "a",  "ආ": "a",  "ඇ": "a",  "ඈ": "a",
    "ඉ": "i",  "ඊ": "i",  "උ": "u",  "ඌ": "u",
    "ඍ": "ru", "ඎ": "ru", "ඏ": "lu", "ඐ": "lu",
    "එ": "e",  "ඒ": "e",  "ඓ": "ai",
    "ඔ": "o",  "ඕ": "o",  "ඖ": "au",
}

_VOWEL_SIGNS = {
    "ා": "a",  "ැ": "a",  "ෑ": "a",
    "ි": "i",  "ී": "i",
    "ු": "u",  "ූ": "u",
    "ෘ": "ru",
    "ෙ": "e",  "ේ": "e",  "ෛ": "ai",
    "ො": "o",  "ෝ": "o",  "ෞ": "au",
    "ෟ": "lu",
}

# Combining marks that carry no vowel value for our purposes.
_SINHALA_SIGNS = {
    "ං": "n",   # anusvara
    "ඃ": "h",   # visarga
}

_ZERO_WIDTH = {"‍", "‌"}


def sinhala_to_latin(text):
    """Transliterate Sinhala Unicode to a collapsed Latin skeleton.

    Applies the abugida rule: a consonant carries an implicit /a/ unless it
    is immediately followed by a dependent vowel sign or the virama.
    Non-Sinhala characters pass through untouched.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _ZERO_WIDTH:
            i += 1
            continue
        if ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
            # Look past any zero-width joiner to the real next character.
            j = i + 1
            while j < n and text[j] in _ZERO_WIDTH:
                j += 1
            nxt = text[j] if j < n else ""
            if nxt == _VIRAMA:
                i = j + 1              # vowel killed, emit consonant only
            elif nxt in _VOWEL_SIGNS:
                out.append(_VOWEL_SIGNS[nxt])
                i = j + 1
            else:
                out.append("a")        # implicit vowel
                i += 1
            continue
        if ch in _INDEPENDENT_VOWELS:
            out.append(_INDEPENDENT_VOWELS[ch])
        elif ch in _SINHALA_SIGNS:
            out.append(_SINHALA_SIGNS[ch])
        elif ch in _VOWEL_SIGNS or ch == _VIRAMA:
            pass                       # stray sign with no consonant; drop
        else:
            out.append(ch)
        i += 1
    return "".join(out)


# --- Latin -> collapsed skeleton --------------------------------------------

# Aspirated / conventional digraphs folded onto the plain consonant. Applied
# as one alternation so folds cannot cascade into each other.
_DIGRAPHS = re.compile(r"(th|dh|kh|gh|ph|bh|jh|ch|sh)")
_DIGRAPH_MAP = {
    "th": "t", "dh": "d", "kh": "k", "gh": "g", "ph": "p",
    "bh": "b", "jh": "j", "ch": "c", "sh": "s",
}

# Collapses gemination and long vowels (aa/ee/nn/kk), which is what makes
# "eka"/"ekka" and "mokada"/"mokadda" unify across romanization conventions.
# Digits are deliberately excluded: this corpus is full of spoken card
# numbers, NICs and phone numbers, and a blanket `(.)\1+` turns 500 into 50,
# 000 into 0, and 002 into 02 — silently rewriting the exact content the
# downstream C4 PII work depends on.
_DOUBLED = re.compile(r"([^\W\d_])\1+", re.UNICODE)


def strip_diacritics(text):
    """Drop combining marks so ā/ṭ/ṇ/ḷ collapse onto a/t/n/l."""
    decomposed = unicodedata.normalize("NFD", text)
    kept = [c for c in decomposed if not unicodedata.combining(c)]
    return unicodedata.normalize("NFC", "".join(kept))


def fuzzy_latin(text):
    """Fold romanization-convention variation out of Latin text."""
    text = text.lower()
    text = strip_diacritics(text)
    text = _DIGRAPHS.sub(lambda m: _DIGRAPH_MAP[m.group(1)], text)
    text = text.replace("w", "v")      # ව is romanized both ways
    text = _DOUBLED.sub(r"\1", text)   # gemination + long vowels (aa, ee, ...)
    return text


# --- public scoring entry points --------------------------------------------

def _squash(text):
    """Strip punctuation/symbols, collapse whitespace — WITHOUT breaking Sinhala.

    The obvious `re.sub(r"[^\\w\\s]", " ", text)` is wrong for Sinhala and was
    silently corrupting every score computed with it. Sinhala is an abugida:
    vowels attach to consonants as *combining marks* (ි ා ෙ, and the virama ්),
    which are Unicode category Mn. Python's `\\w` does not match Mn, so that
    pattern replaces every vowel sign with a space and shatters each word into
    loose consonants — "ට්‍රිප් එකට" becomes "ට ර ප එකට".

    The damage is not neutral. Shredded text scores *better* than it should:
    a 50-word Sinhala utterance becomes ~50 single-character "words", and
    frequent letters (ක, න, ම, ප) then match the equally-shredded hypothesis
    by coincidence, crediting matches that never happened. Any WER measured
    that way is optimistic for exactly the recordings written in Sinhala.

    So filter by Unicode category instead, keeping letters (L*), numbers (N*)
    and marks (M*), and dropping punctuation (P*), symbols (S*) and control
    characters (C*).
    """
    out = []
    for ch in text:
        if ch.isspace() or unicodedata.category(ch)[0] in ("P", "S", "C"):
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def normalize_raw(text):
    """Baseline normalization: lowercase + strip punctuation only.

    This is what evaluate.py already did. Script differences survive it, so
    WER computed on this is 'raw-script WER'.
    """
    return _squash(text.lower())


def normalize_scripted(text):
    """Full script-normalizing pass: both scripts -> one phonetic skeleton.

    WER computed on this is 'normalized WER'.
    """
    text = sinhala_to_latin(text)
    text = fuzzy_latin(text)
    return _squash(text)


# --- self-test ---------------------------------------------------------------

_SELF_TEST = [
    # (Sinhala Unicode, human romanization as found in gold) -> must agree
    ("මම", "mama"),
    ("ගෙදර", "gedara"),
    ("තියෙනවා", "thiyenawa"),
    ("අලුත්", "aluth"),
    ("ඔයාට", "oyata"),
]


def _self_test():
    print("script_normalize self-test: Sinhala Unicode vs human romanization")
    print(f"{'sinhala':16} {'-> skeleton':16} {'romanized':16} {'-> skeleton':16} match")
    ok = True
    for sin, rom in _SELF_TEST:
        a = normalize_scripted(sin)
        b = normalize_scripted(rom)
        match = a == b
        ok = ok and match
        print(f"{sin:16} {a:16} {rom:16} {b:16} {'YES' if match else 'NO'}")
    print("\nAll agree." if ok else "\nSome disagree -- check the mapping tables.")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
