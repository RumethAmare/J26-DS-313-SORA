#!/usr/bin/env python3
"""
lid_rules.py -- rule-based SI/EN/OTHER tagging with calibrated confidence.

This is the same heuristic as SORA_Dataset/scripts/c1/lid_baseline.py
("Method A" in the plan), refactored so it can be imported by the token-stream
builder and so it reports *why* it made each call and *how sure* it is.

The label logic is unchanged, so accuracy is unchanged. What is added:

  * `lang_method` -- which rule fired, so mixed-provenance output stays
    auditable once Task 3 adds a fastText method alongside this one.
  * `lang_confidence` -- calibrated per rule rather than flat.

Calibration matters here because the rules are wildly unequal. A Sinhala
script match is deterministic; the numeric rule is guessing against gold data
that contradicts itself ("8"->EN but "075"->OTHER, per Section D). Reporting
one confidence for both would erase that, and downstream consumers (C2/C3/C4)
would have no way to tell a certainty from a coin flip.

See docs/TOKEN_SCHEMA.md for the full specification.
"""
import math
import re

from wordfreq import word_frequency

SINHALA_RE = re.compile(r"[඀-෿]")
TAMIL_RE = re.compile(r"[஀-௿]")
NUMERIC_RE = re.compile(r"^[\d.,:]+$")

EN_FREQ_THRESHOLD = 1e-5

# Script matches are deterministic; nothing else in the corpus produces these
# code points.
SCRIPT_CONFIDENCE = 0.99

# Deliberately uncertain. Gold's own numeric tagging is inconsistent, so a
# high confidence here would be asserting something the training signal does
# not support.
NUMERIC_CONFIDENCE = 0.50

# How many decades away from the threshold counts as "fully confident".
CONFIDENCE_SPAN_DECADES = 2.0

_STRIP = ".,:;!?\"'()"


def _scaled(distance_decades, floor, ceiling):
    """Map a log-frequency distance from the threshold onto [floor, ceiling]."""
    frac = min(1.0, max(0.0, distance_decades / CONFIDENCE_SPAN_DECADES))
    return round(floor + (ceiling - floor) * frac, 4)


def predict(token):
    """Return (lang, confidence, method) for one surface token."""
    if SINHALA_RE.search(token):
        return "SI", SCRIPT_CONFIDENCE, "unicode_sinhala"
    if TAMIL_RE.search(token):
        return "OTHER", SCRIPT_CONFIDENCE, "unicode_tamil"

    stripped = token.strip(_STRIP)
    if stripped and NUMERIC_RE.match(stripped):
        return "EN", NUMERIC_CONFIDENCE, "numeric"

    word = stripped.lower()
    if not word:
        # Punctuation-only token. Matches the baseline's EN default, but there
        # is no evidence at all, so say so.
        return "EN", 0.50, "numeric"

    freq = word_frequency(word, "en")
    if freq >= EN_FREQ_THRESHOLD:
        decades_above = math.log10(freq / EN_FREQ_THRESHOLD)
        return "EN", _scaled(decades_above, 0.50, 0.95), "wordfreq_en"

    # No English evidence -> assume romanized Sinhala, the dominant
    # Latin-script class in this corpus. This is negative evidence only, so
    # the ceiling is lower than the EN branch's.
    if freq <= 0:
        decades_below = CONFIDENCE_SPAN_DECADES  # unknown to English entirely
    else:
        decades_below = math.log10(EN_FREQ_THRESHOLD / freq)
    return "SI", _scaled(decades_below, 0.50, 0.90), "fallback_si"


def predict_lang(token):
    """Label only — matches lid_baseline.predict_lang for parity checking."""
    return predict(token)[0]


if __name__ == "__main__":
    samples = [
        "ආයුබෝවන්", "Golden", "එකට", "the", "mama", "oyata",
        "075", "8", "வணக்கம்", "thiyenawa", "report",
    ]
    print(f"{'token':16} {'lang':6} {'conf':>6}  method")
    for s in samples:
        lang, conf, method = predict(s)
        print(f"{s:16} {lang:6} {conf:>6.2f}  {method}")
