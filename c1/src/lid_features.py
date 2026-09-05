#!/usr/bin/env python3
"""
Stage C feature functions -- what train_lid_crf.py (Phase 2) trains sklearn-crfsuite
against. Pure library module: no model, no training here, so it can be built and
tested now without waiting on the relabel patch or the R0017/R0023 correction --
those affect which LABEL a token should carry, not what FEATURES describe it.

WHY LEXICAL FEATURES, NOT JUST SCRIPT
--------------------------------------
docs/TOKEN_SCHEMA.md found 56 tokens (43 distinct) labelled EN but written in
Sinhala script -- English loanwords like haotel ("hotel"), roomsu ("rooms"),
nambara ("number"). A classifier built on script features alone gets every one
of these wrong by construction: the script says Sinhala, the label says
English. So alongside the script flag, every token also gets a folded-surface
lookup against a small closed set of known Sinhala-script loanwords, plus
generic lexical shape features (n-grams, length) that at least let the CRF
notice these words look different from ordinary Sinhala vocabulary.

The reverse case matters too: 56 SI tokens sit in Latin script (romanized
recordings), so "is_sinhala_script" alone would also mislabel every one of
those as English. Neither script nor a fixed lexicon is sufficient alone --
that is the whole reason this is a sequence model with a feature set, not a
lookup table.

Usage:
    python -m src.lid_features --selftest
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import sora_paths
from src.audit_labels import is_num
from src.script_fold import fold, is_sinhala_script

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover -- selftest reports this clearly instead
    zipf_frequency = None

# A handful of the actual loanwords TOKEN_SCHEMA.md found written in Sinhala
# script. Not exhaustive -- the CRF's job is to generalize past this list --
# but it gives the feature set a concrete, correct anchor for the hardest case
# rather than asking the model to learn it from character n-grams alone.
KNOWN_SI_SCRIPT_LOANWORDS = {
    fold(w) for w in [
        "හොටෙල්", "රූම්ස්", "එන්අයිසී", "නම්බර්", "ඇඩ්රස්",
        "ඇඩ්වාන්ස්", "ට්‍රිප්", "රිපෝට්",
    ]
}


def _char_ngrams(word, n, edge):
    """Prefix (edge='pre') or suffix (edge='suf') of length n, lowercase."""
    w = word.lower()
    if len(w) < n:
        return w
    return w[:n] if edge == "pre" else w[-n:]


def _en_zipf(word):
    """English frequency score, 0 (unknown/foreign-looking) to ~7 (very common).
    Only meaningful for Latin-script words; scored 0.0 otherwise."""
    if zipf_frequency is None or not word.isascii():
        return 0.0
    return round(zipf_frequency(word.lower(), "en"), 2)


def word_shape(token):
    """Coarse shape signature: 'Xx' (capitalized), 'xx' (lower), 'XX' (upper),
    or a digit/punctuation description -- classic NER-style feature."""
    if not token:
        return "empty"
    if is_num(token):
        return "num"
    if token[0].isupper() and token[1:].islower():
        return "Xx"
    if token.isupper():
        return "XX"
    if token.islower():
        return "xx"
    return "mixed"


def token_features(tokens, i):
    """
    Feature dict for tokens[i], the shape sklearn-crfsuite wants per position.

    `tokens` is a list of raw strings (surface forms) for one utterance, in
    order -- deliberately NOT the gold label list, so this function is exactly
    what runs at inference time too, with no access to answers.
    """
    tok = tokens[i]
    folded = fold(tok)
    sinhala_script = is_sinhala_script(tok)

    feats = {
        "bias": 1.0,
        "token.lower": tok.lower(),
        "token.folded": folded,
        "token.len": len(tok),
        "token.shape": word_shape(tok),
        "token.is_num": is_num(tok),
        "token.is_sinhala_script": sinhala_script,
        "token.has_apostrophe_zwj": "‍" in tok,
        "token.en_zipf": _en_zipf(tok),
        "token.known_si_loanword": folded in KNOWN_SI_SCRIPT_LOANWORDS,
        "token.prefix2": _char_ngrams(tok, 2, "pre"),
        "token.prefix3": _char_ngrams(tok, 3, "pre"),
        "token.suffix2": _char_ngrams(tok, 2, "suf"),
        "token.suffix3": _char_ngrams(tok, 3, "suf"),
        "position.is_first": i == 0,
        "position.is_last": i == len(tokens) - 1,
        "position.index": i,
    }

    # Neighbor LEXICAL features only -- never the neighbor's gold label. The
    # CRF's own transition potentials model label-to-label dependence; feeding
    # a neighbor's true label in as an input feature would leak the answer
    # during training and be unavailable at inference in the same form.
    for offset, prefix in ((-1, "prev"), (1, "next")):
        j = i + offset
        if 0 <= j < len(tokens):
            neighbor = tokens[j]
            feats[f"{prefix}.lower"] = neighbor.lower()
            feats[f"{prefix}.is_sinhala_script"] = is_sinhala_script(neighbor)
            feats[f"{prefix}.is_num"] = is_num(neighbor)
        else:
            feats[f"{prefix}.lower"] = "<BOS>" if offset < 0 else "<EOS>"

    return feats


def sequence_features(tokens):
    """One utterance -> list of feature dicts, sklearn-crfsuite's per-sequence
    input shape (X for one training/inference example)."""
    return [token_features(tokens, i) for i in range(len(tokens))]


def _selftest():
    """
    Run the feature extractor against real gold utterances, including R0017 --
    the hardest case in the corpus, per TOKEN_SCHEMA.md: its romanized Sinhala
    is annotated as EN corpus-wide (a known, documented mislabel, not a bug
    here), which makes it exactly the sequence where features should look
    genuinely ambiguous rather than trivially separable.

    This checks the code runs cleanly end-to-end and produces sane-looking
    features -- not model quality, which needs a trained CRF (Phase 2).
    """
    if zipf_frequency is None:
        print("WARNING: wordfreq not installed -- en_zipf will read 0.0 for "
              "every token. Install it before training the real CRF.")

    test_recordings = ["J26DS313_R0017", "J26DS313_R0002"]
    failures = 0

    for rid in test_recordings:
        try:
            tokens = sora_paths.load_tokens(rid)
        except FileNotFoundError:
            print(f"  SKIP {rid}: not found")
            continue

        # First utterance only, to keep the printout readable.
        first_utt = tokens[0]["utt_id"]
        utt_tokens = [t for t in tokens if t["utt_id"] == first_utt]
        surfaces = [t["token"] for t in utt_tokens]

        print(f"\n{rid} / {first_utt}: {len(surfaces)} tokens")
        try:
            feats = sequence_features(surfaces)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        for tok, gold, f in zip(surfaces, utt_tokens, feats):
            print(f"  {tok!r:20} gold={gold['lang']:5} "
                  f"shape={f['token.shape']:6} si_script={f['token.is_sinhala_script']!s:5} "
                  f"en_zipf={f['token.en_zipf']:5} loanword={f['token.known_si_loanword']}")

        if len(feats) != len(surfaces):
            print(f"  FAILED: expected {len(surfaces)} feature dicts, got {len(feats)}")
            failures += 1

    if failures:
        print(f"\n{failures} failure(s).")
        return 1
    print("\nFeature extraction ran cleanly on all test utterances.")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
