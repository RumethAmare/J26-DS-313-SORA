#!/usr/bin/env python3
"""
lid_data.py -- shared data loading + feature shaping for Task 3 LID work.

Kept separate from the trainer so that train_lid_fasttext.py and lid_hybrid.py
cannot drift apart in how they read gold, which recordings they exclude, or how
they turn a token into a fastText input line. A mismatch there would produce a
model that scores well in cross-validation and behaves differently in the
predictor, which is exactly the kind of bug that is invisible until it matters.
"""
import json
import os
import re

import lid_rules

DATASET_ROOT = "/mnt/F/SLIIT/Research/SORA_Dataset"
GOLD_DIR = os.path.join(DATASET_ROOT, "annotations", "c1")

# Excluded from all Task 3 train/eval sets per the Section D data-quality
# audit (docs/DATA_QUALITY_NOTES.md):
#   R0017 -- every token tagged EN including obvious romanized Sinhala
#   R0008 -- token file structurally malformed (junk in timestamps)
# Both are gold-data bugs, so training on them teaches the wrong thing rather
# than merely adding noise.
EXCLUDED_RECORDINGS = {"J26DS313_R0017", "J26DS313_R0008"}

LABELS = ["SI", "EN", "OTHER"]

_STRIP = ".,:;!?\"'()"
_HAS_DIGIT = re.compile(r"\d")


def token_script(token):
    """Which branch of the rule-based tagger owns this token.

    'latin' is the only class fastText ever sees: Sinhala and Tamil script are
    resolved deterministically by Unicode range, and pure numerics are a
    separate (badly-annotated) category of their own.
    """
    if lid_rules.SINHALA_RE.search(token):
        return "sinhala"
    if lid_rules.TAMIL_RE.search(token):
        return "tamil"
    stripped = token.strip(_STRIP)
    if stripped and lid_rules.NUMERIC_RE.match(stripped):
        return "numeric"
    return "latin"


def shape_features(token):
    """Orthographic markers appended to the fastText input line.

    Character n-grams alone miss capitalisation, because the token is
    lower-cased before n-gramming (otherwise 'Galle' and 'galle' share no
    subwords at all, which wastes the little data available). Capitalisation is
    the single strongest cue for the dominant error class here: 273 of the 430
    Latin-script heuristic errors are EN tokens misread as SI, and most are
    proper nouns -- 'Nawaloka', 'Dehiwala', 'Sampath', 'NIC'. Handing the model
    an explicit case marker recovers that signal.
    """
    feats = []
    stripped = token.strip(_STRIP)
    if stripped.isupper() and len(stripped) > 1:
        feats.append("#allcaps")
    elif stripped[:1].isupper():
        feats.append("#cap")
    else:
        feats.append("#lower")
    if _HAS_DIGIT.search(stripped):
        feats.append("#hasdigit")
    if not stripped.isalnum():
        feats.append("#punct")
    n = len(stripped)
    feats.append("#short" if n <= 3 else ("#med" if n <= 7 else "#long"))
    return feats


def to_fasttext_line(token, label=None):
    """One fastText input line: lower-cased surface + shape markers."""
    surface = token.strip(_STRIP).lower()
    if not surface:
        surface = "#empty"
    parts = [surface] + shape_features(token)
    text = " ".join(parts)
    return f"__label__{label} {text}" if label else text


def load_gold_tokens(exclude=True):
    """All gold tokens as dicts, grouped nowhere -- caller decides.

    Each dict carries the original fields plus 'recording' and 'script'.
    """
    out = []
    for fname in sorted(os.listdir(GOLD_DIR)):
        if not fname.endswith(".tokens.jsonl"):
            continue
        rid = fname[: -len(".tokens.jsonl")]
        if exclude and rid in EXCLUDED_RECORDINGS:
            continue
        with open(os.path.join(GOLD_DIR, fname), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                tok = json.loads(line)
                tok["recording"] = rid
                tok["script"] = token_script(tok["token"])
                out.append(tok)
    return out


def per_class_metrics(pairs, labels=None):
    """pairs = [(gold, pred), ...] -> {label: {precision, recall, f1, support}}."""
    labels = labels or LABELS
    out = {}
    for lbl in labels:
        tp = sum(1 for g, p in pairs if g == lbl and p == lbl)
        support = sum(1 for g, _ in pairs if g == lbl)
        predicted = sum(1 for _, p in pairs if p == lbl)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        out[lbl] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "predicted": predicted,
        }
    correct = sum(1 for g, p in pairs if g == p)
    out["_accuracy"] = round(correct / len(pairs), 4) if pairs else 0.0
    out["_n"] = len(pairs)
    out["_macro_f1"] = round(
        sum(out[l]["f1"] for l in labels) / len(labels), 4)
    return out


def make_folds(recordings, k=5, key=None):
    """Assign whole recordings to k folds, balancing a per-recording statistic.

    Folding by recording (never by token) is what keeps cross-validation
    honest: tokens from one conversation share speaker, topic and vocabulary,
    so splitting them across train and test leaks. Recordings are sorted by
    `key` and dealt out in a snake order so each fold gets a comparable mix
    rather than one fold collecting all the English-heavy conversations.
    """
    ordered = sorted(recordings, key=key) if key else sorted(recordings)
    folds = [[] for _ in range(k)]
    for i, rid in enumerate(ordered):
        cycle, pos = divmod(i, k)
        idx = pos if cycle % 2 == 0 else (k - 1 - pos)   # snake
        folds[idx].append(rid)
    return folds
