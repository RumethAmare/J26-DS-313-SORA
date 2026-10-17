#!/usr/bin/env python3
"""
corpus.py -- one place for where the SORA dataset lives and which of its
recordings C1 may use.

Every C1 script used to hardcode the WSL path to the dataset and carry its own
copy of the exclusion list. Two of those copies had already drifted
(lid_data.py and evaluate_normalized.py), and the hardcoded path breaks the
moment the scripts run from Windows instead of WSL. Both now live here.

DATASET_ROOT resolves in this order:
  1. $SORA_DATASET_ROOT, if set
  2. the WSL mount  /mnt/F/SLIIT/Research/SORA_Dataset
  3. the Windows path  F:/SLIIT/Research/SORA_Dataset
"""
import os

_CANDIDATE_ROOTS = (
    "/mnt/F/SLIIT/Research/SORA_Dataset",
    "F:/SLIIT/Research/SORA_Dataset",
)


def _resolve_root():
    env = os.environ.get("SORA_DATASET_ROOT")
    if env:
        return env
    for root in _CANDIDATE_ROOTS:
        if os.path.isdir(root):
            return root
    # Fall through to the WSL path so error messages name the expected place.
    return _CANDIDATE_ROOTS[0]


DATASET_ROOT = _resolve_root()
GOLD_DIR = os.path.join(DATASET_ROOT, "annotations", "c1")
AUDIO_DIR = os.path.join(DATASET_ROOT, "processed", "audio")
MANIFEST_PATH = os.path.join(DATASET_ROOT, "manifests", "recordings_current.csv")

RID_PREFIX = "J26DS313_"
TOKENS_SUFFIX = ".tokens.jsonl"


def canonical_rid(rid):
    """'R0062' -> 'J26DS313_R0062'; already-prefixed ids pass through.

    R0062-R0065 were committed without the corpus prefix (in C1, C2 and C4).
    Canonicalising on read keeps them joinable with audio, manifest and
    prediction files without renaming anything in the shared dataset repo.
    """
    return rid if rid.startswith(RID_PREFIX) else RID_PREFIX + rid


# Recordings whose C1 gold is unusable, with the reason. Excluded from every
# C1 train/eval set: training on them teaches the wrong thing rather than
# merely adding noise. See docs/DATA_QUALITY_NOTES.md.
EXCLUDED_RECORDINGS = {
    "J26DS313_R0008": "token file structurally malformed (junk in timestamps)",
    "J26DS313_R0017": "every token tagged EN, including obvious romanized Sinhala",
    # R0053 (bulk EN, near-duplicate of R0052) was re-annotated 2026-10-04
    # and passes audit_gold.py; no longer excluded.
    "J26DS313_R0054": "every token tagged EN despite code-mixed speech",
    "J26DS313_R0055": "every token tagged EN despite code-mixed speech",
    "J26DS313_R0056": "290/291 tokens tagged EN despite code-mixed speech",
    "J26DS313_R0058": "369/373 tokens tagged SI, including plain English "
                      "('Customer service desk')",
    "J26DS313_R0059": "every token tagged EN despite code-mixed speech",
    # Partial mislabelling: a large share of unambiguous romanized-Sinhala
    # function words (eka, mata, oyata ...) tagged EN, measured by
    # audit_gold.py. If the easiest words are this wrong, the hard ones are
    # worse; c1_lid_report.md gives results with and without these.
    "J26DS313_R0051": "partial: 14% of Sinhala function words tagged EN",
    "J26DS313_R0062": "partial: 51% of Sinhala function words tagged EN",
    "J26DS313_R0063": "partial: 35% of Sinhala function words tagged EN",
    "J26DS313_R0064": "partial: 22% of Sinhala function words tagged EN",
    "J26DS313_R0065": "partial: 26% of Sinhala function words tagged EN",
}

# The subset above that is excluded for *partial* mislabelling, so a
# sensitivity run can put them back (SORA_C1_KEEP_PARTIAL=1).
PARTIAL_MISLABEL = {r for r, why in EXCLUDED_RECORDINGS.items()
                    if why.startswith("partial:")}
if os.environ.get("SORA_C1_KEEP_PARTIAL") == "1":
    EXCLUDED_RECORDINGS = {r: why for r, why in EXCLUDED_RECORDINGS.items()
                           if r not in PARTIAL_MISLABEL}


def gold_token_files():
    """(canonical_rid, path) for every gold tokens file, sorted by rid."""
    out = []
    for fname in os.listdir(GOLD_DIR):
        if fname.endswith(TOKENS_SUFFIX):
            rid = canonical_rid(fname[: -len(TOKENS_SUFFIX)])
            out.append((rid, os.path.join(GOLD_DIR, fname)))
    return sorted(out)


def gold_tokens_path(rid):
    """Path to a recording's gold tokens, whichever filename form it uses."""
    for name in (canonical_rid(rid), canonical_rid(rid)[len(RID_PREFIX):]):
        path = os.path.join(GOLD_DIR, name + TOKENS_SUFFIX)
        if os.path.exists(path):
            return path
    return os.path.join(GOLD_DIR, canonical_rid(rid) + TOKENS_SUFFIX)
