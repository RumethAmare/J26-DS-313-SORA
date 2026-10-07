#!/usr/bin/env python3
"""
pretag_relabel.py -- draft language labels for the bulk-mislabelled recordings.

Twelve gold recordings have bulk or partial language mislabelling
(see corpus.EXCLUDED_RECORDINGS and results/c1_gold_audit.md). Re-labelling
~2,600 tokens by hand from scratch is slow; correcting a good draft is not.
This writes that draft using the Task 3 hybrid LID (Unicode rules + fastText),
which was trained without any of these recordings, so it is not echoing their
bad labels back.

THIS IS NOT GOLD. Output goes to c1/predictions/pretag/, never into the shared
dataset repo. A person must review it before it replaces anything there.

Each output token keeps every original field and adds:
  lang                  the suggested label
  lang_original         what the gold file said
  lang_confidence       hybrid LID confidence
  lang_method           which rule / model produced it
  switch                re-derived from the suggested `lang`, within utterance
  needs_review          True where the suggestion differs from the original or
                        confidence is below REVIEW_CONFIDENCE

Usage:
    python pretag_relabel.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corpus
from lid_hybrid import HybridLID

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(C1_ROOT, "predictions", "pretag")
REPORT_PATH = os.path.join(C1_ROOT, "results", "c1_pretag_summary.md")

# Only recordings whose problem is the *labels*. R0008's labels are fine (its
# timestamps are the problem), so a label draft would not help it.
PRETAG_RECORDINGS = [
    "J26DS313_R0017", "J26DS313_R0054", "J26DS313_R0055",
    "J26DS313_R0056", "J26DS313_R0058", "J26DS313_R0059",
    # partial mislabelling -- here the draft mostly flags the slipped tokens
    "J26DS313_R0051", "J26DS313_R0062", "J26DS313_R0063", "J26DS313_R0064",
    "J26DS313_R0065",
]

REVIEW_CONFIDENCE = 0.8


def main():
    lid = HybridLID()
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []

    for rid in PRETAG_RECORDINGS:
        path = corpus.gold_tokens_path(rid)
        toks = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

        out, prev_lang, prev_utt = [], None, None
        for t in toks:
            lang, conf, method = lid.predict(t["token"])
            if t.get("utt_id") != prev_utt:
                prev_lang = None
            row = dict(t)
            row["lang_original"] = t.get("lang")
            row["lang"] = lang
            row["lang_confidence"] = conf
            row["lang_method"] = method
            row["switch"] = prev_lang is not None and lang != prev_lang
            row["needs_review"] = lang != t.get("lang") or conf < REVIEW_CONFIDENCE
            out.append(row)
            prev_lang, prev_utt = lang, t.get("utt_id")

        out_path = os.path.join(OUT_DIR, f"{rid}.tokens.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in out:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        counts = {l: sum(1 for r in out if r["lang"] == l) for l in ("SI", "EN", "OTHER")}
        changed = sum(1 for r in out if r["lang"] != r["lang_original"])
        review = sum(1 for r in out if r["needs_review"])
        summary.append((rid, len(out), counts, changed, review,
                        sum(1 for r in out if r["switch"])))
        print(f"{rid}: {len(out)} tokens, suggested SI={counts['SI']} EN={counts['EN']} "
              f"OTHER={counts['OTHER']}, changed {changed}, needs review {review}")

    L = ["# Draft re-labels for bulk-mislabelled recordings\n\n",
         "Produced by `scripts/pretag_relabel.py` using the Task 3 hybrid LID "
         f"(`{os.path.relpath(lid.model_path, C1_ROOT)}`). **Drafts, not gold** — "
         "files are in `predictions/pretag/` and must be reviewed before replacing "
         "anything in `SORA_Dataset/annotations/c1/`.\n\n",
         f"A token is marked `needs_review` if the suggestion differs from the "
         f"original label or its confidence is below {REVIEW_CONFIDENCE}. The hybrid's "
         "cross-validated token accuracy is ~93%, so roughly 1 suggestion in 15 "
         "will be wrong; it never predicts OTHER, so Tamil words (e.g. "
         "`Vanakkam`) must be fixed by hand.\n\n",
         "| recording | tokens | suggested SI | suggested EN | labels changed "
         "| needs review | switches |\n|---|---|---|---|---|---|---|\n"]
    for rid, n, c, changed, review, sw in summary:
        L.append(f"| {rid} | {n} | {c['SI']} | {c['EN']} | {changed} | {review} | {sw} |\n")
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.writelines(L)
    print(f"\nWrote {OUT_DIR}\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
