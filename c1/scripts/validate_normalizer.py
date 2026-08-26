#!/usr/bin/env python3
"""
validate_normalizer.py -- how lossy is the Task 1 script normalizer?

script_normalize.py folds two things at once, and they pull in opposite
directions:

  * WANTED   -- the same word written in different scripts or different
                romanization conventions must reach one skeleton, otherwise
                normalized WER is not measuring recognition at all.
                e.g. mokada / mokadda / mokadha / මොකද  ->  "mokada"

  * UNWANTED -- distinct Sinhala word forms must NOT collapse together, or
                the hypothesis gets credit for words it never produced.
                e.g. karana ("does") vs karanna ("to do") both -> "karana",
                because the gemination fold that fixes romanization variance
                also erases real morphological contrast.

The second kind cannot be removed without giving up the first: human
annotators genuinely spell the same word "eka" and "ekka", so a normalizer
that preserves gemination would fail to unify them. The fold is therefore a
deliberate trade, and this script measures its size so the trade is stated
with a number rather than assumed to be small.

Direction of the bias: over-merging can only ever make matching EASIER, so
normalized WER is a mild *lower bound* on true error -- optimistic, never
pessimistic. That matters for how the Task 1 result should be read.

Writes c1/results/c1_normalizer_validation.json.
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import script_normalize as sn

DATASET_ROOT = "/mnt/F/SLIIT/Research/SORA_Dataset"
C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(DATASET_ROOT, "annotations", "c1")
RESULTS_DIR = os.path.join(C1_ROOT, "results")

_SIN_RANGE = range(0x0D80, 0x0E00)


def has_sinhala(word):
    return any(ord(c) in _SIN_RANGE for c in word)


def main():
    words = []
    for path in sorted(glob.glob(os.path.join(GOLD_DIR, "*.transcript.json"))):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for utt in doc["utterances"]:
            words.extend(sn.normalize_raw(utt["text"]).split())

    vocab = set(words)
    by_skeleton = collections.defaultdict(set)
    for w in vocab:
        by_skeleton[sn.normalize_scripted(w)].add(w)

    merges = {k: v for k, v in by_skeleton.items() if len(v) > 1}

    # A merge that spans scripts is the normalizer doing its job: it is the
    # only way a romanized reference can ever match a Sinhala-Unicode
    # hypothesis. A merge *within* one script is pure loss of contrast.
    cross_script, within_script = {}, {}
    for skel, forms in merges.items():
        if any(has_sinhala(f) for f in forms) and any(not has_sinhala(f) for f in forms):
            cross_script[skel] = forms
        else:
            within_script[skel] = forms

    n_vocab = len(vocab)
    n_merged_items = sum(len(v) for v in merges.values())

    print("script_normalize collapse audit (gold C1 vocabulary)")
    print(f"  distinct raw word forms          : {n_vocab}")
    print(f"  distinct skeletons after folding : {len(by_skeleton)}")
    print(f"  skeletons covering >1 form       : {len(merges)}")
    print(f"  word forms involved in a merge   : {n_merged_items} "
          f"({100 * n_merged_items / n_vocab:.1f}% of vocabulary)")
    print()
    print(f"  cross-script merges (WANTED)     : {len(cross_script)}")
    print(f"  within-script merges (loss)      : {len(within_script)}")
    print()

    print("Largest cross-script merges — these are the normalizer working:")
    for skel, forms in sorted(cross_script.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {skel:16} <- {sorted(forms)}")
    print()
    print("Largest within-script merges — these are contrast genuinely lost:")
    for skel, forms in sorted(within_script.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {skel:16} <- {sorted(forms)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "c1_normalizer_validation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "note": "Merges make matching easier, so normalized WER is a mild "
                    "lower bound on true error (optimistic, never pessimistic).",
            "n_raw_vocab": n_vocab,
            "n_skeletons": len(by_skeleton),
            "n_merged_skeletons": len(merges),
            "n_vocab_items_merged": n_merged_items,
            "pct_vocab_merged": round(100 * n_merged_items / n_vocab, 2),
            "n_cross_script_merges": len(cross_script),
            "n_within_script_merges": len(within_script),
            "cross_script_examples": {
                k: sorted(v) for k, v in
                sorted(cross_script.items(), key=lambda x: -len(x[1]))[:25]
            },
            "within_script_examples": {
                k: sorted(v) for k, v in
                sorted(within_script.items(), key=lambda x: -len(x[1]))[:25]
            },
        }, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
