#!/usr/bin/env python3
"""
Build c1/data/manifest.csv — the locked train/eval split.

Derives from the team's manifests/recordings_current.csv, adding three columns
this component needs:

  split         train | eval        (assigned once, then never moved)
  script_group  native | romanized | mixed | english_only
  has_audio     whether processed/audio/<rid>.wav exists

WHY LOCK IT NOW
---------------
The corpus is 52 minutes today against the 15-20 h that
C1_FINAL_IMPLEMENTATION.md §6.2 assumes. The split still has to be fixed before
any number is measured, because Phase 2's "before vs. after fine-tuning" claim
is only meaningful if both sides were scored on the same untouched recordings.
Recordings added later join the train pool; the eval set only ever grows by an
explicit, recorded decision.

The split is stratified by script_group so the native-vs-romanized comparison
that drives the Phase 0 decoding-mode decision is measurable on the eval set
too, not just on the corpus as a whole.

Usage:
    python -m src.build_manifest            # writes, refuses to clobber
    python -m src.build_manifest --force    # re-roll the split (rarely correct)
"""
import argparse
import csv
import os
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import sora_paths
from src.script_fold import is_sinhala_script

SEED = 313  # J26-DS-313
EVAL_TARGET_FRACTION = 0.28  # of duration, per script group
NATIVE_SHARE_MIN = 0.10      # share of words in Sinhala script to call it native

OUT_PATH = os.path.join(sora_paths.data_dir(), "manifest.csv")

FIELDNAMES = [
    "recording_id", "split", "script_group", "has_audio", "duration_s",
    "n_tokens", "n_si_native", "n_si_latin", "speakers", "status", "source_group",
]


def classify_script(rid):
    """
    Which script this recording is written in, from the transcript TEXT.

    Deliberately independent of the language labels. An earlier version counted
    SI-labelled tokens, which made the answer only as good as the annotation —
    and R0017, whose Sinhala is entirely mislabelled EN, came out as
    "english_only" purely because it had no SI-labelled tokens. Script is a
    property of the characters on the page, so it is read from the characters.

    Counts word-level: a token is native if it contains any Sinhala-block
    character, Latin if it has letters and none. Digits and punctuation are
    script-neutral and ignored.
    """
    native = latin = 0
    for utt in sora_paths.load_transcript(rid).get("utterances", []):
        for word in utt.get("text", "").split():
            if is_sinhala_script(word):
                native += 1
            elif any(c.isalpha() for c in word):
                latin += 1

    # The question is which script the recording writes SINHALA in, not what
    # share of the text is Sinhala — these are heavily code-mixed, so even a
    # native-script recording is 30-50% Latin English by word count. Presence
    # of native script is therefore the signal, not its majority.
    total = native + latin
    if total == 0:
        group = "empty"
    elif native == 0:
        group = "romanized"
    elif native / total >= NATIVE_SHARE_MIN:
        group = "native"
    else:
        group = "mixed"  # a handful of native words in an otherwise Latin transcript

    tokens = sora_paths.load_tokens(rid)
    return group, len(tokens), native, latin


def assign_splits(rows):
    """
    Greedy duration-stratified assignment, seeded so it reproduces exactly.

    Within each script group, shuffle and take recordings into eval until the
    group's eval share of duration reaches the target. Groups with fewer than
    two recordings go entirely to train: a singleton in eval would leave the
    train pool with zero examples of that group, which is worse than an eval
    set that under-represents it.
    """
    rng = random.Random(SEED)
    by_group = {}
    for row in rows:
        by_group.setdefault(row["script_group"], []).append(row)

    for group, members in sorted(by_group.items()):
        for row in members:
            row["split"] = "train"

        if len(members) < 2:
            print(f"  {group:13} n={len(members)} -> all train (too few to split)")
            continue

        total = sum(float(r["duration_s"]) for r in members)
        order = sorted(members, key=lambda r: r["recording_id"])
        rng.shuffle(order)

        chosen = 0.0
        for row in order:
            if chosen / total >= EVAL_TARGET_FRACTION:
                break
            row["split"] = "eval"
            chosen += float(row["duration_s"])

        n_eval = sum(1 for r in members if r["split"] == "eval")
        print(f"  {group:13} n={len(members):2}  eval={n_eval} "
              f"({chosen/60:.1f}/{total/60:.1f} min = {chosen/total:.0%})")
    return rows


def reclassify():
    """
    Refresh the descriptive columns without touching a single split assignment.

    `script_group` is a description, not the split, so improving how it is
    derived should not cost the lock — re-rolling would silently change which
    recordings are held out and invalidate every comparison already reported.
    """
    if not os.path.exists(OUT_PATH):
        print(f"{OUT_PATH} does not exist yet. Run without --reclassify first.")
        return 1

    with open(OUT_PATH, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    changes = []
    for row in rows:
        rid = row["recording_id"]
        group, n_tokens, native, latin = classify_script(rid)
        if row["script_group"] != group:
            changes.append((rid, row["script_group"], group))
        row["script_group"] = group
        row["n_tokens"] = n_tokens
        row["n_si_native"] = native
        row["n_si_latin"] = latin

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Reclassified {len(rows)} recordings; splits untouched.")
    if changes:
        print(f"{len(changes)} script_group change(s):")
        for rid, old, new in changes:
            print(f"  {rid}: {old} -> {new}")
    else:
        print("No script_group changed.")
    print(f"\nWrote {OUT_PATH}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing split (this invalidates prior comparisons)")
    ap.add_argument("--reclassify", action="store_true",
                    help="refresh script_group and token counts, keeping every "
                         "split assignment exactly as it is")
    args = ap.parse_args()

    if args.reclassify:
        return reclassify()

    if os.path.exists(OUT_PATH) and not args.force:
        print(f"{OUT_PATH} already exists — the split is locked.")
        print("Re-rolling it invalidates every before/after comparison already "
              "reported. Pass --force only if that is genuinely intended.")
        return 1

    upstream = {}
    with open(sora_paths.manifest_path(), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            upstream[row["recording_id"]] = row

    rows = []
    excluded = []
    for rid in sora_paths.gold_recording_ids():
        if not sora_paths.has_audio(rid):
            excluded.append(rid)
            continue

        group, n_tokens, n_native, n_latin = classify_script(rid)
        up = upstream.get(rid, {})
        rows.append({
            "recording_id": rid,
            "split": "",
            "script_group": group,
            "has_audio": "true",
            "duration_s": up.get("duration_s") or "0",
            "n_tokens": n_tokens,
            "n_si_native": n_native,
            "n_si_latin": n_latin,
            "speakers": up.get("speakers", ""),
            "status": up.get("status", "UNKNOWN"),
            "source_group": up.get("source_group", ""),
        })

    if excluded:
        print(f"Excluded (gold tokens but no .wav, so unscorable): {excluded}\n")

    print(f"Splitting {len(rows)} scorable recordings (seed={SEED}, "
          f"target {EVAL_TARGET_FRACTION:.0%} of duration to eval):")
    assign_splits(rows)

    rows.sort(key=lambda r: r["recording_id"])
    sora_paths.ensure_dir(sora_paths.data_dir())
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    eval_rows = [r for r in rows if r["split"] == "eval"]
    train_rows = [r for r in rows if r["split"] == "train"]
    eval_min = sum(float(r["duration_s"]) for r in eval_rows) / 60
    train_min = sum(float(r["duration_s"]) for r in train_rows) / 60

    print(f"\n  train: {len(train_rows):2} recordings, {train_min:5.1f} min")
    print(f"  eval : {len(eval_rows):2} recordings, {eval_min:5.1f} min "
          f"-> {', '.join(r['recording_id'] for r in eval_rows)}")
    print(f"\nWrote {OUT_PATH}")
    print("This split is now LOCKED. Later phases must import src.split_guard "
          "rather than re-deriving it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
