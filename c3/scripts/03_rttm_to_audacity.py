"""
STEP 3 - Turn the machine's draft into something you can edit in Audacity.

Makes one label file per clip. In Audacity:
    File > Open              ../processed/Segment-N.wav
    File > Import > Labels   ../draft_labels/Segment-N.txt

You then drag the edges until they match what you actually hear.

TIDYING
-------
The model emits flicker at speaker changes - segments of 20-100 milliseconds
that no human said and that you cannot grab in Audacity at any sane zoom level.
Segment-2 had three of them, two lasting 17 ms. This script removes them.

It ONLY affects the editable labels. The model's real answer in ../rttm_draft is
never touched - that is your baseline hypothesis and it must stay exactly as the
model produced it, or your DER means nothing.

    --min-turn 0.15    drop anything shorter (default 0.15 s)
    --merge-gap 0.30   join same-speaker segments closer than this
                       (default 0.30 s, same threshold as the annotation guideline)
    --raw              no tidying, show exactly what the model said

Usage:
    python 03_rttm_to_audacity.py ../rttm_draft ../draft_labels
    python 03_rttm_to_audacity.py ../rttm_draft ../draft_labels --min-turn 0.25
"""

import argparse
import sys
from pathlib import Path


def read_rttm(path: Path):
    """Return [(start, end, speaker), ...] sorted by start time."""
    segs = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur, spk = float(p[3]), float(p[4]), p[7]
            segs.append((start, start + dur, spk))
    return sorted(segs)


def tidy_speaker_names(segs):
    """SPEAKER_00 / SPEAKER_01 -> spk01 / spk02, in order of first appearance."""
    order, mapping = [], {}
    for _, _, spk in segs:
        if spk not in mapping:
            mapping[spk] = f"spk{len(order) + 1:02d}"
            order.append(spk)
    return [(s, e, mapping[spk]) for s, e, spk in segs], mapping


def merge_close(segs, gap):
    """Join consecutive same-speaker segments separated by less than `gap`."""
    out, merged = [], 0
    for seg in sorted(segs):
        if out and out[-1][2] == seg[2] and seg[0] - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], seg[1]), seg[2])
            merged += 1
            continue
        out.append(seg)
    return out, merged


def drop_flicker(segs, min_turn):
    """Remove sub-threshold fragments. Returns (kept, dropped_list)."""
    kept = [s for s in segs if s[1] - s[0] >= min_turn]
    dropped = [s for s in segs if s[1] - s[0] < min_turn]
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rttm_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--min-turn", type=float, default=0.15)
    ap.add_argument("--merge-gap", type=float, default=0.30)
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()

    rttm_dir, out_dir = Path(args.rttm_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(rttm_dir.glob("*.rttm"))
    if not files:
        sys.exit(f"No .rttm files in {rttm_dir}")

    if args.raw:
        print("--raw: showing the model's output untouched\n")
    else:
        print(f"tidying: dropping turns under {args.min_turn}s, "
              f"merging same-speaker gaps under {args.merge_gap}s")
        print("(the model's own output in the rttm folder is NOT modified)\n")

    print(f"{'clip':<14}{'labels':>8}{'dropped':>9}{'merged':>8}   speakers")
    print("-" * 62)

    total_dropped = []
    for f in files:
        segs, mapping = tidy_speaker_names(read_rttm(f))
        n_before = len(segs)
        dropped, merged = [], 0

        if not args.raw:
            # merge first (a fragment may just be a split turn), then drop
            segs, merged = merge_close(segs, args.merge_gap)
            segs, dropped = drop_flicker(segs, args.min_turn)
            segs, extra = merge_close(segs, args.merge_gap)
            merged += extra

        target = out_dir / f"{f.stem}.txt"
        with open(target, "w", encoding="utf-8") as fh:
            for start, end, spk in segs:
                fh.write(f"{start:.3f}\t{end:.3f}\t{spk}\n")

        spk_list = ",".join(sorted({s[2] for s in segs}))
        print(f"{f.stem:<14}{len(segs):>8}{len(dropped):>9}{merged:>8}   {spk_list}")
        for d in dropped:
            total_dropped.append((f.stem, d))

    if total_dropped:
        print(f"\n{len(total_dropped)} flicker fragments removed:")
        for clip, (s, e, spk) in total_dropped:
            print(f"  {clip:<12} {s:7.3f} - {e:7.3f}  ({e - s:.3f}s)  {spk}")
        print("\nThese were model artefacts, not speech. If you disagree with any,")
        print("re-run with --raw and judge by ear.")

    print(f"\nLabel files written to: {out_dir}")
    print("\nIn Audacity, for each clip:")
    print("  1. File > Open              ../processed/Segment-N.wav")
    print("  2. File > Import > Labels   ../draft_labels/Segment-N.txt")
    print("  3. Fix by ear (see ANNOTATION_RULES.md)")
    print("  4. File > Export > Export Labels  ->  ../corrected_labels/Segment-N.txt")
    print("     ^ SAVE INTO corrected_labels, NOT draft_labels.")
    print("       Exporting over the draft loses your work silently.")


if __name__ == "__main__":
    main()
