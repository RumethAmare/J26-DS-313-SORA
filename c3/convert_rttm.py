"""
Drop any .rttm file into this folder, then run:

    python convert_rttm.py

It converts every .rttm sitting here into an Audacity label file in
draft_labels/, ready to import.

No arguments. No paths to type. Works from anywhere as long as the .rttm is
next to this script.

It does NOT touch rttm_draft/ - that folder holds your baseline and must stay
exactly as the model produced it.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "draft_labels"

MIN_TURN = 0.15    # drop anything shorter - model flicker, not speech
MERGE_GAP = 0.30   # join same-speaker segments closer than this


def read_rttm(path):
    segs = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur = float(p[3]), float(p[4])
            segs.append((start, start + dur, p[7]))
    return sorted(segs)


def rename_speakers(segs):
    """SPEAKER_00 -> spk01, in order of first appearance."""
    mapping = {}
    for _, _, spk in segs:
        if spk not in mapping:
            mapping[spk] = f"spk{len(mapping) + 1:02d}"
    return [(s, e, mapping[k]) for s, e, k in segs], mapping


def merge_close(segs):
    out, n = [], 0
    for seg in sorted(segs):
        if out and out[-1][2] == seg[2] and seg[0] - out[-1][1] < MERGE_GAP:
            out[-1] = (out[-1][0], max(out[-1][1], seg[1]), seg[2])
            n += 1
            continue
        out.append(seg)
    return out, n


def main():
    files = sorted(HERE.glob("*.rttm"))

    if not files:
        print(f"No .rttm files found in:\n  {HERE}\n")
        print("Put the file you downloaded from Colab in that folder,")
        print("then run this again.  (If it came as a .zip, extract it first.)")
        return

    OUT.mkdir(exist_ok=True)
    print(f"found {len(files)} .rttm file(s)\n")

    for f in files:
        segs, mapping = rename_speakers(read_rttm(f))
        if not segs:
            print(f"{f.name}: no SPEAKER lines found - is this really an RTTM?")
            continue

        before = len(segs)
        segs, merged = merge_close(segs)
        dropped = [s for s in segs if s[1] - s[0] < MIN_TURN]
        segs = [s for s in segs if s[1] - s[0] >= MIN_TURN]
        segs, extra = merge_close(segs)
        merged += extra

        target = OUT / f"{f.stem}.txt"
        with open(target, "w", encoding="utf-8") as fh:
            for start, end, spk in segs:
                fh.write(f"{start:.3f}\t{end:.3f}\t{spk}\n")

        speakers = ",".join(sorted({s[2] for s in segs}))
        print(f"{f.name}")
        print(f"  {before} labels in  ->  {len(segs)} out "
              f"({len(dropped)} flicker dropped, {merged} merged)")
        print(f"  speakers: {speakers}")
        for s, e, spk in dropped:
            print(f"    dropped {s:.3f}-{e:.3f}  ({e - s:.3f}s)  {spk}")
        print(f"  written: draft_labels\\{target.name}")
        print()

    print("Now in Audacity:")
    print("  File > Open              processed\\<clip>.wav")
    print("  File > Import > Labels   draft_labels\\<clip>.txt")
    print()
    print("When you're done, export to corrected_labels\\ - NOT draft_labels.")


if __name__ == "__main__":
    main()
