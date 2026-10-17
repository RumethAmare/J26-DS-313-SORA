"""
STEP 5 - Score the machine against your corrected annotation.

Answers one question: how much of the conversation did pyannote get wrong,
before you improve anything? That is your "before" number.

PROTOCOL v1 (C3_Diarization_Roadmap, section 2) is hard-wired here:
    * Overlap is INCLUDED (skip_overlap = False). No hiding the hard part.
    * Reported at collar 0.0 AND collar 0.25.
    * The UEM file is passed in explicitly - nothing is approximated.
    * The error is broken into its three parts, plus speaker-count accuracy.

"Collar" = a small forgiveness window at each boundary. 0.0 forgives nothing;
0.25 forgives a quarter-second either side, which is the usual convention.
Both get reported so the number is never ambiguous.

Setup, once:
    pip install pyannote.metrics

Usage:
    python 05_score.py ../rttm_truth ../rttm_draft ../results
"""

import csv
import sys
from datetime import date
from pathlib import Path

COLLARS = [0.0, 0.25]

# Which model produced the hypothesis folder. Printed in the header and the log
# row so a number is never separated from the model that made it. Override with
# a 4th argument if you score a different run.
MODEL = "pyannote/speaker-diarization-community-1 (pyannote.audio 4.x)"


def load_annotation(path: Path, uri: str):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=uri)
    for i, line in enumerate(open(path, encoding="utf-8")):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur, spk = float(p[3]), float(p[4]), p[7]
            if dur <= 0:
                continue
            ann[Segment(start, start + dur), f"t{i}"] = spk
    return ann


def load_uem(path: Path, uri: str):
    from pyannote.core import Segment, Timeline
    tl = Timeline(uri=uri)
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 4:
            tl.add(Segment(float(p[2]), float(p[3])))
    return tl


def main(truth_dir: str, hyp_dir: str, out_dir: str) -> None:
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        sys.exit("pyannote.metrics is not installed.  ->  pip install pyannote.metrics")

    truth_dir, hyp_dir, out_dir = Path(truth_dir), Path(hyp_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truths = sorted(truth_dir.glob("*.rttm"))
    if not truths:
        sys.exit(f"No ground-truth RTTM in {truth_dir}. Run step 4 first.")

    rows = []
    pooled = {c: {"confusion": 0.0, "missed detection": 0.0,
                  "false alarm": 0.0, "total": 0.0} for c in COLLARS}

    for tf in truths:
        clip = tf.stem
        hf = hyp_dir / f"{clip}.rttm"
        uf = truth_dir / f"{clip}.uem"
        if not hf.exists():
            print(f"  ! no machine output for {clip} - skipped")
            continue

        ref = load_annotation(tf, clip)
        hyp = load_annotation(hf, clip)
        uem = load_uem(uf, clip) if uf.exists() else None
        if uem is None:
            print(f"  ! no .uem for {clip}. Scoring the whole clip. "
                  f"Protocol v1 wants an explicit UEM - regenerate it in step 4.")

        row = {"clip": clip,
               "true_speakers": len(ref.labels()),
               "found_speakers": len(hyp.labels())}

        for collar in COLLARS:
            metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
            comp = metric(ref, hyp, uem=uem, detailed=True)
            total = comp["total"]
            der = ((comp["confusion"] + comp["missed detection"]
                    + comp["false alarm"]) / total * 100) if total else 0.0
            tag = f"c{collar}"
            row[f"DER_{tag}"] = der
            row[f"miss_{tag}"] = comp["missed detection"] / total * 100 if total else 0.0
            row[f"fa_{tag}"] = comp["false alarm"] / total * 100 if total else 0.0
            row[f"conf_{tag}"] = comp["confusion"] / total * 100 if total else 0.0
            for k in pooled[collar]:
                pooled[collar][k] += comp[k]

        rows.append(row)

    if not rows:
        sys.exit("Nothing scored.")

    print("\n" + "=" * 84)
    print(f"ZERO-SHOT BASELINE - {MODEL}")
    print("no speaker count given, overlap included, explicit UEM")
    print("=" * 84)
    for collar in COLLARS:
        t = f"c{collar}"
        print(f"\n--- collar {collar} " + "-" * 62)
        print(f"{'clip':<14}{'DER %':>9}{'missed':>9}{'extra':>9}"
              f"{'mixed up':>10}{'spk true':>10}{'spk found':>11}")
        for r in rows:
            print(f"{r['clip']:<14}{r[f'DER_{t}']:>9.2f}{r[f'miss_{t}']:>9.2f}"
                  f"{r[f'fa_{t}']:>9.2f}{r[f'conf_{t}']:>10.2f}"
                  f"{r['true_speakers']:>10}{r['found_speakers']:>11}")
        p = pooled[collar]
        pooled_der = ((p["confusion"] + p["missed detection"] + p["false alarm"])
                      / p["total"] * 100) if p["total"] else 0.0
        print("-" * 72)
        print(f"{'ALL CLIPS':<14}{pooled_der:>9.2f}"
              f"{p['missed detection'] / p['total'] * 100:>9.2f}"
              f"{p['false alarm'] / p['total'] * 100:>9.2f}"
              f"{p['confusion'] / p['total'] * 100:>10.2f}")

    n_right = sum(1 for r in rows if r["true_speakers"] == r["found_speakers"])
    print(f"\nSpeaker count correct on {n_right} of {len(rows)} clips.")

    stamp = date.today().isoformat()
    csv_path = out_dir / f"baseline_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    print("\n" + "-" * 84)
    print("Paste this line into C3_Experiment_Log (one row, protocol stated):")
    p0, p25 = pooled[0.0], pooled[0.25]
    d0 = (p0["confusion"] + p0["missed detection"] + p0["false alarm"]) / p0["total"] * 100
    d25 = (p25["confusion"] + p25["missed detection"] + p25["false alarm"]) / p25["total"] * 100
    print(f"  E003 | {stamp} | zero-shot baseline, pilot | 7 clips, "
          f"{len(rows)} scored, pilot v0.0 | {MODEL}, defaults, no speaker "
          f"count | DER {d0:.2f}% @collar0.0, {d25:.2f}% @collar0.25 | "
          f"overlap included | explicit UEM | PROCESS TEST - too little audio "
          f"to quote as a research result")
    print("-" * 84)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print(__doc__)
        sys.exit(1)
    if len(sys.argv) == 5:
        MODEL = sys.argv[4]
    main(sys.argv[1], sys.argv[2], sys.argv[3])
