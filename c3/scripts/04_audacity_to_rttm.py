"""
STEP 4 - Turn your corrected Audacity labels into the three official files.

For each clip it writes:
    <clip>.rttm          who spoke when          (this is the ground truth)
    <clip>.uem           which parts count when scoring
    <clip>.events.jsonl  overlaps and back-channels

Label text you may use in Audacity (see ANNOTATION_RULES.md):
    spk01, spk02, spk03, spk04     normal speech
    spk02-BC                       a back-channel ("mm", "ow", "hari")
    NOSCORE                        unusable stretch, excluded from scoring

Rules applied automatically (Dataset_Building_Guide, section 4):
    * Gaps under 0.30 s inside the same speaker's turn are merged.
    * Overlaps are kept as-is - never collapsed to one speaker.
    * NOSCORE regions are cut out of the UEM.

Usage:
    python 04_audacity_to_rttm.py ../corrected_labels ../processed ../rttm_truth
"""

import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

MERGE_GAP = 0.30       # seconds - from the annotation guideline
NOSCORE_LABELS = {"noscore", "no-score", "skip", "unusable"}


def audio_duration(wav: Path) -> float:
    """Length of the audio in seconds.

    Tries ffprobe first (handles any format). Falls back to the standard
    library `wave` module, which is exact for the 16 kHz mono PCM WAVs this
    project uses and needs nothing installed. The fallback exists so the
    one-click console works on a machine without ffmpeg on PATH.
    """
    try:
        return float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(wav)]))
    except (OSError, subprocess.CalledProcessError, ValueError):
        import wave
        with wave.open(str(wav), "rb") as w:
            return w.getnframes() / float(w.getframerate())


def read_labels(path: Path):
    """Audacity label file -> [(start, end, text), ...]."""
    rows = []
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if not line.strip() or line.startswith("\\"):
            continue        # blank, or Audacity's frequency continuation line
        parts = line.split("\t")
        if len(parts) < 3:
            parts = line.split()
            if len(parts) < 3:
                print(f"  ! skipped unreadable line: {line!r}")
                continue
            parts = [parts[0], parts[1], " ".join(parts[2:])]
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            print(f"  ! skipped unreadable line: {line!r}")
            continue
        if end < start:
            start, end = end, start
        rows.append((start, end, parts[2].strip()))
    return sorted(rows)


def parse_label(text: str):
    """
    'spk02'      -> ('spk02', '')      normal turn
    'spk02-BC'   -> ('spk02', 'bc')    back-channel: said WHILE someone else talks
    'spk02-ACK'  -> ('spk02', 'ack')   minimal response: its own turn, nobody underneath
    """
    t = text.strip().lower()
    for suffix, kind in (("-bc", "bc"), ("-ack", "ack")):
        if t.endswith(suffix):
            return t[: -len(suffix)].strip(), kind
    return t, ""


def merge_close(segs):
    """Join same-speaker, same-kind segments separated by < MERGE_GAP."""
    out, merged = [], 0
    for seg in sorted(segs):
        if out:
            ps, pe, pspk, pkind = out[-1]
            s, e, spk, kind = seg
            if spk == pspk and kind == pkind and s - pe < MERGE_GAP:
                out[-1] = (ps, max(pe, e), pspk, pkind)
                merged += 1
                continue
        out.append(seg)
    return out, merged


def subtract(regions, holes, eps=1e-6):
    """Remove every hole from every region. Returns the surviving pieces."""
    result = []
    for rs, re_ in regions:
        pieces = [(rs, re_)]
        for hs, he in holes:
            nxt = []
            for ps, pe in pieces:
                if he <= ps or hs >= pe:
                    nxt.append((ps, pe))
                    continue
                if hs > ps:
                    nxt.append((ps, hs))
                if he < pe:
                    nxt.append((he, pe))
            pieces = nxt
        result.extend(p for p in pieces if p[1] - p[0] > eps)
    return result


def process(label_file: Path, wav_dir: Path, out_dir: Path) -> dict:
    clip = label_file.stem
    wav = wav_dir / f"{clip}.wav"
    if not wav.exists():
        print(f"  ! no audio found at {wav} - skipping")
        return {}

    duration = audio_duration(wav)
    rows = read_labels(label_file)

    speech, noscore, bad = [], [], []
    for start, end, text in rows:
        if text.lower() in NOSCORE_LABELS:
            noscore.append((start, end))
            continue
        spk, kind = parse_label(text)
        if not spk.startswith("spk"):
            bad.append(text)
            continue
        speech.append((start, end, spk, kind))

    if bad:
        print(f"  ! unrecognised labels (ignored): {sorted(set(bad))}")
        print("    speaker labels must look like spk01, spk02, spk02-BC, or NOSCORE")

    speech, merged = merge_close(speech)

    # --- RTTM. Speaker field is the clean id. The -BC tag lives in events.jsonl,
    #     because a scorer would otherwise treat spk02 and spk02-BC as two people
    #     and inflate the error rate. See README, "One deviation from the contract".
    rttm_path = out_dir / f"{clip}.rttm"
    with open(rttm_path, "w", encoding="utf-8") as fh:
        for start, end, spk, _k in speech:
            fh.write(f"SPEAKER {clip} 1 {start:.3f} {end - start:.3f} "
                     f"<NA> <NA> {spk} <NA> <NA>\n")

    # --- UEM: the whole clip, minus the NOSCORE holes.
    scored = subtract([(0.0, duration)], noscore)
    uem_path = out_dir / f"{clip}.uem"
    with open(uem_path, "w", encoding="utf-8") as fh:
        for start, end in scored:
            fh.write(f"{clip} 1 {start:.3f} {end:.3f}\n")

    # --- events: overlaps between different speakers, plus back-channels.
    events = []
    for (as_, ae, aspk, _), (bs, be, bspk, _) in combinations(speech, 2):
        if aspk == bspk:
            continue
        lo, hi = max(as_, bs), min(ae, be)
        if hi - lo > 0.01:
            events.append({"type": "overlap", "start": round(lo, 2),
                           "end": round(hi, 2),
                           "speakers": sorted([aspk, bspk])})
    mislabelled = []
    for start, end, spk, kind in speech:
        if not kind:
            continue
        # does anyone ELSE speak during this?
        covered = any(o_spk != spk and min(end, o_e) - max(start, o_s) > 0.01
                      for o_s, o_e, o_spk, _ in speech)
        if kind == "bc" and not covered:
            mislabelled.append((start, end, spk))
        events.append({
            "type": "backchannel" if covered else "acknowledgment",
            "start": round(start, 2), "end": round(end, 2), "speaker": spk,
            "tagged": kind,
        })
    events.sort(key=lambda e: (e["start"], e["type"]))

    ev_path = out_dir / f"{clip}.events.jsonl"
    with open(ev_path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    if mislabelled:
        print(f"  !! {len(mislabelled)} label(s) tagged -BC with nobody talking underneath.")
        print("     A back-channel happens WHILE another speaker holds the floor.")
        print("     If it stands alone it is a minimal response - re-tag it -ACK.")
        for s_, e_, spk_ in mislabelled:
            print(f"       {s_:7.2f} - {e_:7.2f}  {spk_}")

    speech_time = sum(e - s for s, e, _, _ in speech)
    overlap_time = sum(e["end"] - e["start"] for e in events
                       if e["type"] == "overlap")
    n_bc = sum(1 for e in events if e["type"] == "backchannel")
    n_ack = sum(1 for e in events if e["type"] == "acknowledgment")
    speakers = sorted({spk for _, _, spk, _ in speech})

    return {
        "clip": clip, "duration": duration, "turns": len(speech),
        "speakers": len(speakers), "speech_s": speech_time,
        "overlap_s": overlap_time, "overlap_pct":
            100 * overlap_time / speech_time if speech_time else 0.0,
        "backchannels": n_bc, "acks": n_ack,
        "noscore_s": sum(e - s for s, e in noscore),
        "merged": merged,
    }


def main(label_dir: str, wav_dir: str, out_dir: str) -> None:
    label_dir, wav_dir, out_dir = Path(label_dir), Path(wav_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(label_dir.glob("*.txt"))
    if not files:
        sys.exit(f"No label files (.txt) in {label_dir}")

    stats = []
    for f in files:
        print(f"\n{f.name}")
        s = process(f, wav_dir, out_dir)
        if s:
            stats.append(s)
            print(f"  {s['turns']} turns, {s['speakers']} speakers, "
                  f"{s['overlap_pct']:.1f}% overlap, "
                  f"{s['backchannels']} back-channels, {s['acks']} minimal responses"
                  + (f", {s['merged']} tiny gaps merged" if s['merged'] else ""))

    if not stats:
        return

    print("\n" + "=" * 78)
    print("CORPUS SNAPSHOT  (these numbers go in your Data Analysis Report)")
    print("=" * 78)
    print(f"{'clip':<14}{'len s':>8}{'turns':>7}{'spk':>5}"
          f"{'speech s':>10}{'overlap %':>11}{'BC':>5}{'ACK':>5}{'noscore s':>11}")
    for s in stats:
        print(f"{s['clip']:<14}{s['duration']:>8.1f}{s['turns']:>7}"
              f"{s['speakers']:>5}{s['speech_s']:>10.1f}"
              f"{s['overlap_pct']:>11.1f}{s['backchannels']:>5}{s['acks']:>5}"
              f"{s['noscore_s']:>11.1f}")

    tot_d = sum(s["duration"] for s in stats)
    tot_sp = sum(s["speech_s"] for s in stats)
    tot_ov = sum(s["overlap_s"] for s in stats)
    print("-" * 78)
    print(f"{'TOTAL':<14}{tot_d:>8.1f}{sum(s['turns'] for s in stats):>7}"
          f"{'':>5}{tot_sp:>10.1f}"
          f"{(100 * tot_ov / tot_sp if tot_sp else 0):>11.1f}"
          f"{sum(s['backchannels'] for s in stats):>5}"
          f"{sum(s['acks'] for s in stats):>5}"
          f"{sum(s['noscore_s'] for s in stats):>11.1f}")
    print(f"\n{tot_d / 60:.2f} minutes of audio, "
          f"{tot_ov:.1f} s of it with two people talking at once.")

    if tot_ov == 0:
        print("\n  !! ZERO overlap across every clip.")
        print("     Four people in one room for six minutes with no interruption")
        print("     at all is unlikely. Check you actually marked the moments")
        print("     where two voices run together - that is the whole point of C3.")

    print(f"\nWritten to: {out_dir}")
    print("Next: python 05_score.py")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
