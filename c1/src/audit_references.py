#!/usr/bin/env python3
"""
Audit reference quality, so WER numbers can be trusted before they are quoted.

Phase 0 turned up two defects that a WER table alone hides, because a bad
reference and a bad model both just look like a high number:

  R0008  22 gold words for 62.5 s of audio (0.35 words/s) with 56% of the
         timeline uncovered, while Whisper found 111 words. The reference
         transcribes a fraction of the speech, so its WER of 5.045 measures
         the annotation, not the model.

  R0017  237 tokens labelled EN and zero labelled SI, in a recording the
         manifest declares as SI+EN — and whose text is plainly code-mixed
         ("Oyata connection ekak oneda?", "mama Chaminda"). Every Sinhala word
         carries an EN label.

Both are cheap to detect mechanically, and both must be settled before a WER
or an LID F1 goes in a report.

Checks
------
1. Word density (words per second of audio). Conversational Sinhala-English
   runs 1.5-2.7 w/s here; well under that means an incomplete reference.
2. Timeline coverage — how much of the audio any utterance spans.
3. Manifest agreement — a recording the manifest calls SI+EN that has zero
   tokens of one of those languages.

Usage:
    python -m src.audit_references
"""
import csv
import json
import os
import sys
from collections import Counter

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import sora_paths

# Below this, a reference is too sparse to be a complete transcript. The corpus
# median is ~1.85 w/s; 1.2 sits well under every plausible recording and only
# catches the genuine outlier.
MIN_WORDS_PER_SEC = 1.2
MAX_TIMELINE_GAP = 0.55  # fraction of audio with no utterance over it


def manifest_languages():
    """recording_id -> declared language set, e.g. {'SI','EN'} from 'SI+EN'."""
    out = {}
    with open(sora_paths.manifest_path(), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            langs = (row.get("languages") or "").replace(" ", "")
            out[row["recording_id"]] = {p for p in langs.split("+") if p}
    return out


def audit_one(rid, declared, duration):
    transcript = sora_paths.load_transcript(rid)
    utts = transcript.get("utterances", [])
    text = " ".join(u["text"] for u in utts)
    n_words = len(text.split())

    covered = sum(max(0.0, u["end"] - u["start"]) for u in utts)
    wps = n_words / duration if duration else 0.0
    gap = 1 - (covered / duration) if duration else 1.0

    labels = Counter(t.get("lang") for t in sora_paths.load_tokens(rid))

    issues = []
    if wps < MIN_WORDS_PER_SEC:
        issues.append(f"sparse-reference: {wps:.2f} words/s")
    if gap > MAX_TIMELINE_GAP:
        issues.append(f"timeline-gap: {gap:.0%} of audio uncovered")
    for lang in declared:
        if lang in ("SI", "EN") and labels.get(lang, 0) == 0:
            issues.append(f"manifest-declares-{lang}-but-zero-{lang}-tokens")

    return {
        "recording": rid,
        "duration_s": round(duration, 1),
        "n_words": n_words,
        "words_per_sec": round(wps, 2),
        "timeline_gap": round(gap, 3),
        "declared_languages": sorted(declared),
        "label_counts": dict(labels),
        "issues": issues,
    }


def main():
    declared_by_rid = manifest_languages()
    durations = {}
    with open(sora_paths.manifest_path(), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            durations[row["recording_id"]] = float(row.get("duration_s") or 0)

    results = []
    for rid in sora_paths.gold_recording_ids():
        results.append(audit_one(rid, declared_by_rid.get(rid, set()),
                                 durations.get(rid, 0.0)))

    flagged = [r for r in results if r["issues"]]

    print("=" * 78)
    print("REFERENCE AUDIT — can these WER numbers be trusted?")
    print("=" * 78)
    print(f"\n{'recording':22} {'dur_s':>7} {'words':>6} {'w/s':>5} {'gap':>6}  issues")
    for r in sorted(results, key=lambda r: r["words_per_sec"]):
        flag = "; ".join(r["issues"]) if r["issues"] else ""
        print(f"{r['recording']:22} {r['duration_s']:>7.1f} {r['n_words']:>6} "
              f"{r['words_per_sec']:>5.2f} {r['timeline_gap']:>5.0%}  {flag}")

    print(f"\n{len(flagged)} of {len(results)} recordings flagged.\n")
    for r in flagged:
        print(f"  {r['recording']}: {'; '.join(r['issues'])}")
        print(f"      labels={r['label_counts']}  declared={r['declared_languages']}")

    exclude = [r["recording"] for r in flagged
               if any(i.startswith(("sparse-reference", "timeline-gap"))
                      for i in r["issues"])]
    relabel = [r["recording"] for r in flagged
               if any("manifest-declares" in i for i in r["issues"])]

    if exclude:
        print(f"\nRecommend EXCLUDING from ASR scoring (reference is incomplete, "
              f"so WER measures the annotation): {exclude}")
    if relabel:
        print(f"Recommend RELABELLING before any LID training or evaluation "
              f"(language labels contradict the manifest): {relabel}")

    out = {
        "thresholds": {
            "min_words_per_sec": MIN_WORDS_PER_SEC,
            "max_timeline_gap": MAX_TIMELINE_GAP,
        },
        "n_recordings": len(results),
        "n_flagged": len(flagged),
        "recommend_exclude_from_asr_scoring": exclude,
        "recommend_relabel_before_lid": relabel,
        "per_recording": results,
    }
    sora_paths.ensure_dir(sora_paths.data_dir())
    path = os.path.join(sora_paths.data_dir(), "reference_audit.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
