#!/usr/bin/env python3
"""
build_token_stream.py -- Task 2: end-to-end C1 token stream.

Whisper word-level decode -> per-token LID -> switch computation -> one
`c1/predictions/token_stream/<rid>.tokens.jsonl` per recording, in the
extended schema documented in docs/TOKEN_SCHEMA.md.

This is the glue that turns C1's separate pieces into the artifact C2/C3/C4
actually consume: words with timestamps, an ASR confidence, a language tag, a
language confidence, and switch-point markers.

Two things worth knowing before reading the output:

1. The ASR underneath is the off-the-shelf baseline, which Task 1 measured at
   ~0.98-1.00 WER on this corpus. The token stream is therefore structurally
   correct and substantively unreliable. It exists so the schema and the
   downstream contract can be built and tested now; the numbers in it should
   not be used as evidence of anything until fine-tuning happens.

2. The `--language` choice materially changes what comes out. Task 1 found
   Sinhala-mode decoding *abstains* (emits ~25% of expected words) while
   English-forced decoding over-generates but recovers ~40x more reference
   words. Neither is "correct"; they fail differently. Default is auto-detect
   because that is the honest deployment default, but pass `--language en` to
   reproduce the higher-recall arm.

Usage:
    python3 build_token_stream.py                     # all recordings, auto
    python3 build_token_stream.py --language en
    python3 build_token_stream.py --recordings J26DS313_R0015
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda_env  # noqa: E402

cuda_env.ensure_cuda_libs()

import lid_rules  # noqa: E402

DATASET_ROOT = "/mnt/F/SLIIT/Research/SORA_Dataset"
C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(DATASET_ROOT, "processed", "audio")
GOLD_DIR = os.path.join(DATASET_ROOT, "annotations", "c1")
OUT_DIR = os.path.join(C1_ROOT, "predictions", "token_stream")
RESULTS_DIR = os.path.join(C1_ROOT, "results")


def gold_recording_ids():
    ids = set()
    for fname in os.listdir(GOLD_DIR):
        if fname.endswith(".transcript.json"):
            ids.add(fname[: -len(".transcript.json")])
    return sorted(ids)


def build_tokens_for_segment(rid, seg_index, words):
    """Turn one Whisper segment's words into extended-schema token dicts."""
    utt_id = f"{rid}_u{seg_index:03d}"
    tokens = []
    prev_lang = None
    for tok_id, word in enumerate(words):
        surface = word.word.strip()
        if not surface:
            continue
        lang, lang_conf, method = lid_rules.predict(surface)
        # switch is derived, never predicted: it is a property of the label
        # sequence. First token of an utterance is always False -- there is no
        # preceding token to switch away from.
        switch = prev_lang is not None and lang != prev_lang
        tokens.append({
            "utt_id": utt_id,
            "tok_id": tok_id,
            "token": surface,
            "lang": lang,
            "start": round(word.start, 2),
            "end": round(word.end, 2),
            "switch": switch,
            "asr_confidence": round(float(word.probability), 4),
            "lang_confidence": lang_conf,
            "lang_method": method,
        })
        prev_lang = lang
    # tok_id must be contiguous after skipping blanks
    for i, t in enumerate(tokens):
        t["tok_id"] = i
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", default="small")
    ap.add_argument("--language", default=None,
                    help="force a language code (e.g. si, en); default auto-detect")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--recordings", nargs="+", default=None)
    a = ap.parse_args()

    from faster_whisper import WhisperModel

    all_ids = a.recordings or gold_recording_ids()
    rids = [r for r in all_ids if os.path.exists(os.path.join(AUDIO_DIR, f"{r}.wav"))]
    missing = [r for r in all_ids if r not in rids]
    if missing:
        print(f"Skipping {len(missing)} recording(s) with no audio: {missing}")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Building token stream for {len(rids)} recording(s) "
          f"(model={a.model_size}, language={a.language or 'auto'})")

    model = WhisperModel(a.model_size, device="cuda", compute_type=a.compute_type)

    summary = []
    totals = {"tokens": 0, "switches": 0, "SI": 0, "EN": 0, "OTHER": 0}
    method_counts = {}

    for i, rid in enumerate(rids, 1):
        t0 = time.time()
        segments, info = model.transcribe(
            os.path.join(AUDIO_DIR, f"{rid}.wav"),
            beam_size=5,
            vad_filter=True,
            language=a.language,
            word_timestamps=True,   # required for per-word probability
        )

        all_tokens = []
        for seg_index, seg in enumerate(segments, 1):
            if not seg.words:
                continue
            all_tokens.extend(build_tokens_for_segment(rid, seg_index, seg.words))

        out_path = os.path.join(OUT_DIR, f"{rid}.tokens.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for t in all_tokens:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")

        n_switch = sum(1 for t in all_tokens if t["switch"])
        langs = {l: sum(1 for t in all_tokens if t["lang"] == l)
                 for l in ("SI", "EN", "OTHER")}
        for t in all_tokens:
            method_counts[t["lang_method"]] = method_counts.get(t["lang_method"], 0) + 1

        totals["tokens"] += len(all_tokens)
        totals["switches"] += n_switch
        for l in ("SI", "EN", "OTHER"):
            totals[l] += langs[l]

        mean_asr = (sum(t["asr_confidence"] for t in all_tokens) / len(all_tokens)
                    if all_tokens else 0.0)
        mean_lang = (sum(t["lang_confidence"] for t in all_tokens) / len(all_tokens)
                     if all_tokens else 0.0)

        summary.append({
            "recording": rid,
            "detected_language": info.language,
            "n_tokens": len(all_tokens),
            "n_switches": n_switch,
            "lang_counts": langs,
            "mean_asr_confidence": round(mean_asr, 4),
            "mean_lang_confidence": round(mean_lang, 4),
        })

        print(f"  [{i:2}/{len(rids)}] {rid}  {len(all_tokens):4} tokens  "
              f"{n_switch:3} switches  SI={langs['SI']:4} EN={langs['EN']:4} "
              f"OTHER={langs['OTHER']:3}  asrConf={mean_asr:.2f}  {time.time() - t0:5.1f}s")

    print(f"\nTotal: {totals['tokens']} tokens, {totals['switches']} switch points")
    print(f"  language mix: SI={totals['SI']} EN={totals['EN']} OTHER={totals['OTHER']}")
    print(f"  LID method mix: {method_counts}")
    print(f"  wrote {len(rids)} files to {OUT_DIR}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "c1_token_stream_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "note": "Structure is correct; content rests on an off-the-shelf ASR "
                    "baseline measured at ~0.98-1.00 WER (see c1_report.md). Do not "
                    "read these counts as corpus statistics -- gold tokens are the "
                    "source for that.",
            "model_size": a.model_size,
            "language_setting": a.language or "auto",
            "totals": totals,
            "lang_method_counts": method_counts,
            "per_recording": summary,
        }, fh, indent=2, ensure_ascii=False)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
