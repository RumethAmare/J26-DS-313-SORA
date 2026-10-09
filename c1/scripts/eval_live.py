#!/usr/bin/env python3
"""
eval_live.py -- realistic accuracy and latency of live_transcribe.py.

Streams each held-out TEST recording (whole WAV, not pre-cut clips) through
live_transcribe.py exactly as microphone audio would arrive -- Silero VAD
finds the speech and decides where utterances end -- then scores the joined
transcript against the recording's full gold transcript.

This is the number to expect in real use. It is normally higher (worse) than
the clip-based WER, because the model no longer gets the annotators'
utterance boundaries for free.

Usage:
    python eval_live.py --model medium
    python eval_live.py --model large-v3 --realtime    # also measure live latency

Writes results/c1_asr_live_<run>.json and predictions/live/eval_<run>/<rid>.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corpus  # noqa: E402
import finetune_whisper as fw  # noqa: E402
import numpy as np  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def gold_text(rid):
    path = corpus.gold_tokens_path(rid).replace(".tokens.jsonl", ".transcript.json")
    with open(path, encoding="utf-8") as fh:
        utts = sorted(json.load(fh)["utterances"], key=lambda u: (u.get("start") is None, u.get("start") or 0))
    return " ".join(u["text"].strip() for u in utts if (u.get("text") or "").strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium", help="medium | large-v3")
    ap.add_argument("--realtime", action="store_true",
                    help="stream at real speed (slow; measures true live latency)")
    ap.add_argument("--max-utterance-s", type=float, default=25.0)
    ap.add_argument("--silence-ms", type=int, default=600)
    a = ap.parse_args()
    tag = f"_max{a.max_utterance_s:g}s" if a.max_utterance_s != 25.0 else ""
    tag += f"_sil{a.silence_ms}" if a.silence_ms != 600 else ""

    import ct2_decode
    run = ct2_decode.MODELS.get(a.model, a.model)
    out_dir = os.path.join(fw.C1_ROOT, "predictions", "live", f"eval_{run}{tag}")
    os.makedirs(out_dir, exist_ok=True)
    rids = json.load(open(os.path.join(fw.DATA_DIR, "split_summary.json"),
                          encoding="utf-8"))["splits"]["test"]["recordings"]

    refs, hyps, per, lags, t_all = [], [], [], [], time.time()
    for rid in rids:
        wav = os.path.join(corpus.AUDIO_DIR, f"{rid}.wav")
        out = os.path.join(out_dir, f"{rid}.json")
        cmd = [sys.executable, os.path.join(SCRIPTS, "live_transcribe.py"), "--model", a.model,
               "--input-file", wav, "--out", out, "--no-partials",
               "--max-utterance-s", str(a.max_utterance_s), "--silence-ms", str(a.silence_ms)]
        if a.realtime:
            cmd.append("--realtime")
        t0 = time.time()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        wall = time.time() - t0
        doc = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else {"utterances": []}
        hyp = " ".join(u["text"] for u in doc["utterances"] if u["text"])
        ref = gold_text(rid)
        refs.append(ref)
        hyps.append(hyp)
        s = fw.score([ref], [hyp])
        lag = [u["latency_after_speech_s"] for u in doc["utterances"] if u["end_reason"] == "pause"]
        lags += lag
        per.append({"recording": rid, "utterances": len(doc["utterances"]),
                    "wer": s["norm_wer"], "words_recovered": s["recall_of_ref_words"],
                    "wall_s": round(wall, 1),
                    "median_latency_s": round(float(np.median(lag)), 2) if lag else None})
        print(f"{rid}: {len(doc['utterances']):3} utterances, WER {s['norm_wer']:.3f}, "
              f"{wall:.0f}s", flush=True)

    total = fw.score(refs, hyps)
    res = {"run": run + tag, "max_utterance_s": a.max_utterance_s, "silence_ms": a.silence_ms,
           "mode": "realtime" if a.realtime else "as fast as possible",
           "what": "whole test recordings streamed through live_transcribe.py; VAD finds the "
                   "utterances; joined transcript scored against the full gold transcript",
           "live_test": total, "per_recording": per,
           "latency_after_speech_s": {"median": round(float(np.median(lags)), 2) if lags else None,
                                      "p90": round(float(np.percentile(lags, 90)), 2) if lags else None,
                                      "n": len(lags)},
           "wall_s": round(time.time() - t_all, 1)}
    clip = os.path.join(fw.RESULTS_DIR, f"c1_asr_ct2_{run}.json")
    if os.path.exists(clip):
        res["clip_based_wer"] = json.load(open(clip, encoding="utf-8"))["ct2_test"]["norm_wer"]
    with open(os.path.join(fw.RESULTS_DIR, f"c1_asr_live_{run}{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f"\n{run} LIVE (whole recordings): WER {total['norm_wer']:.3f} "
          f"(clip-based {res.get('clip_based_wer')}), CER {total['norm_cer']:.3f}, "
          f"words recovered {100 * total['recall_of_ref_words']:.1f}%, "
          f"latency after pause median {res['latency_after_speech_s']['median']} s")


if __name__ == "__main__":
    main()
