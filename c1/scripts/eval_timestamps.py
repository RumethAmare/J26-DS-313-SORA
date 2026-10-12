#!/usr/bin/env python3
"""
eval_timestamps.py -- how accurate are the word timestamps C1 hands to C3?

On the 111 held-out test clips, compares each method's word times with the
gold token times. NOTE: the gold times are not hand-placed -- the dataset
workflow (SORA_Dataset/docs/WORKFLOW_per_recording_updated.md, step 5) made
them with WhisperX's English wav2vec2 forced aligner on the corrected text.
So "error" here is disagreement with another aligner, not true error; the
gold_ctc row shows how far two aligners disagree on identical text.

The number that matters for C3 is "same speaker turn": the gold speaker turns
(annotations/c3/<rid>.rttm) are looked up at the predicted word's midpoint
and at the gold word's midpoint -- would C3 give the word the same speaker?

Methods:
  whisper   faster-whisper cross-attention times (word_timestamps=True)
  ctc       MMS forced alignment of the same fine-tuned transcript (word_align.py)
  gold_ctc  MMS forced alignment of the GOLD text -- the aligner's own ceiling,
            separating timing error from recognition error

Only words the model got right can be compared: hypothesis and gold words
are matched on their script-normalised form (the same normalisation as the
WER), in order. Errors are in milliseconds.

Usage:
    python eval_timestamps.py --model medium large-v3

Writes results/c1_timestamps_<run>.json and results/c1_timestamps_report.md.
"""
import argparse
import copy
import glob
import json
import os
import sys
import tempfile
import time
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ct2_decode  # noqa: E402  (sets up CUDA libs before faster_whisper)
import corpus  # noqa: E402
import finetune_whisper as fw  # noqa: E402
import numpy as np  # noqa: E402
import script_normalize as sn  # noqa: E402

# ctc+interp@T: words aligned with score < T re-timed between confident
# neighbours; ctc+whisper@T: ... re-timed with Whisper's own time (word_align.repair)
THRESHOLDS = (0.05, 0.1, 0.2)
METHODS = ("whisper", "ctc", *[f"ctc+interp@{t}" for t in THRESHOLDS],
           *[f"ctc+whisper@{t}" for t in THRESHOLDS], "gold_ctc", "gold_ctc+interp@0.1")
# read by the status bar (~/.claude/c1_status.py)
PROGRESS = os.path.join(tempfile.gettempdir(), "c1_eval_timestamps_progress.json")
EDGE = 0.05          # gold tokens may sit this far outside the clip bounds
RTTM_DIR = os.path.join(os.path.dirname(corpus.GOLD_DIR), "c3")


def turns(rid, cache={}):
    """Gold speaker turns [(start, end, speaker)] from the C3 RTTM."""
    if rid not in cache:
        out = []
        path = os.path.join(RTTM_DIR, f"{corpus.canonical_rid(rid)}.rttm")
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                f = line.split()
                if len(f) > 7 and f[0] == "SPEAKER":
                    out.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
        cache[rid] = out
    return cache[rid]


def speaker_at(trs, t):
    """Speaker of the turn containing t, else of the nearest turn."""
    if not trs:
        return None
    best = min(trs, key=lambda x: 0 if x[0] <= t <= x[1] else min(abs(t - x[0]), abs(t - x[1])))
    return best[2]


def word_error(h, g, trs, offset=0.0):
    """(|start err|, |end err|, midpoint inside gold word, same speaker turn);
    h/g times relative to `offset` within the recording."""
    hm, gm = (h["start"] + h["end"]) / 2, (g["start"] + g["end"]) / 2
    same = None if not trs else speaker_at(trs, hm + offset) == speaker_at(trs, gm + offset)
    return (abs(h["start"] - g["start"]), abs(h["end"] - g["end"]), g["start"] <= hm <= g["end"], same)


def gold_tokens(rid, cache={}):
    if rid not in cache:
        with open(corpus.gold_tokens_path(rid), encoding="utf-8") as fh:
            cache[rid] = [json.loads(l) for l in fh if l.strip()]
    return cache[rid]


def clip_gold(row):
    """Gold tokens inside the clip, with times relative to the clip start."""
    out = []
    for t in gold_tokens(row["recording"]):
        if t.get("start") is None or t.get("end") is None:
            continue
        if t["start"] >= row["start"] - EDGE and t["end"] <= row["end"] + EDGE:
            out.append({"word": t["token"], "start": t["start"] - row["start"],
                        "end": t["end"] - row["start"], "utt_id": t["utt_id"]})
    return out


def pairs(hyp, gold):
    """(hyp word, gold word) for words recognised correctly, in order."""
    key = lambda w: sn.normalize_scripted(w["word"]) or "\0"
    a, b = [key(w) for w in hyp], [key(w) for w in gold]
    out = []
    for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        out += [(hyp[blk.a + k], gold[blk.b + k]) for k in range(blk.size)]
    return out


def summarise(errs, n_gold):
    if not errs:
        return {"matched_words": 0}
    s = np.array([e[0] for e in errs]) * 1000
    e = np.array([e[1] for e in errs]) * 1000
    mid_in = np.array([e[2] for e in errs])
    both = np.maximum(s, e)
    same = [x[3] for x in errs if x[3] is not None]
    return {
        "matched_words": len(errs), "gold_words": n_gold,
        "start_err_ms": {"median": round(float(np.median(s))), "mean": round(float(s.mean())),
                         "p90": round(float(np.percentile(s, 90)))},
        "end_err_ms": {"median": round(float(np.median(e))), "mean": round(float(e.mean())),
                       "p90": round(float(np.percentile(e, 90)))},
        "within_ms": {str(t): round(float((both <= t).mean()), 4) for t in (50, 100, 200, 500)},
        "midpoint_inside_gold_word": round(float(mid_in.mean()), 4),
        "same_speaker_turn_as_gold": round(float(np.mean(same)), 4) if same else None,
        "speaker_turn_words": len(same),
    }


def evaluate(model_name):
    run, model = ct2_decode.load_model(model_name)
    test = fw.load_split("test")
    audio = fw.AudioCache()
    errs = {m: [] for m in METHODS}
    n_gold, align_s, t0 = 0, [], time.time()
    for i, row in enumerate(test):
        wave = audio.clip(row)
        gold = clip_gold(row)
        n_gold += len(gold)
        text, words, _, _ = ct2_decode.transcribe(model, wave, timing="whisper")
        by = {"whisper": words}
        ta = time.time()
        by["ctc"] = ct2_decode.align_words(wave, copy.deepcopy(words))
        align_s.append(time.time() - ta)
        for t in THRESHOLDS:
            by[f"ctc+interp@{t}"] = ct2_decode.align_words(wave, copy.deepcopy(words), min_score=t)
            by[f"ctc+whisper@{t}"] = ct2_decode.align_words(wave, copy.deepcopy(words), min_score=t,
                                                           use_whisper=True)
        by["gold_ctc"] = ct2_decode.align_words(wave, [{"word": g["word"]} for g in gold])
        by["gold_ctc+interp@0.1"] = ct2_decode.align_words(wave, [{"word": g["word"]} for g in gold],
                                                            min_score=0.1)
        for m in METHODS:
            for h, g in pairs(by[m], gold):
                if h.get("start") is None:
                    continue
                errs[m].append(word_error(h, g, turns(row["recording"]), row["start"]))
        print(f"\r{run}: clip {i + 1}/{len(test)}", end="", flush=True)
        with open(PROGRESS, "w", encoding="utf-8") as fh:
            json.dump({"run": run, "done": i + 1, "total": len(test), "started": t0,
                       "updated": time.time()}, fh)
    print()
    res = {"run": run, "test_clips": len(test),
           "what": "word start/end vs hand-checked gold token times, words recognised "
                   "correctly only (script-normalised match)",
           "methods": {m: summarise(errs[m], n_gold) for m in METHODS},
           "ctc_align_s_per_clip": {"median": round(float(np.median(align_s)), 3),
                                    "p90": round(float(np.percentile(align_s, 90)), 3)},
           "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(fw.RESULTS_DIR, f"c1_timestamps_{run}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    for m in METHODS:
        r = res["methods"][m]
        print(f"  {m:20} matched {r['matched_words']:5}  start median {r['start_err_ms']['median']:4} ms  "
              f"end median {r['end_err_ms']['median']:4} ms  within 100 ms {100 * r['within_ms']['100']:.0f}%  "
              f"midpoint in gold word {100 * r['midpoint_inside_gold_word']:.0f}%  "
              f"same speaker turn {100 * (r['same_speaker_turn_as_gold'] or 0):.1f}%")
    del model
    return res


def write_report():
    rows = []
    for p in sorted(glob.glob(os.path.join(fw.RESULTS_DIR, "c1_timestamps_*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        for m in METHODS:
            x = r["methods"].get(m) or {}
            if not x.get("matched_words"):
                continue
            rows.append(f"| {r['run']} | {m} | {x['matched_words']} | {x['start_err_ms']['median']} | "
                        f"{x['end_err_ms']['median']} | {x['start_err_ms']['p90']} / {x['end_err_ms']['p90']} | "
                        f"{100 * x['within_ms']['50']:.0f}% | {100 * x['within_ms']['100']:.0f}% | "
                        f"{100 * x['within_ms']['200']:.0f}% | {100 * x['midpoint_inside_gold_word']:.0f}% | "
                        f"{100 * (x.get('same_speaker_turn_as_gold') or 0):.1f}% |")
    md = f"""# C1 word timestamps — accuracy for C3

Generated by `scripts/eval_timestamps.py`. Each token's `start`/`end` is compared with the
gold token times on the 111 held-out test clips. Only words the model recognised
correctly are compared, matched in order on script-normalised form.

**The gold times are not hand-placed.** The dataset workflow (step 5) made them with
WhisperX's *English* wav2vec2 forced aligner. The millisecond columns therefore
measure disagreement with that aligner, not true error, and the **gold_ctc** row shows
how much two aligners disagree on identical text. The column that matters for C3 is
**same speaker turn**: the gold RTTM turn (`annotations/c3/<rid>.rttm`) is looked up at
the predicted word's midpoint and at the gold word's midpoint, to check that C3 would
give the word the same speaker.

- **whisper**: faster-whisper's cross-attention estimate (`word_timestamps=True`).
- **ctc**: MMS forced alignment (`word_align.py`) of the same fine-tuned transcript.
- **gold_ctc**: the same aligner on the *gold* text. This is the aligner's own ceiling.

"Within X ms" means both the start and the end are within X ms of gold. "Midpoint in
gold word" is the share of words whose centre falls inside the gold word's span.

| model | method | words compared | start median ms | end median ms | p90 start / end ms | within 50 ms | within 100 ms | within 200 ms | midpoint in gold word | same speaker turn |
|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + """

## Findings (2026-10-12)

- **CTC forced alignment is the default** (`live_transcribe.py --timing ctc`). On large-v3
  it cuts the median start error from 270 ms (Whisper cross-attention) to 60 ms. It agrees
  with Whisper on speaker turns (95.4% vs 95.5%) and runs *after* the text is shown,
  ~0.05–0.1 s per utterance.
- Whisper's cross-attention times get *worse* as the model gets larger: small 180 ms,
  medium 90 ms, large-v3 270 ms. Whisper keeps a near-constant offset with few large
  misses, so it does well on speaker turns despite the poor millisecond scores.
- **About 5% of words land in a different speaker turn from the gold**, even with the gold
  text (gold_ctc 93.8%). Half of these are > 1 s off and have a very low
  `align_score` (median 0.05 vs 0.35 overall), so the score flags them. C3 should treat
  tokens with `align_score` < ~0.05 as weak evidence at a turn boundary.
- **Re-timing low-score words (`+interp@T`, `+whisper@T`) made every metric worse**, so
  it is off by default (`ct2_decode.align_words(min_score=None)`). The low-score words
  are not simply misplaced between good neighbours.
- **The gold times are WhisperX-aligned, not hand-placed, and they are wrong around
  numbers.** WhisperX's English aligner has no characters for digits, so tokens like
  `199456712034.` could not be aligned, and the snapped times nearby got compressed.
  - Example, R0018: gold has "September 21st 1994." at 38.78–39.30 s, three words in
    0.5 s. CTC on the *gold* text puts them at 40.69–41.61 s, matching the live output
    (40.71–41.61) with scores of 0.86–1.0.
  - C3's RTTM turns were derived from the same times, so they inherit the error.
  - Over whole recordings, confident words (`align_score` ≥ 0.3) more than 1 s from gold
    sit within 8 tokens of a gold number 58% of the time, against 37% for all words.
  - **Some of the "error" in this report is therefore gold error.** This is worth raising
    with the dataset team: re-align the gold with a digit-aware aligner, or spell numbers
    out before WhisperX.

## Whole recordings (what C3 receives)

`eval_live.py --model large-v3 --tag ctc` streams the 10 test recordings through the
live pipeline. It covers VAD cuts, 25 s force-cuts and session-absolute times, and gives
`results/c1_asr_live_large-v3_v4_aug_ctc.json`:

- WER 0.374 and a median latency of 1.54 s after a pause, unchanged by the alignment step.
- **Start error: median 100 ms**, 40% within 100 ms, 65% within 500 ms.
- **Same speaker turn as gold: 92.7% of words, or 96.2% for words with
  `align_score` ≥ 0.2** (75% of words).
- The tail is heavier than on clips: 21% of words are > 1 s from gold. That is 23% in
  utterances ended by a 25 s force-cut and 19% in those ended by a pause. There are two
  causes:
  - **Gold errors near numbers**, as above.
  - **Text that doesn't match the audio:** words Whisper misheard or dropped, and the
    speech right after a 25 s force-cut. The aligner has to put those words somewhere,
    and gives them low scores.
- R0073 and R0075 have no RTTM, so the speaker-turn column covers the other 8 test
  recordings.
"""
    with open(os.path.join(fw.RESULTS_DIR, "c1_timestamps_report.md"), "w", encoding="utf-8") as fh:
        fh.write(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["medium", "large-v3"])
    a = ap.parse_args()
    for m in a.model:
        evaluate(m)
    write_report()


if __name__ == "__main__":
    main()
