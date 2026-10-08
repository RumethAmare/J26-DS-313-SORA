#!/usr/bin/env python3
"""
prepare_asr_finetune.py -- build train/val/test clip lists for Whisper fine-tuning.

Each clip is a span of one recording's audio plus its gold transcript text.
Audio is not copied: the trainer slices clips out of the WAVs on the fly.

Which recordings:
  * every recording with gold C1 and audio. corpus.EXCLUDED_RECORDINGS is NOT
    applied: it lists recordings whose SI/EN token *labels* are wrong, and ASR
    only uses the utterance text, which those recordings have right
  * minus any whose gold timestamps run more than MAX_OVERRUN_S past the end
    of their audio (checked against the WAV each run), since their text
    would not line up with the clip it is paired with; smaller overruns
    just drop the stray utterances. Re-time such recordings with
    SORA_Dataset/tools/c1/realign_mms.py rather than dropping them

Clips: consecutive utterances are merged while the span stays under
MAX_CLIP_S, giving the model context and fewer, fuller training examples.
Utterances with null timestamps or longer than Whisper's 30s window are
dropped.

Split: by RECORDING, never by clip, so no conversation is in both training
and test. Stratified by Sinhala script convention (Unicode vs romanized) so
the test set exercises both, with a fixed seed.

The split is sticky: if split_summary.json already exists, every recording
it placed keeps its split and only newly eligible recordings are added, all
to train. Val/test therefore stay fixed and WER stays comparable across
reruns as data is added. Delete split_summary.json to re-draw from scratch.

Text is kept exactly as annotated (Sinhala Unicode or romanized). Scoring
uses script_normalize so the script a model chooses is not penalised.

Outputs c1/data/asr_finetune/{train,val,test}.jsonl and split_summary.json.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audit_gold
import corpus

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(C1_ROOT, "data", "asr_finetune")

MAX_CLIP_S = 25.0
END_TOLERANCE_S = 0.25   # an utterance ending later than this past the WAV is dropped
MAX_OVERRUN_S = 5.0      # gold ending later than this past the WAV: timeline is
                         # wrong, so skip the whole recording (R0019 and R0053
                         # overrun by 0.9 s / 1.9 s on their last utterance only)
MAX_GAP_S = 2.0          # do not merge across a long silence
WHISPER_WINDOW_S = 30.0
TEST_FRAC, VAL_FRAC = 0.15, 0.10
SEED = 20261008


def load_utterances(rid):
    path = corpus.gold_tokens_path(rid).replace(".tokens.jsonl", ".transcript.json")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    good, dropped = [], 0
    for u in doc["utterances"]:
        s, e, text = u.get("start"), u.get("end"), (u.get("text") or "").strip()
        if (not isinstance(s, (int, float)) or not isinstance(e, (int, float))
                or e <= s or e - s > WHISPER_WINDOW_S or not text):
            dropped += 1
            continue
        good.append((float(s), float(e), text))
    return sorted(good), dropped


def merge(utts):
    clips, cur = [], None
    for s, e, text in utts:
        if cur and s - cur["end"] <= MAX_GAP_S and e - cur["start"] <= MAX_CLIP_S:
            cur["end"] = e
            cur["text"] += " " + text
        else:
            if cur:
                clips.append(cur)
            cur = {"start": s, "end": e, "text": text}
    if cur:
        clips.append(cur)
    return clips


def main():
    import soundfile as sf

    manifest_ids = audit_gold.read_manifest_ids()
    recs, skipped = {}, {}
    for rid, path in corpus.gold_token_files():
        wav = os.path.join(corpus.AUDIO_DIR, f"{rid}.wav")
        if not os.path.exists(wav):
            skipped[rid] = "no audio"
            continue
        dur = sf.info(wav).duration
        utts, dropped = load_utterances(rid)
        if any(e > dur + MAX_OVERRUN_S for _, e, _ in utts):
            skipped[rid] = "timestamps past end of audio"
            continue
        utts = [u for u in utts if u[1] <= dur + END_TOLERANCE_S]
        recs[rid] = {
            "clips": merge(utts),
            "dropped_utts": dropped,
            "script": audit_gold.audit_file(rid, path, manifest_ids)[0]
            ["sinhala_script_convention"],
        }

    # Keep the previous assignment for recordings already placed.
    summary_path = os.path.join(OUT_DIR, "split_summary.json")
    split = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            for name, info in json.load(fh)["splits"].items():
                split.update({rid: name for rid in info["recordings"] if rid in recs})
        new = sorted(r for r in recs if r not in split)
        split.update({rid: "train" for rid in new})
        print(f"kept previous split for {len(split) - len(new)} recordings; "
              f"{len(new)} new -> train: {[r[-5:] for r in new]}")
    else:
        # Stratified split by recording.
        rng = random.Random(SEED)
        for conv in sorted({r["script"] for r in recs.values()}):
            group = sorted(r for r, v in recs.items() if v["script"] == conv)
            rng.shuffle(group)
            n_test = max(1, round(TEST_FRAC * len(group))) if len(group) > 2 else 0
            n_val = max(1, round(VAL_FRAC * len(group))) if len(group) > 3 else 0
            for i, rid in enumerate(group):
                split[rid] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {"seed": SEED, "max_clip_s": MAX_CLIP_S, "skipped": skipped, "splits": {}}
    for name in ("train", "val", "test"):
        rows = []
        for rid in sorted(r for r, s in split.items() if s == name):
            for k, c in enumerate(recs[rid]["clips"]):
                rows.append({"recording": rid, "clip": k,
                             "audio": os.path.join(corpus.AUDIO_DIR, f"{rid}.wav"),
                             "start": round(c["start"], 3), "end": round(c["end"], 3),
                             "text": c["text"]})
        with open(os.path.join(OUT_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        secs = sum(r["end"] - r["start"] for r in rows)
        rids = sorted({r["recording"] for r in rows})
        summary["splits"][name] = {
            "recordings": rids, "n_recordings": len(rids), "n_clips": len(rows),
            "hours": round(secs / 3600, 3),
            "scripts": {c: sum(1 for r in rids if recs[r]["script"] == c)
                        for c in sorted({recs[r]["script"] for r in rids})},
        }
        print(f"{name:5}: {len(rids):2} recordings, {len(rows):4} clips, "
              f"{secs / 3600:.2f} h speech  {summary['splits'][name]['scripts']}")
    summary["dropped_utterances"] = sum(v["dropped_utts"] for v in recs.values())
    with open(os.path.join(OUT_DIR, "split_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"skipped {len(skipped)} recordings: "
          f"{sorted((r[-5:], why) for r, why in skipped.items())}")
    print(f"dropped {summary['dropped_utterances']} utterances (null time / >30s / empty)")


if __name__ == "__main__":
    main()
