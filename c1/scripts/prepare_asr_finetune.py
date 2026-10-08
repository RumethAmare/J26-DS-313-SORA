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
Utterances with null timestamps are dropped.

Utterances longer than Whisper's 30 s window cannot be used whole: the model
only hears the first 30 s, so the rest of the transcript would teach it to
invent words. In TRAIN recordings they are split at word boundaries using the
C1 token timestamps -- recursively at the largest pause between consecutive
words until every piece is <= MAX_CLIP_S -- and each piece keeps exactly its
own words. TRAIN clips are also capped at MAX_TEXT_TOKENS of transcript
(dense utterances split the same way, merging stops at the cap), so no
training label is ever truncated by finetune_whisper.py. In VAL/TEST recordings they are still dropped, so those sets stay
identical to the ones earlier models were scored on. An utterance is only
split if every one of its tokens has a timestamp.

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

# Token budget for a TRAIN clip's transcript. finetune_whisper.py truncates
# labels at 440 tokens (start/language/task markers + text + end marker);
# a truncated label loses its end-of-transcript marker and teaches the model
# that text does not stop when the audio does, which shows up as repetition
# loops. Sinhala costs ~35 tokens per second of speech, so a 25 s clip can
# reach ~970 tokens. 400 text tokens leaves room for the markers and for
# small differences between the small/medium/large-v3 tokenizers.
MAX_TEXT_TOKENS = 400
_TOKENIZER = None


def n_tokens(text):
    """Whisper text-token count (no special tokens)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import WhisperTokenizer
        _TOKENIZER = WhisperTokenizer.from_pretrained("openai/whisper-small")
    return len(_TOKENIZER(text, add_special_tokens=False).input_ids)


def _timed(t):
    return isinstance(t.get("start"), (int, float)) and isinstance(t.get("end"), (int, float))


def split_long(tokens):
    """Cut a token run into pieces of <= MAX_CLIP_S and <= MAX_TEXT_TOKENS.

    Returns [(start, end, text), ...]. The cut goes where the gap between one
    word's end and the next word's start is largest, so pieces break at
    natural pauses rather than mid-phrase; recurse until every piece fits.
    """
    span = tokens[-1]["end"] - tokens[0]["start"]
    text = " ".join(t["token"] for t in tokens)
    if len(tokens) < 2 or (span <= MAX_CLIP_S and n_tokens(text) <= MAX_TEXT_TOKENS):
        return [(float(tokens[0]["start"]), float(tokens[-1]["end"]), text)]
    gaps = [(tokens[i + 1]["start"] - tokens[i]["end"], i) for i in range(len(tokens) - 1)]
    _, cut = max(gaps)
    return split_long(tokens[:cut + 1]) + split_long(tokens[cut + 1:])


def load_utterances(rid, split_long_utts=False):
    """(utterances, n_dropped, n_long_split).

    With split_long_utts (train only), utterances longer than Whisper's window
    OR with more than MAX_TEXT_TOKENS of transcript are cut at word boundaries;
    ones that cannot be cut (some token untimed) are dropped rather than kept
    truncated.
    """
    tok_path = corpus.gold_tokens_path(rid)
    with open(tok_path.replace(".tokens.jsonl", ".transcript.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    by_utt = {}
    if split_long_utts:
        with open(tok_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    by_utt.setdefault(t.get("utt_id"), []).append(t)
    good, dropped, n_split = [], 0, 0
    for u in doc["utterances"]:
        s, e, text = u.get("start"), u.get("end"), (u.get("text") or "").strip()
        if (not isinstance(s, (int, float)) or not isinstance(e, (int, float))
                or e <= s or not text):
            dropped += 1
            continue
        too_dense = split_long_utts and n_tokens(text) > MAX_TEXT_TOKENS
        if e - s > WHISPER_WINDOW_S or too_dense:
            toks = sorted(by_utt.get(u["utt_id"], []), key=lambda t: t.get("tok_id", 0))
            if split_long_utts and toks and all(_timed(t) for t in toks):
                pieces = split_long(toks)
                # A piece still longer than a clip can only be one word the
                # aligner stretched over a long span (R0063's phone numbers
                # got 34 s and 42 s): its timing is wrong, so drop it.
                fit = [pc for pc in pieces if pc[1] - pc[0] <= MAX_CLIP_S]
                dropped += len(pieces) - len(fit)
                good.extend(fit)
                n_split += 1
            else:
                dropped += 1
            continue
        good.append((float(s), float(e), text))
    return sorted(good), dropped, n_split


def merge(utts, token_cap=False):
    """Merge consecutive utterances into clips <= MAX_CLIP_S (and, with
    token_cap, <= MAX_TEXT_TOKENS of transcript)."""
    clips, cur = [], None
    for s, e, text in utts:
        if (cur and s - cur["end"] <= MAX_GAP_S and e - cur["start"] <= MAX_CLIP_S
                and (not token_cap or n_tokens(cur["text"] + " " + text) <= MAX_TEXT_TOKENS)):
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
        utts, dropped, _ = load_utterances(rid)
        if any(e > dur + MAX_OVERRUN_S for _, e, _ in utts):
            skipped[rid] = "timestamps past end of audio"
            continue
        utts = [u for u in utts if u[1] <= dur + END_TOLERANCE_S]
        utts_s, dropped_s, n_split = load_utterances(rid, split_long_utts=True)
        utts_s = [u for u in utts_s if u[1] <= dur + END_TOLERANCE_S]
        recs[rid] = {
            "clips": merge(utts),                 # val/test: long utterances dropped
            "clips_split": merge(utts_s, token_cap=True),  # train: long/dense split
            "dropped_utts": dropped,
            "dropped_utts_split": dropped_s,
            "long_split": n_split,
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
    summary = {"seed": SEED, "max_clip_s": MAX_CLIP_S,
               "max_text_tokens_train": MAX_TEXT_TOKENS, "skipped": skipped, "splits": {}}
    for name in ("train", "val", "test"):
        rows = []
        clip_key = "clips_split" if name == "train" else "clips"
        for rid in sorted(r for r, s in split.items() if s == name):
            for k, c in enumerate(recs[rid][clip_key]):
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
    in_train = [r for r, s in split.items() if s == "train"]
    not_train = [r for r, s in split.items() if s != "train"]
    summary["long_utterances_split_in_train"] = sum(recs[r]["long_split"] for r in in_train)
    summary["dropped_utterances"] = (sum(recs[r]["dropped_utts_split"] for r in in_train)
                                     + sum(recs[r]["dropped_utts"] for r in not_train))
    summary["long_utterances_dropped_in_val_test"] = sum(
        recs[r]["dropped_utts"] - recs[r]["dropped_utts_split"] for r in not_train)
    with open(os.path.join(OUT_DIR, "split_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"skipped {len(skipped)} recordings: "
          f"{sorted((r[-5:], why) for r, why in skipped.items())}")
    print(f"split {summary['long_utterances_split_in_train']} long utterances in train; "
          f"kept out {summary['long_utterances_dropped_in_val_test']} long ones in val/test "
          f"so those sets stay unchanged")
    print(f"dropped {summary['dropped_utterances']} utterances (null time / empty, "
          f"plus the val/test long ones)")


if __name__ == "__main__":
    main()
