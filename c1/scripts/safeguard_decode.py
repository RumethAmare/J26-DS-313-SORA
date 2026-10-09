#!/usr/bin/env python3
"""
safeguard_decode.py -- EXPERIMENT: re-score fine-tuned Whisper adapters with
loop safeguards that leave real repetition alone.

Greedy decoding sometimes gets stuck repeating a word or phrase ("ok, ok, ok,
..." x190). Blunt fixes like no_repeat_ngram_size would also break real speech:
32% of gold clips contain an immediate repeat (හරි හරි, ඔව් ඔව්). So a clip is
treated as LOOPING only if it is more repetitive than ANY gold transcript in
the corpus (thresholds measured on all 983 train/val/test clips):

    measure                               gold max   loop threshold
    gzip compression ratio of the text      2.73        > 3.0
    same word repeated consecutively          6         > 10
    2-6 word phrase repeated consecutively    3         > 5
    same 1-8 char unit repeated in a word     -         > 8   ("dhaidhaidhai...")
    words per second of audio               8.3         > 10 (+5 words slack)

Procedure per test clip, starting from the run's SAVED greedy predictions
(greedy decoding is deterministic, so they are exactly what the model emits):
  1. not looping                -> kept exactly as is
  2. looping                    -> re-decoded with beam search (5 beams,
                                   output length capped to the audio)
  3. beam output still looping  -> truncated where the repetition starts
                                   (two copies of the repeated unit kept)

Nothing existing is modified. Writes:
  predictions/finetune/<run>_safeguard_test.json   (fine-tuning details at end)
  results/c1_asr_safeguard_<run>.json
and verifies that every non-looping clip is unchanged and that no gold
transcript would itself be flagged.

Usage:
    python safeguard_decode.py --runs small_v4 small_v4_aug medium_v4 medium_v4_aug large-v3_v4
"""
import argparse
import json
import os
import re
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

import finetune_whisper as fw

CR_MAX = 3.0
WORD_RUN_MAX = 10
PHRASE_RUN_MAX = 5
CHAR_RUN_MAX = 8
WORDS_PER_S_MAX = 10.0
WORD_SLACK = 5
NUM_BEAMS = 5

_CHAR_LOOP = re.compile(r"(.{1,8}?)\1{%d,}" % CHAR_RUN_MAX)


def compression_ratio(text):
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))


def _norm_words(text):
    return [w.strip(".,?!;:").lower() for w in text.split()]


def phrase_run(words):
    """(k, start, copies) of the longest consecutive repetition of a k-word unit."""
    best = (1, 0, 1)
    for k in range(1, 7):
        i = 0
        while i + k <= len(words):
            c, j = 1, i + k
            while j + k <= len(words) and words[j:j + k] == words[i:i + k]:
                c += 1
                j += k
            if c > best[2] and (c > (WORD_RUN_MAX if k == 1 else PHRASE_RUN_MAX)):
                best = (k, i, c)
            i += 1
    return best


def loop_reasons(text, dur):
    r = []
    if compression_ratio(text) > CR_MAX:
        r.append("compression")
    k, _, c = phrase_run(_norm_words(text))
    if c > (WORD_RUN_MAX if k == 1 else PHRASE_RUN_MAX):
        r.append(f"{k}-word unit x{c}")
    if _CHAR_LOOP.search(text):
        r.append("char loop")
    if len(text.split()) > WORDS_PER_S_MAX * dur + WORD_SLACK:
        r.append("too long for audio")
    return r


def truncate_loop(text, dur):
    """Cut a looping text where the repetition starts, keeping two copies."""
    m = _CHAR_LOOP.search(text)
    if m:
        text = text[:m.start()] + m.group(1) * 2
    words = text.split()
    k, start, c = phrase_run(_norm_words(text))
    if c > (WORD_RUN_MAX if k == 1 else PHRASE_RUN_MAX):
        words = words[:start + 2 * k]
    cap = int(WORDS_PER_S_MAX * dur + WORD_SLACK)
    words = words[:cap]
    text = " ".join(words)
    # A loop of varied phrases can survive the cuts above: trim from the end
    # until the text is no more repetitive than real speech.
    while words and compression_ratio(" ".join(words)) > CR_MAX:
        words = words[:-1]
    return " ".join(words)


def beam_decode(model, processor, rows, audio, language):
    out = []
    for r in rows:
        feats, _ = fw.make_batch([r], processor, audio, with_labels=False)
        dur = r["end"] - r["start"]
        max_tok = min(fw.MAX_LABEL_TOKENS, int(40 * dur) + 20)
        with torch.no_grad():
            ids = model.generate(input_features=feats.to("cuda", dtype=torch.bfloat16),
                                 language=language, task="transcribe",
                                 num_beams=NUM_BEAMS, do_sample=False, max_new_tokens=max_tok)
        out.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return out


def load_preds(run):
    path = os.path.join(fw.PRED_DIR, f"{run}_ft_test.json")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return d["predictions"] if isinstance(d, dict) else d


def check_gold():
    """No gold transcript may be flagged -- otherwise the thresholds are wrong."""
    flagged = []
    for split in ("train", "val", "test"):
        for r in fw.load_split(split):
            if loop_reasons(r["text"], r["end"] - r["start"]):
                flagged.append((r["recording"], r["clip"], loop_reasons(r["text"], r["end"] - r["start"])))
    return flagged


def run_one(run, test):
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    trained = json.load(open(os.path.join(fw.RESULTS_DIR, f"c1_asr_finetune_{run}.json"),
                             encoding="utf-8"))
    model_name, language = trained["model"], trained.get("language", "en")
    preds = load_preds(run)
    key = lambda r: (r["recording"], r["clip"])
    by = {key(p): p["hypothesis"] for p in preds}
    assert len(by) == len(test) and all(key(r) in by for r in test), \
        f"{run}: saved predictions do not match the current test split"
    greedy = [by[key(r)] for r in test]
    flags = [loop_reasons(h, r["end"] - r["start"]) for h, r in zip(greedy, test)]
    idx = [i for i, f in enumerate(flags) if f]
    fw.log(f"{run}: {len(idx)}/{len(test)} clips flagged as looping")

    final, action = list(greedy), ["kept"] * len(test)
    t0 = time.time()
    if idx:
        hf_id = f"openai/whisper-{model_name}"
        processor = WhisperProcessor.from_pretrained(hf_id)
        processor.tokenizer.set_prefix_tokens(language=language, task="transcribe")
        base = WhisperForConditionalGeneration.from_pretrained(hf_id, dtype=torch.bfloat16)
        base.generation_config.forced_decoder_ids = None
        model = PeftModel.from_pretrained(base, os.path.join(fw.MODEL_DIR, run)).to("cuda").eval()
        beams = beam_decode(model, processor, [test[i] for i in idx], fw.AudioCache(), language)
        for i, b in zip(idx, beams):
            dur = test[i]["end"] - test[i]["start"]
            if not loop_reasons(b, dur):
                final[i], action[i] = b, "beam re-decode"
            else:
                final[i], action[i] = truncate_loop(b, dur), "beam + truncate"
        del model, base
        torch.cuda.empty_cache()

    unchanged_ok = all(final[i] == greedy[i] for i in range(len(test)) if not flags[i])
    refs = [r["text"] for r in test]
    before, after = fw.score(refs, greedy), fw.score(refs, final)
    after["decode_s"] = round(time.time() - t0, 1)
    still = sum(1 for h, r in zip(final, test) if loop_reasons(h, r["end"] - r["start"]))
    out = {
        "run": run, "model": model_name, "experiment": "loop safeguards (decode-time)",
        "thresholds": {"compression_ratio": CR_MAX, "word_run": WORD_RUN_MAX,
                       "phrase_run": PHRASE_RUN_MAX, "char_run": CHAR_RUN_MAX,
                       "words_per_s": WORDS_PER_S_MAX, "word_slack": WORD_SLACK,
                       "fallback_beams": NUM_BEAMS},
        "n_clips": len(test), "n_flagged": len(idx),
        "actions": {a: action.count(a) for a in sorted(set(action))},
        "non_looping_clips_unchanged": unchanged_ok,
        "still_looping_after": still,
        "greedy_test": before, "safeguard_test": after,
        "flagged_clips": [{"recording": test[i]["recording"], "clip": test[i]["clip"],
                           "reasons": flags[i], "action": action[i],
                           "words_before": len(greedy[i].split()),
                           "words_after": len(final[i].split()),
                           "reference_words": len(refs[i].split())} for i in idx],
    }
    fw.save_predictions(f"{run}_safeguard_test.json", test, final, fine_tuning={
        **{k: v for k, v in out.items() if k != "flagged_clips"},
        "adapter": os.path.relpath(os.path.join(fw.MODEL_DIR, run), fw.C1_ROOT),
        "trained_with": {k: trained.get(k) for k in (
            "note", "lr", "lora_r", "epochs_max", "patience", "batch", "seed", "augment",
            "best_epoch", "curve", "train_minutes", "data")}})
    with open(os.path.join(fw.RESULTS_DIR, f"c1_asr_safeguard_{run}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    fw.log(f"{run}: WER {before['norm_wer']:.3f} -> {after['norm_wer']:.3f}, "
           f"recovered {100 * before['recall_of_ref_words']:.1f}% -> "
           f"{100 * after['recall_of_ref_words']:.1f}%, extra words "
           f"{before['insertions']} -> {after['insertions']}; actions {out['actions']}; "
           f"non-looping unchanged={unchanged_ok}; still looping {still}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()

    bad = check_gold()
    fw.log(f"gold check: {len(bad)} of the gold transcripts would be flagged")
    if bad:
        raise SystemExit(f"thresholds flag real speech, aborting: {bad[:5]}")
    test = fw.load_split("test")
    for run in a.runs:
        run_one(run, test)


if __name__ == "__main__":
    main()
