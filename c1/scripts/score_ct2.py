#!/usr/bin/env python3
"""
score_ct2.py -- accuracy of an exported faster-whisper model on the C1 test set.

Checks that export_ct2.py kept the fine-tuned model's accuracy, using exactly
the decoding live_transcribe.py uses (ct2_decode.py). Same 111 test clips as
every other result in the training log.

Usage:
    python score_ct2.py --model medium        # or large-v3, or a run name

Writes results/c1_asr_ct2_<run>.json and
predictions/finetune/<run>_ct2_test.json (fine-tuning details at the end).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ct2_decode  # noqa: E402  (sets up CUDA libs before faster_whisper)
import finetune_whisper as fw  # noqa: E402
import safeguard_decode as sg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="medium | large-v3 | <run name>")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--tag", default="", help="suffix for output names (experiments)")
    a = ap.parse_args()

    run, model = ct2_decode.load_model(a.model, a.compute_type)
    name = f"{run}_{a.tag}" if a.tag else run
    test = fw.load_split("test")
    audio = fw.AudioCache()

    hyps, actions, t0 = [], [], time.time()
    audio_s = 0.0
    for r in test:
        wave = audio.clip(r)
        audio_s += len(wave) / ct2_decode.SR
        text, _, _, act = ct2_decode.transcribe(model, wave)
        actions.append(act)
        hyps.append(text)
    wall = time.time() - t0

    refs = [r["text"] for r in test]
    s = fw.score(refs, hyps)
    s.update({"decode_s": round(wall, 1), "audio_s": round(audio_s, 1),
              "rtf": round(wall / audio_s, 3),
              "looping_after_decode": sum(1 for h, r in zip(hyps, test)
                                          if sg.loop_reasons(h, r["end"] - r["start"])),
              "actions": {k: actions.count(k) for k in sorted(set(actions))}})
    # the export records which training run it came from (folder may carry a suffix)
    src = json.load(open(os.path.join(ct2_decode.CT2_DIR, run, "c1_export.json"),
                         encoding="utf-8"))["run"]
    trained = json.load(open(os.path.join(fw.RESULTS_DIR, f"c1_asr_finetune_{src}.json"),
                             encoding="utf-8"))
    hf = {}
    p = os.path.join(fw.RESULTS_DIR, f"c1_asr_safeguard_{src}.json")
    if os.path.exists(p):
        hf = json.load(open(p, encoding="utf-8"))["safeguard_test"]
    out = {"run": name, "engine": f"faster-whisper (CTranslate2, compute {a.compute_type})",
           "decode_opts": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in ct2_decode.DECODE_OPTS.items()},
           "ct2_test": s, "hf_safeguard_test": hf or None}
    fw.save_predictions(f"{name}_ct2_test.json", test, hyps, fine_tuning={
        **out, "model_dir": os.path.relpath(os.path.join(ct2_decode.CT2_DIR, run), fw.C1_ROOT),
        "trained_with": {k: trained.get(k) for k in (
            "note", "lr", "lora_r", "augment", "best_epoch", "curve", "data")}})
    with open(os.path.join(fw.RESULTS_DIR, f"c1_asr_ct2_{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    cmp = f" (HF + safeguards: {hf['norm_wer']:.3f})" if hf else ""
    print(f"{run}: WER {s['norm_wer']:.3f}{cmp}, CER {s['norm_cer']:.3f}, recovered "
          f"{100 * s['recall_of_ref_words']:.1f}%, extra words {s['insertions']}, "
          f"still looping {s['looping_after_decode']}, actions {s['actions']}, "
          f"{wall:.0f}s for {audio_s / 60:.1f} min audio (RTF {s['rtf']})")


if __name__ == "__main__":
    main()
