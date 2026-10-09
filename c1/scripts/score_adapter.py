#!/usr/bin/env python3
"""
score_adapter.py -- re-score an already trained LoRA adapter on the CURRENT
test split, with exactly the decoding and scoring finetune_whisper.py uses.

Needed whenever the test split changes (e.g. gold timestamps get fixed and
test clips are rebuilt): numbers from different test splits are not
comparable, and retraining just to re-score would waste hours.

Usage:
    python score_adapter.py --run small              # adapter in models/whisper_lora/small
    python score_adapter.py --run small --baseline   # also score the off-the-shelf model

Writes results/c1_asr_rescore_<run>.json and predictions/finetune/<run>_rescore_test.json.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import finetune_whisper as fw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="adapter folder name under models/whisper_lora")
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()

    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    adapter_dir = os.path.join(fw.MODEL_DIR, a.run)
    trained = json.load(open(os.path.join(fw.RESULTS_DIR, f"c1_asr_finetune_{a.run}.json"),
                             encoding="utf-8"))
    model_name, language = trained["model"], trained.get("language", "en")
    micro, _ = fw.BATCH[model_name]
    hf_id = f"openai/whisper-{model_name}"

    test = fw.load_split("test")
    summary = json.load(open(os.path.join(fw.DATA_DIR, "split_summary.json"), encoding="utf-8"))
    processor = WhisperProcessor.from_pretrained(hf_id)
    processor.tokenizer.set_prefix_tokens(language=language, task="transcribe")
    audio = fw.AudioCache()
    refs = [r["text"] for r in test]
    out = {"run": a.run, "model": model_name, "language": language,
           "test_split": {k: v for k, v in summary["splits"]["test"].items() if k != "recordings"},
           "trained_on": trained.get("data", {}).get("train")}

    base = WhisperForConditionalGeneration.from_pretrained(hf_id, dtype=torch.bfloat16)
    base.generation_config.forced_decoder_ids = None
    if a.baseline:
        base.to("cuda")
        t0 = time.time()
        hyps = fw.transcribe(base, processor, test, audio, micro * 2, language)
        out["baseline_test"] = {**fw.score(refs, hyps), "decode_s": round(time.time() - t0, 1)}
        fw.log(f"{a.run} BASELINE on current test: {out['baseline_test']}")

    model = PeftModel.from_pretrained(base, adapter_dir).to("cuda")
    t0 = time.time()
    hyps = fw.transcribe(model, processor, test, audio, micro * 2, language)
    out["finetuned_test"] = {**fw.score(refs, hyps), "decode_s": round(time.time() - t0, 1)}
    fw.save_predictions(f"{a.run}_rescore_test.json", test, hyps,
                        fine_tuning={**out, "adapter": os.path.relpath(adapter_dir, fw.C1_ROOT),
                                     "trained_with": {k: trained.get(k) for k in (
                                         "lr", "lora_r", "epochs_max", "patience", "batch",
                                         "seed", "best_epoch", "curve", "train_minutes")}})
    fw.log(f"{a.run} FINE-TUNED on current test: {out['finetuned_test']}")

    path = os.path.join(fw.RESULTS_DIR, f"c1_asr_rescore_{a.run}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    fw.log(f"wrote {os.path.relpath(path, fw.C1_ROOT)}")


if __name__ == "__main__":
    main()
