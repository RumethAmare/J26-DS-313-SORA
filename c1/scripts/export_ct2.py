#!/usr/bin/env python3
"""
export_ct2.py -- turn a fine-tuned LoRA adapter into a faster-whisper model.

faster-whisper (CTranslate2) is what live transcription runs on: it is several
times faster than Hugging Face decoding and has the loop safeguards built in.
It cannot load a LoRA adapter, so the adapter is first merged into the base
Whisper weights, then the merged model is converted to CTranslate2 float16.

Usage:
    python export_ct2.py --run medium_v4_aug
    python export_ct2.py --run large-v3_v4_aug

Writes c1/models/ct2/<run>/ (git-ignored; rebuild with this script).
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_DIR = os.path.join(C1_ROOT, "models", "whisper_lora")
CT2_DIR = os.path.join(C1_ROOT, "models", "ct2")
RESULTS_DIR = os.path.join(C1_ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="adapter folder under models/whisper_lora")
    ap.add_argument("--quantization", default="float16")
    ap.add_argument("--merge-dtype", default="float16", choices=["float16", "float32"],
                    help="precision the adapter is merged in")
    ap.add_argument("--suffix", default="", help="output folder suffix (experiments)")
    a = ap.parse_args()

    from ctranslate2.converters import TransformersConverter
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    trained = json.load(open(os.path.join(RESULTS_DIR, f"c1_asr_finetune_{a.run}.json"),
                             encoding="utf-8"))
    hf_id = f"openai/whisper-{trained['model']}"
    out_dir = os.path.join(CT2_DIR, a.run + a.suffix)
    tmp = tempfile.mkdtemp(prefix=f"merge_{a.run}_")
    t0 = time.time()
    try:
        print(f"merging {a.run} into {hf_id} ...", flush=True)
        base = WhisperForConditionalGeneration.from_pretrained(
            hf_id, dtype=getattr(torch, a.merge_dtype))
        merged = PeftModel.from_pretrained(base, os.path.join(LORA_DIR, a.run)).merge_and_unload()
        merged.generation_config.forced_decoder_ids = None
        merged.save_pretrained(tmp, safe_serialization=True)
        WhisperProcessor.from_pretrained(hf_id).save_pretrained(tmp)
        del base, merged

        print(f"converting to CTranslate2 ({a.quantization}) ...", flush=True)
        copy = [f for f in ("tokenizer.json", "preprocessor_config.json")
                if os.path.exists(os.path.join(tmp, f))]
        TransformersConverter(tmp, copy_files=copy,
                              load_as_float16=a.merge_dtype == "float16").convert(
            out_dir, quantization=a.quantization, force=True)
        # faster-whisper reads the mel-band count (80, or 128 for large-v3)
        # from preprocessor_config.json; transformers 5 no longer writes it on
        # save, so take it from the original model.
        from huggingface_hub import hf_hub_download
        shutil.copy(hf_hub_download(hf_id, "preprocessor_config.json"),
                    os.path.join(out_dir, "preprocessor_config.json"))
        with open(os.path.join(out_dir, "c1_export.json"), "w", encoding="utf-8") as fh:
            json.dump({"run": a.run, "base": hf_id, "quantization": a.quantization, "merge_dtype": a.merge_dtype,
                       "adapter": os.path.relpath(os.path.join(LORA_DIR, a.run), C1_ROOT),
                       "trained_note": trained.get("note"), "best_epoch": trained.get("best_epoch"),
                       "exported": time.strftime("%Y-%m-%dT%H:%M")}, fh, indent=2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 2**30
    print(f"wrote {out_dir} ({size:.2f} GiB) in {time.time() - t0:.0f}s; files: {sorted(os.listdir(out_dir))}")

    from faster_whisper import WhisperModel
    WhisperModel(out_dir, device="cuda", compute_type="float16")
    print("faster-whisper loads it: OK")


if __name__ == "__main__":
    main()
