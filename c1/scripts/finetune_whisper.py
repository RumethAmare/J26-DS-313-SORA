#!/usr/bin/env python3
"""
finetune_whisper.py -- LoRA fine-tuning of Whisper on the C1 gold transcripts.

    python finetune_whisper.py --model small
    python finetune_whisper.py --model medium
    python finetune_whisper.py --model large-v3

Data comes from prepare_asr_finetune.py (clip lists split by recording).

WHAT IS MEASURED
----------------
Before training, the off-the-shelf model is scored on the test split; after
training, the fine-tuned model is scored on the same clips with the same
decoding (greedy, language token fixed). So "before" vs "after" isolates what
fine-tuning bought. Metrics:

  * norm WER  -- both sides transliterated to one phonetic skeleton
                 (script_normalize), so Sinhala-Unicode vs romanized output
                 is not charged as error. The headline.
  * raw WER   -- lower-case + punctuation strip only.
  * words recovered -- hits / reference words; WER is saturated near 1.0 on
                 this corpus off the shelf, so this shows movement WER hides.

WHY LoRA FOR ALL THREE
----------------------
Full fine-tuning of medium/large-v3 does not fit in 8 GB, and the same method
for every size keeps the comparison fair. Only the adapter (q/k/v/out and the
feed-forward projections) trains; the base weights stay frozen in bf16.

WHY THE LANGUAGE TOKEN IS `en`
------------------------------
Requested configuration. After fine-tuning, the language token is just a
fixed prompt; the model learns to emit the code-mixed transcript as written.

Outputs:
  c1/models/whisper_lora/<model>/            best adapter (by val loss)
  c1/results/c1_asr_finetune_<model>.json    metrics, curves, config
  c1/predictions/finetune/<model>_{base,ft}_test.jsonl   test transcripts
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import torch

import script_normalize as sn

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(C1_ROOT, "data", "asr_finetune")
MODEL_DIR = os.path.join(C1_ROOT, "models", "whisper_lora")
RESULTS_DIR = os.path.join(C1_ROOT, "results")
PRED_DIR = os.path.join(C1_ROOT, "predictions", "finetune")

SR = 16000
MAX_LABEL_TOKENS = 440

# per-model defaults: (micro-batch, grad-accumulation) -> effective batch 16
BATCH = {"small": (8, 2), "medium": (4, 4), "large-v3": (2, 8)}


def log(msg, fh=None):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def load_split(name):
    with open(os.path.join(DATA_DIR, f"{name}.jsonl"), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


class AudioCache:
    """Keep each recording's waveform in memory once; slice clips from it."""

    def __init__(self):
        self.cache = {}

    def clip(self, row):
        path = row["audio"]
        if path not in self.cache:
            audio, sr = sf.read(path, dtype="float32")
            assert sr == SR and audio.ndim == 1, f"{path}: expected 16 kHz mono"
            self.cache[path] = audio
        a = self.cache[path]
        return a[int(row["start"] * SR): int(row["end"] * SR)]


def make_batch(rows, processor, audio, with_labels=True):
    feats = processor.feature_extractor(
        [audio.clip(r) for r in rows], sampling_rate=SR, return_tensors="pt").input_features
    if not with_labels:
        return feats, None
    ids = [processor.tokenizer(r["text"]).input_ids[:MAX_LABEL_TOKENS] for r in rows]
    # The model prepends <|startoftranscript|> itself when shifting labels.
    sot = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    ids = [x[1:] if x and x[0] == sot else x for x in ids]
    width = max(len(x) for x in ids)
    labels = torch.full((len(ids), width), -100, dtype=torch.long)
    for i, x in enumerate(ids):
        labels[i, :len(x)] = torch.tensor(x)
    return feats, labels


def score(refs, hyps):
    import jiwer
    nr = [sn.normalize_scripted(t) for t in refs]
    nh = [sn.normalize_scripted(t) for t in hyps]
    rr = [sn.normalize_raw(t) for t in refs]
    rh = [sn.normalize_raw(t) for t in hyps]
    keep = [i for i, r in enumerate(nr) if r.strip()]
    nr, nh = [nr[i] for i in keep], [nh[i] for i in keep]
    rr, rh = [rr[i] for i in keep], [rh[i] for i in keep]
    out = jiwer.process_words(nr, nh)
    n_ref = sum(len(r.split()) for r in nr)
    n_hyp = sum(len(h.split()) for h in nh)
    return {
        "norm_wer": round(out.wer, 4),
        "raw_wer": round(jiwer.wer(rr, rh), 4),
        "norm_cer": round(jiwer.cer(nr, nh), 4),
        "words_recovered": out.hits,
        "n_ref_words": n_ref,
        "recall_of_ref_words": round(out.hits / n_ref, 4) if n_ref else 0.0,
        "output_ratio": round(n_hyp / n_ref, 3) if n_ref else 0.0,
        "substitutions": out.substitutions, "deletions": out.deletions,
        "insertions": out.insertions,
    }


@torch.no_grad()
def transcribe(model, processor, rows, audio, batch_size, language):
    model.eval()
    hyps = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        feats, _ = make_batch(chunk, processor, audio, with_labels=False)
        feats = feats.to("cuda", dtype=torch.bfloat16)
        out = model.generate(input_features=feats, language=language, task="transcribe",
                             num_beams=1, do_sample=False, max_new_tokens=MAX_LABEL_TOKENS)
        hyps.extend(processor.batch_decode(out, skip_special_tokens=True))
    return [h.strip() for h in hyps]


@torch.no_grad()
def eval_loss(model, processor, rows, audio, batch_size):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        feats, labels = make_batch(chunk, processor, audio)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_features=feats.to("cuda", dtype=torch.bfloat16),
                        labels=labels.to("cuda"))
        total += float(out.loss) * len(chunk)
        n += len(chunk)
    return total / max(n, 1)


def save_adapter(model, out_dir, attempts=10, wait_s=3.0):
    """Save the LoRA adapter, surviving a briefly locked file on Windows.

    Overwriting adapter_model.safetensors in place failed once with "Access is
    denied" (small_lr3e-4, epoch 2) -- typically antivirus or the search
    indexer holding the file it just saw written. Save into a fresh temp dir,
    then move each file into place, retrying while the target is locked.
    """
    import shutil
    tmp = out_dir + "_saving"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    for name in os.listdir(tmp):
        src, dst = os.path.join(tmp, name), os.path.join(out_dir, name)
        for k in range(attempts):
            try:
                os.replace(src, dst)
                break
            except PermissionError:
                if k == attempts - 1:
                    raise
                time.sleep(wait_s)
    shutil.rmtree(tmp, ignore_errors=True)


def save_predictions(name, rows, hyps):
    os.makedirs(PRED_DIR, exist_ok=True)
    with open(os.path.join(PRED_DIR, name), "w", encoding="utf-8") as fh:
        for r, h in zip(rows, hyps):
            fh.write(json.dumps({"recording": r["recording"], "clip": r["clip"],
                                 "start": r["start"], "end": r["end"],
                                 "reference": r["text"], "hypothesis": h},
                                ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(BATCH))
    ap.add_argument("--language", default="en")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2,
                    help="stop after this many epochs without val-loss improvement")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--tag", default="",
                    help="suffix for output names, so an experiment does not "
                         "overwrite the main run (e.g. --tag lr3e-4)")
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    micro, accum = BATCH[a.model]
    run_name = f"{a.model}_{a.tag}" if a.tag else a.model
    out_dir = os.path.join(MODEL_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logf = open(os.path.join(out_dir, "train.log"), "a", encoding="utf-8")

    train, val, test = load_split("train"), load_split("val"), load_split("test")
    log(f"model={a.model} language={a.language} train={len(train)} val={len(val)} "
        f"test={len(test)} clips; batch {micro}x{accum}", logf)

    hf_id = f"openai/whisper-{a.model}"
    processor = WhisperProcessor.from_pretrained(hf_id)
    processor.tokenizer.set_prefix_tokens(language=a.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(hf_id, dtype=torch.bfloat16)
    model.generation_config.forced_decoder_ids = None
    model.to("cuda")
    audio = AudioCache()

    result = {"model": a.model, "run": run_name, "epochs_max": a.epochs,
              "patience": a.patience, "hf_id": hf_id, "language": a.language,
              "method": "LoRA", "lora_r": a.lora_r, "lr": a.lr,
              "batch": {"micro": micro, "accum": accum}, "seed": a.seed,
              "data": json.load(open(os.path.join(DATA_DIR, "split_summary.json"),
                                     encoding="utf-8"))["splits"]}
    for k in result["data"]:
        result["data"][k].pop("recordings", None)

    # --- baseline: off-the-shelf, same clips, same decoding -----------------
    if not a.skip_baseline:
        t0 = time.time()
        hyps = transcribe(model, processor, test, audio, micro * 2, a.language)
        result["baseline_test"] = score([r["text"] for r in test], hyps)
        result["baseline_test"]["decode_s"] = round(time.time() - t0, 1)
        save_predictions(f"{run_name}_base_test.jsonl", test, hyps)
        log(f"BASELINE test: {result['baseline_test']}", logf)

    # --- LoRA ---------------------------------------------------------------
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
    model = get_peft_model(model, lcfg)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()          # adapters train in fp32
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"trainable params {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", logf)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / (micro * accum))
    total_steps = steps_per_epoch * a.epochs
    warmup = max(1, int(0.1 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    curve, best, bad_epochs = [], float("inf"), 0
    val0 = eval_loss(model, processor, val, audio, micro * 2)
    log(f"epoch 0 val_loss {val0:.4f}", logf)
    curve.append({"epoch": 0, "val_loss": round(val0, 4)})
    t_train = time.time()
    step = 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        running, n_micro = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i in range(0, len(order), micro):
            chunk = [train[j] for j in order[i:i + micro]]
            feats, labels = make_batch(chunk, processor, audio)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_features=feats.to("cuda", dtype=torch.bfloat16),
                            labels=labels.to("cuda"))
            (out.loss / accum).backward()
            running += float(out.loss)
            n_micro += 1
            if n_micro % accum == 0 or i + micro >= len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        train_loss = running / max(n_micro, 1)
        vl = eval_loss(model, processor, val, audio, micro * 2)
        curve.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                      "val_loss": round(vl, 4),
                      "elapsed_min": round((time.time() - t_train) / 60, 1)})
        log(f"epoch {epoch} train_loss {train_loss:.4f} val_loss {vl:.4f} "
            f"({(time.time() - t_train) / 60:.1f} min, peak GPU "
            f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", logf)
        if vl < best - 1e-4:
            best, bad_epochs = vl, 0
            save_adapter(model, out_dir)
            result["best_epoch"] = epoch
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                log(f"early stop: no val improvement for {a.patience} epochs", logf)
                break
        json.dump({**result, "curve": curve}, open(
            os.path.join(RESULTS_DIR, f"c1_asr_finetune_{run_name}.json"), "w",
            encoding="utf-8"), indent=2, ensure_ascii=False)

    result["curve"] = curve
    result["train_minutes"] = round((time.time() - t_train) / 60, 1)
    result["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    # --- reload best adapter, score test -----------------------------------
    from peft import PeftModel
    base = WhisperForConditionalGeneration.from_pretrained(hf_id, dtype=torch.bfloat16)
    base.generation_config.forced_decoder_ids = None
    model = PeftModel.from_pretrained(base, out_dir).to("cuda")
    t0 = time.time()
    hyps = transcribe(model, processor, test, audio, micro * 2, a.language)
    result["finetuned_test"] = score([r["text"] for r in test], hyps)
    result["finetuned_test"]["decode_s"] = round(time.time() - t0, 1)
    save_predictions(f"{run_name}_ft_test.jsonl", test, hyps)
    log(f"FINE-TUNED test: {result['finetuned_test']}", logf)

    with open(os.path.join(RESULTS_DIR, f"c1_asr_finetune_{run_name}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    log(f"wrote results/c1_asr_finetune_{run_name}.json", logf)


if __name__ == "__main__":
    main()
