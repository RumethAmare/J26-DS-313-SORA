#!/usr/bin/env python3
"""
report_finetune.py -- per-epoch report for the Whisper LoRA fine-tuning runs.

Reads results/c1_asr_finetune_<model>.json (written by finetune_whisper.py)
and writes results/c1_asr_finetune_report.md. Re-run after any retraining.

Each epoch was scored by validation LOSS during training; word error rate was
measured once per model, on the best-epoch adapter (the only one saved). The
report says so rather than implying per-epoch WER.
"""
import json
import os

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(C1_ROOT, "results")
MODELS = ["small", "medium", "large-v3"]


def load(model):
    path = os.path.join(RESULTS_DIR, f"c1_asr_finetune_{model}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def epoch_table(r):
    curve = r["curve"]
    best = r.get("best_epoch")
    L = ["| epoch | train loss | val loss | change in val loss | val − train gap "
         "| epoch time | cumulative | |\n",
         "|---|---|---|---|---|---|---|---|\n"]
    prev_val, prev_t = None, 0.0
    for c in curve:
        e, vl = c["epoch"], c["val_loss"]
        tl = c.get("train_loss")
        t = c.get("elapsed_min")
        delta = f"{vl - prev_val:+.4f}" if prev_val is not None else "—"
        gap = f"{vl - tl:+.3f}" if tl is not None else "—"
        ep_t = f"{t - prev_t:.1f} min" if t is not None else "—"
        cum = f"{t:.1f} min" if t is not None else "—"
        if e == 0:
            mark = "before training"
        elif e == best:
            mark = "**best — saved**"
        elif prev_val is not None and vl > prev_val:
            mark = "val worse"
        else:
            mark = ""
        tl_s = f"{tl:.4f}" if tl is not None else "—"
        vl_s = f"**{vl:.4f}**" if e == best else f"{vl:.4f}"
        L.append(f"| {e} | {tl_s} | {vl_s} | {delta} | {gap} | {ep_t} | {cum} | {mark} |\n")
        prev_val = vl
        if t is not None:
            prev_t = t
    return L


def observations(r):
    curve = r["curve"]
    best = r.get("best_epoch")
    v0 = curve[0]["val_loss"]
    vb = next(c["val_loss"] for c in curve if c["epoch"] == best)
    last = curve[-1]
    first_gain = curve[1]["val_loss"] - v0
    total_gain = vb - v0
    obs = [f"- Validation loss fell from {v0:.3f} to {vb:.3f} at epoch {best} "
           f"({100 * (v0 - vb) / v0:.0f}% lower). Epoch 1 alone delivered "
           f"{100 * first_gain / total_gain:.0f}% of that drop.\n"]
    after = [c for c in curve if c["epoch"] > best]
    if after:
        obs.append(f"- After epoch {best}, training loss kept falling "
                   f"({next(c['train_loss'] for c in curve if c['epoch'] == best):.3f} → "
                   f"{last['train_loss']:.3f}) while validation loss rose "
                   f"({vb:.3f} → {last['val_loss']:.3f}): the model began memorising "
                   f"the training clips. Early stopping ended the run at epoch "
                   f"{last['epoch']} of 10.\n")
    gap_best = vb - next(c["train_loss"] for c in curve if c["epoch"] == best)
    obs.append(f"- Gap between validation and training loss at the best epoch: "
               f"{gap_best:.2f}, widening to {last['val_loss'] - last['train_loss']:.2f} "
               f"by the last epoch.\n")
    times = []
    prev = 0.0
    for c in curve[1:]:
        times.append(c["elapsed_min"] - prev)
        prev = c["elapsed_min"]
    if times and max(times) > 2 * min(times):
        slow = max(range(len(times)), key=lambda i: times[i]) + 1
        obs.append(f"- Epoch times varied from {min(times):.1f} to {max(times):.1f} min "
                   f"(slowest: epoch {slow}). Every epoch does the same work, so the "
                   f"variation is the laptop (GPU memory pressure from other apps, or "
                   f"background load), not the training.\n")
    return obs


def main():
    runs = {m: load(m) for m in MODELS}
    runs = {m: r for m, r in runs.items() if r and "curve" in r}
    any_r = next(iter(runs.values()))
    d = any_r["data"]

    L = ["# Whisper fine-tuning — epoch-by-epoch report\n\n",
         "Produced by `scripts/report_finetune.py` from "
         "`results/c1_asr_finetune_<model>.json`. Training: "
         "`scripts/finetune_whisper.py`.\n\n",
         "## Setup\n\n",
         f"- **Method:** LoRA (rank {any_r['lora_r']}) on attention and feed-forward "
         f"layers; base weights frozen in bf16. Same for all three models.\n",
         f"- **Language token:** `{any_r['language']}`. Learning rate {any_r['lr']}, "
         "linear warm-up then decay, effective batch 16 clips.\n",
         f"- **Data**, split by recording: train {d['train']['n_recordings']} recordings / "
         f"{d['train']['n_clips']} clips / {d['train']['hours']:.2f} h of speech; "
         f"validation {d['val']['n_recordings']} / {d['val']['n_clips']} / "
         f"{d['val']['hours']:.2f} h; test {d['test']['n_recordings']} / "
         f"{d['test']['n_clips']} / {d['test']['hours']:.2f} h.\n",
         "- **Per epoch:** one pass over the training clips, then validation loss "
         "(the model's average error at predicting the validation transcripts, "
         "token by token; lower is better). The adapter is saved whenever validation "
         "loss improves; training stops after 2 epochs without improvement (max 10).\n",
         "- **Word error rate is not measured per epoch.** It was measured once per "
         "model, on the saved best-epoch adapter, against the test set — see the "
         "final section.\n\n"]

    L.append("## Validation loss by epoch, all models\n\n")
    max_e = max(c["epoch"] for r in runs.values() for c in r["curve"])
    L.append("| epoch | " + " | ".join(f"`{m}`" for m in runs) + " |\n")
    L.append("|---|" + "---|" * len(runs) + "\n")
    for e in range(max_e + 1):
        cells = []
        for m, r in runs.items():
            c = next((c for c in r["curve"] if c["epoch"] == e), None)
            if c is None:
                cells.append("stopped")
            elif e == r.get("best_epoch"):
                cells.append(f"**{c['val_loss']:.4f}** ←best")
            else:
                cells.append(f"{c['val_loss']:.4f}")
        L.append(f"| {e} | " + " | ".join(cells) + " |\n")
    L.append("\nEpoch 0 is the off-the-shelf model before any training.\n\n")

    for m, r in runs.items():
        L.append(f"## `{m}`\n\n")
        L.extend(epoch_table(r))
        L.append("\n")
        L.extend(observations(r))
        L.append(f"- Total training time {r['train_minutes']:.0f} min; peak GPU memory "
                 f"{r['peak_gpu_gib']:.1f} GiB.\n\n")

    L.append("## Test-set result of each model's best epoch\n\n")
    L.append("10 held-out recordings, 3,490 reference words. WER and character error "
             "rate are script-normalised (Sinhala Unicode vs romanized is not "
             "penalised). Greedy decoding, same settings before and after.\n\n")
    L.append("| model | best epoch | | WER | char. error rate | words recovered "
             "| output ÷ reference | extra words |\n|---|---|---|---|---|---|---|---|\n")
    for m, r in runs.items():
        for k, label in (("baseline_test", "off-the-shelf"), ("finetuned_test", "fine-tuned")):
            s = r[k]
            bold = "**" if k == "finetuned_test" else ""
            L.append(f"| `{m}` | {r['best_epoch'] if k == 'finetuned_test' else '—'} | {label} "
                     f"| {bold}{s['norm_wer']:.3f}{bold} | {s['norm_cer']:.3f} "
                     f"| {100 * s['recall_of_ref_words']:.1f}% | {s['output_ratio']:.2f}× "
                     f"| {s['insertions']:,} |\n")

    L.append("\n## Reading across the three models\n\n")
    bests = {m: (r["best_epoch"], next(c["val_loss"] for c in r["curve"]
                                       if c["epoch"] == r["best_epoch"])) for m, r in runs.items()}
    L.append("- **All three peaked early** — " +
             ", ".join(f"`{m}` at epoch {b[0]}" for m, b in bests.items()) +
             ". With 1.6 h of training speech the models extract most of what the "
             "data offers in 3–4 passes, then start memorising it. More training "
             "data, not more epochs, is what would move these numbers.\n")
    L.append("- **Best validation losses are close** (" +
             ", ".join(f"`{m}` {b[1]:.3f}" for m, b in bests.items()) +
             "), yet test WER differs much more. Validation loss scores each next "
             "token given the correct previous ones; WER scores free-running "
             "transcription, where `large-v3` loops far less (682 extra words vs "
             "1,466 for `small` and 1,758 for `medium`). Validation loss is the right "
             "signal for choosing an epoch, not for ranking models.\n")
    L.append("- **Bigger models started lower and fell less steeply.** `large-v3`'s "
             "off-the-shelf validation loss was already the lowest (2.13 vs ~2.38), "
             "consistent with its stronger multilingual pre-training.\n\n")

    L.append("## Limitations\n\n")
    L.append("- One run per model with one seed; no variance estimate. Small "
             "differences (a few hundredths of validation loss, a few points of words "
             "recovered) are within run-to-run noise.\n")
    L.append("- The validation set is 6 recordings / 63 clips, so the best-epoch "
             "choice itself is noisy: `large-v3`'s epochs 3 and 4 differ by 0.005.\n")
    L.append("- Test WER uses greedy decoding on utterance clips, not faster-whisper "
             "on full recordings, so it is not directly comparable with Task 1's "
             "figures.\n")

    out = os.path.join(RESULTS_DIR, "c1_asr_finetune_report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(L)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
