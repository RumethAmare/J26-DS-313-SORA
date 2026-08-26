#!/usr/bin/env python3
"""
evaluate_normalized.py -- Task 1 scoring with the script-normalization pass.

Scores every ablation config produced by transcribe_ablation.py against the
gold C1 transcripts, reporting TWO error rates per recording and per config:

  raw-script WER   -- lowercase + strip punctuation only (what evaluate.py in
                      SORA_Dataset already did). Charges the model for
                      decoding in a different script than the annotator used.

  normalized WER   -- both sides pushed through script_normalize into one
                      phonetic skeleton first. Script choice no longer costs
                      anything; what remains is genuine recognition error.

The GAP between them is the size of the scoring artifact. Reporting only one
number conflates "the model mis-heard the words" with "the annotator
romanized and the model didn't", which are completely different problems with
completely different fixes.

Recordings are additionally bucketed by gold script (LATIN / SINHALA / MIXED)
because the artifact is concentrated almost entirely in the romanized-gold
bucket -- that breakdown is the actual finding worth putting in the proposal.

Usage:
    python3 evaluate_normalized.py
    python3 evaluate_normalized.py --configs small_auto small_si
"""
import argparse
import csv
import json
import os
import re

import jiwer

import script_normalize

DATASET_ROOT = "/mnt/F/SLIIT/Research/SORA_Dataset"
C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(DATASET_ROOT, "annotations", "c1")
MANIFEST_PATH = os.path.join(DATASET_ROOT, "manifests", "recordings_current.csv")
PRED_ROOT = os.path.join(C1_ROOT, "predictions")
RESULTS_DIR = os.path.join(C1_ROOT, "results")

# Recordings excluded from C1 modelling per Section D of the data-quality
# audit: R0017's tokens are entirely mislabeled EN, R0008's token file is
# structurally malformed. Their *transcripts* are still scored here (ASR
# doesn't read the token layer) but they are flagged in the output.
D_EXCLUDED = {"J26DS313_R0017", "J26DS313_R0008"}

_SIN = re.compile(r"[඀-෿]")
_LAT = re.compile(r"[A-Za-z]")


def gold_script_bucket(text):
    s = len(_SIN.findall(text))
    l = len(_LAT.findall(text))
    total = s + l
    if total == 0:
        return "EMPTY"
    pct_sin = 100.0 * s / total
    if pct_sin >= 95:
        return "SINHALA"
    if pct_sin <= 5:
        return "LATIN"
    return "MIXED"


def load_manifest_status():
    status = {}
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            status[row["recording_id"]] = row.get("status", "UNKNOWN")
    return status


def concat_text(transcript_json):
    utts = sorted(transcript_json["utterances"], key=lambda u: u["start"])
    return " ".join(u["text"] for u in utts)


def discover_configs():
    if not os.path.isdir(PRED_ROOT):
        return []
    out = []
    for name in sorted(os.listdir(PRED_ROOT)):
        d = os.path.join(PRED_ROOT, name)
        if os.path.isdir(d) and any(f.endswith(".transcript.json") for f in os.listdir(d)):
            out.append(name)
    return out


def safe_wer(ref, hyp):
    """jiwer errors on an empty reference; treat that as undefined."""
    if not ref.strip():
        return float("nan")
    if not hyp.strip():
        return 1.0
    return jiwer.wer(ref, hyp)


def safe_cer(ref, hyp):
    if not ref.strip():
        return float("nan")
    if not hyp.strip():
        return 1.0
    return jiwer.cer(ref, hyp)


def score_config(config, status_by_rid):
    pred_dir = os.path.join(PRED_ROOT, config)
    rows = []
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith(".transcript.json"):
            continue
        rid = fname[: -len(".transcript.json")]
        gold_path = os.path.join(GOLD_DIR, f"{rid}.transcript.json")
        if not os.path.exists(gold_path):
            continue

        with open(gold_path, encoding="utf-8") as fh:
            gold_text = concat_text(json.load(fh))
        with open(os.path.join(pred_dir, fname), encoding="utf-8") as fh:
            pred = json.load(fh)
        pred_text = concat_text(pred)

        raw_ref = script_normalize.normalize_raw(gold_text)
        raw_hyp = script_normalize.normalize_raw(pred_text)
        norm_ref = script_normalize.normalize_scripted(gold_text)
        norm_hyp = script_normalize.normalize_scripted(pred_text)

        rows.append({
            "recording": rid,
            "status": status_by_rid.get(rid, "UNKNOWN"),
            "gold_script": gold_script_bucket(gold_text),
            "d_excluded": rid in D_EXCLUDED,
            "detected_language": pred.get("detected_language"),
            "n_ref_words": len(raw_ref.split()),
            "raw_wer": safe_wer(raw_ref, raw_hyp),
            "raw_cer": safe_cer(raw_ref, raw_hyp),
            "norm_wer": safe_wer(norm_ref, norm_hyp),
            "norm_cer": safe_cer(norm_ref, norm_hyp),
            "_raw_ref": raw_ref, "_raw_hyp": raw_hyp,
            "_norm_ref": norm_ref, "_norm_hyp": norm_hyp,
        })
    return rows


def aggregate(rows):
    rows = [r for r in rows if r["n_ref_words"] > 0]
    if not rows:
        return None
    return {
        "n_recordings": len(rows),
        "n_ref_words": sum(r["n_ref_words"] for r in rows),
        "micro_raw_wer": round(jiwer.wer([r["_raw_ref"] for r in rows],
                                         [r["_raw_hyp"] for r in rows]), 4),
        "micro_norm_wer": round(jiwer.wer([r["_norm_ref"] for r in rows],
                                          [r["_norm_hyp"] for r in rows]), 4),
        "micro_raw_cer": round(jiwer.cer([r["_raw_ref"] for r in rows],
                                         [r["_raw_hyp"] for r in rows]), 4),
        "micro_norm_cer": round(jiwer.cer([r["_norm_ref"] for r in rows],
                                          [r["_norm_hyp"] for r in rows]), 4),
        "macro_raw_wer": round(sum(r["raw_wer"] for r in rows) / len(rows), 4),
        "macro_norm_wer": round(sum(r["norm_wer"] for r in rows) / len(rows), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=None)
    a = ap.parse_args()

    status_by_rid = load_manifest_status()
    configs = a.configs or discover_configs()
    if not configs:
        raise SystemExit(f"No prediction configs found under {PRED_ROOT}. "
                         f"Run transcribe_ablation.py first.")

    print(f"Scoring {len(configs)} config(s): {configs}\n")

    all_results = {}
    for config in configs:
        rows = score_config(config, status_by_rid)
        overall = aggregate(rows)
        by_script = {}
        for bucket in ("LATIN", "MIXED", "SINHALA"):
            sub = [r for r in rows if r["gold_script"] == bucket]
            if sub:
                by_script[bucket] = aggregate(sub)

        print("=" * 78)
        print(f"CONFIG: {config}")
        print("=" * 78)
        print(f"{'recording':18} {'script':8} {'det':4} {'#w':>5} "
              f"{'rawWER':>8} {'normWER':>8} {'gap':>7}")
        for r in sorted(rows, key=lambda r: r["norm_wer"]):
            gap = r["raw_wer"] - r["norm_wer"]
            flag = " *" if r["d_excluded"] else ""
            print(f"{r['recording']:18} {r['gold_script']:8} "
                  f"{str(r['detected_language']):4} {r['n_ref_words']:>5} "
                  f"{r['raw_wer']:>8.3f} {r['norm_wer']:>8.3f} {gap:>7.3f}{flag}")

        print(f"\n  overall: micro raw WER={overall['micro_raw_wer']:.4f}  "
              f"micro norm WER={overall['micro_norm_wer']:.4f}  "
              f"(gap {overall['micro_raw_wer'] - overall['micro_norm_wer']:+.4f})")
        print("  by gold script:")
        for bucket, m in by_script.items():
            print(f"    {bucket:8} n={m['n_recordings']:2}  "
                  f"raw={m['micro_raw_wer']:.4f}  norm={m['micro_norm_wer']:.4f}  "
                  f"gap={m['micro_raw_wer'] - m['micro_norm_wer']:+.4f}")
        print()

        all_results[config] = {
            "overall": overall,
            "by_gold_script": by_script,
            "per_recording": [
                {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in r.items() if not k.startswith("_")}
                for r in rows
            ],
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "c1_ablation_eval.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({
            "note": "raw_wer = lowercase+punct-strip only (script differences "
                    "penalized); norm_wer = both sides transliterated to one "
                    "phonetic skeleton via script_normalize.py.",
            "d_excluded_recordings": sorted(D_EXCLUDED),
            "configs": all_results,
        }, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {out_json}")

    write_report(all_results)


def write_report(all_results):
    L = []
    L.append("# C1 Task 1 — Off-the-shelf ASR baseline (script-normalized + ablation)\n\n")
    L.append("Scores faster-whisper against the gold C1 transcripts under two\n")
    L.append("different normalizations, across a language-forcing ablation.\n\n")

    L.append("## Why two WER numbers\n\n")
    L.append("Gold transcripts are not written in one script. Nine of 27 recordings are\n")
    L.append("100% romanized Latin (`mama`, `thiyenawa`); most of the rest are majority\n")
    L.append("Sinhala Unicode. faster-whisper auto-detects **one** language per recording\n")
    L.append("and decodes the whole thing in that script. Scoring those directly compares\n")
    L.append("romanized reference against Sinhala-Unicode hypothesis, which pins WER near\n")
    L.append("1.0 regardless of whether the model heard the words correctly.\n\n")
    L.append("- **raw-script WER** — lowercase + strip punctuation only. Script mismatch is\n")
    L.append("  charged as error. This is what the original `evaluate.py` reported.\n")
    L.append("- **normalized WER** — both sides transliterated to one phonetic skeleton\n")
    L.append("  (`script_normalize.py`) first, so script choice costs nothing.\n\n")
    L.append("The gap between them measures the scoring artifact; what remains after\n")
    L.append("normalization is genuine recognition error.\n\n")

    L.append("## Ablation summary\n\n")
    L.append("| config | micro raw WER | micro norm WER | gap | macro norm WER |\n")
    L.append("|---|---|---|---|---|\n")
    for config, res in all_results.items():
        o = res["overall"]
        gap = o["micro_raw_wer"] - o["micro_norm_wer"]
        L.append(f"| `{config}` | {o['micro_raw_wer']:.4f} | {o['micro_norm_wer']:.4f} "
                 f"| {gap:+.4f} | {o['macro_norm_wer']:.4f} |\n")
    L.append("\n")

    L.append("## Breakdown by gold script\n\n")
    L.append("The artifact is concentrated in the romanized-gold bucket, which is the\n")
    L.append("point: those recordings were never as badly recognized as raw WER implied.\n\n")
    L.append("| config | gold script | n | micro raw WER | micro norm WER | gap |\n")
    L.append("|---|---|---|---|---|---|\n")
    for config, res in all_results.items():
        for bucket, m in res["by_gold_script"].items():
            gap = m["micro_raw_wer"] - m["micro_norm_wer"]
            L.append(f"| `{config}` | {bucket} | {m['n_recordings']} "
                     f"| {m['micro_raw_wer']:.4f} | {m['micro_norm_wer']:.4f} | {gap:+.4f} |\n")
    L.append("\n")

    L.append("## Per-recording detail\n\n")
    for config, res in all_results.items():
        L.append(f"### `{config}`\n\n")
        L.append("| recording | gold script | detected | #words | raw WER | norm WER | gap |\n")
        L.append("|---|---|---|---|---|---|---|\n")
        for r in sorted(res["per_recording"], key=lambda r: r["norm_wer"]):
            gap = r["raw_wer"] - r["norm_wer"]
            star = " \\*" if r["d_excluded"] else ""
            L.append(f"| {r['recording']}{star} | {r['gold_script']} | "
                     f"{r['detected_language']} | {r['n_ref_words']} | "
                     f"{r['raw_wer']:.3f} | {r['norm_wer']:.3f} | {gap:+.3f} |\n")
        L.append("\n")
    L.append("\\* excluded from C1 modelling per the Section D data-quality audit "
             "(R0017 mislabeled tokens, R0008 malformed token file). Their transcripts "
             "are still ASR-scorable and are shown for completeness.\n\n")

    L.append("## Motivating fine-tuning as future work\n\n")
    L.append("Even after the script artifact is removed, normalized WER stays far above\n")
    L.append("anything usable downstream. That residue is the real problem C1 exists to\n")
    L.append("solve, and it is not a scoring or decoding-parameter issue: off-the-shelf\n")
    L.append("Whisper has essentially no exposure to Sinhala-English code-mixed speech,\n")
    L.append("and forcing a single language code cannot fix input that changes language\n")
    L.append("mid-utterance. The ablation shows how little headroom is available from\n")
    L.append("decoding settings alone — which is precisely the argument for domain\n")
    L.append("fine-tuning on a larger corpus (the TAF's 10–15hr target) rather than\n")
    L.append("further tuning of an off-the-shelf model. Fine-tuning is therefore scoped\n")
    L.append("as post-proposal work, contingent on corpus growth.\n")

    out = os.path.join(RESULTS_DIR, "c1_report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(L)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
