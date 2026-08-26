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


def error_composition(refs, hyps):
    """Break WER into hits / substitutions / deletions / insertions.

    Essential here, because WER alone is actively misleading on this corpus.
    Every config lands near 0.99, which reads as "they all fail identically" --
    but they fail in opposite ways. Sinhala-mode Whisper mostly *abstains*
    (huge deletion count, almost no insertions), while English-mode
    over-generates (insertions and substitutions dominate) and in doing so
    actually recovers far more reference words. Two systems with the same WER
    can differ by an order of magnitude in words recovered, and only the
    composition shows it.
    """
    out = jiwer.process_words(refs, hyps)
    err = out.substitutions + out.deletions + out.insertions
    return {
        "hits": out.hits,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "pct_substitutions": round(100 * out.substitutions / err, 1) if err else 0.0,
        "pct_deletions": round(100 * out.deletions / err, 1) if err else 0.0,
        "pct_insertions": round(100 * out.insertions / err, 1) if err else 0.0,
    }


def aggregate(rows):
    rows = [r for r in rows if r["n_ref_words"] > 0]
    if not rows:
        return None
    norm_refs = [r["_norm_ref"] for r in rows]
    norm_hyps = [r["_norm_hyp"] for r in rows]
    comp = error_composition(norm_refs, norm_hyps)
    n_ref_words = sum(r["n_ref_words"] for r in rows)
    n_hyp_words = sum(len(r["_norm_hyp"].split()) for r in rows)
    return {
        "n_recordings": len(rows),
        "n_ref_words": n_ref_words,
        "n_hyp_words": n_hyp_words,
        # How much output the model produced relative to what it should have.
        # Far below 100% means abstention; above means over-generation.
        "output_ratio": round(n_hyp_words / n_ref_words, 3) if n_ref_words else 0.0,
        # Share of reference words actually recovered. This is the number that
        # separates configs WER collapses together.
        "recall_of_ref_words": round(comp["hits"] / n_ref_words, 4) if n_ref_words else 0.0,
        "error_composition": comp,
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

        c = overall["error_composition"]
        print(f"\n  overall: micro raw WER={overall['micro_raw_wer']:.4f}  "
              f"micro norm WER={overall['micro_norm_wer']:.4f}  "
              f"(gap {overall['micro_raw_wer'] - overall['micro_norm_wer']:+.4f})")
        print(f"  words recovered: {c['hits']}/{overall['n_ref_words']} "
              f"({100 * overall['recall_of_ref_words']:.1f}% of reference)   "
              f"output ratio: {overall['output_ratio']:.2f}x")
        print(f"  error mix: S={c['pct_substitutions']:.0f}% "
              f"D={c['pct_deletions']:.0f}% I={c['pct_insertions']:.0f}%  "
              f"({'abstains' if c['pct_deletions'] > 50 else 'over-generates'})")
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
    L.append("and decodes the whole thing in that script, so a romanized reference gets\n")
    L.append("compared against a Sinhala-Unicode hypothesis.\n\n")
    L.append("- **raw-script WER** — lowercase + strip punctuation only. Script mismatch is\n")
    L.append("  charged as error.\n")
    L.append("- **normalized WER** — both sides transliterated to one phonetic skeleton\n")
    L.append("  (`script_normalize.py`) first, so script choice costs nothing.\n\n")
    L.append("**The expected result did not materialise.** The working assumption was that\n")
    L.append("script mismatch was badly inflating WER and that normalizing would reveal\n")
    L.append("substantially better real performance. It does not: the gap is only about\n")
    L.append("0.3–0.6 WER points. The artifact is real but small, and off-the-shelf\n")
    L.append("performance is genuinely near-total failure rather than a measurement\n")
    L.append("illusion. Both numbers are reported so that claim is checkable.\n\n")

    L.append("### A scoring bug found while building this\n\n")
    L.append("The normalization previously used for C1 scoring,\n")
    L.append("`re.sub(r\"[^\\w\\s]\", \" \", text)`, silently corrupts Sinhala. Sinhala is an\n")
    L.append("abugida whose vowels attach as *combining marks* (Unicode category Mn), and\n")
    L.append("Python's `\\w` does not match Mn — so every vowel sign became a space and each\n")
    L.append("word shattered into loose consonants (`ට්‍රිප් එකට` → `ට ර ප එකට`).\n\n")
    L.append("The damage is not neutral noise: it makes scores look *better* than reality.\n")
    L.append("A 72-word Sinhala utterance becomes ~53 single-character tokens, and frequent\n")
    L.append("letters (ක, න, ම, ප) then match the equally-shredded hypothesis by\n")
    L.append("coincidence, crediting matches that never occurred. Any previously quoted C1\n")
    L.append("WER computed that way — including the ~93.7% figure in the implementation\n")
    L.append("plan — is optimistic for exactly the recordings written in Sinhala. Corrected,\n")
    L.append("raw-script WER is ~0.999. `script_normalize._squash()` now filters by Unicode\n")
    L.append("category instead, keeping letters, numbers and marks.\n\n")

    L.append("## Ablation summary\n\n")
    L.append("| config | micro raw WER | micro norm WER | gap | macro norm WER |\n")
    L.append("|---|---|---|---|---|\n")
    for config, res in all_results.items():
        o = res["overall"]
        gap = o["micro_raw_wer"] - o["micro_norm_wer"]
        L.append(f"| `{config}` | {o['micro_raw_wer']:.4f} | {o['micro_norm_wer']:.4f} "
                 f"| {gap:+.4f} | {o['macro_norm_wer']:.4f} |\n")
    L.append("\n")

    L.append("## Why WER alone is the wrong headline here\n\n")
    L.append("Every configuration lands near 0.99 WER, which reads as \"they all fail\n")
    L.append("the same way\". They do not. The error composition shows two opposite\n")
    L.append("failure modes that WER collapses onto the same number.\n\n")
    L.append("| config | words recovered | % of reference | output ratio | S | D | I | behaviour |\n")
    L.append("|---|---|---|---|---|---|---|---|\n")
    for config, res in all_results.items():
        o = res["overall"]
        c = o["error_composition"]
        behaviour = "abstains" if c["pct_deletions"] > 50 else "over-generates"
        L.append(f"| `{config}` | {c['hits']}/{o['n_ref_words']} "
                 f"| {100 * o['recall_of_ref_words']:.1f}% | {o['output_ratio']:.2f}x "
                 f"| {c['pct_substitutions']:.0f}% | {c['pct_deletions']:.0f}% "
                 f"| {c['pct_insertions']:.0f}% | {behaviour} |\n")
    L.append("\n")
    L.append("Read the *words recovered* column, not the WER column. Sinhala-mode\n")
    L.append("Whisper (`small_auto`, `small_si`) does not mis-transcribe this audio so\n")
    L.append("much as decline to transcribe it: roughly three quarters of its errors are\n")
    L.append("deletions and it emits only a quarter of the expected words. Forced-English\n")
    L.append("decoding emits slightly more than the reference length and recovers an\n")
    L.append("order of magnitude more reference words, paying for it in substitutions and\n")
    L.append("insertions — which is why its WER barely moves.\n\n")
    L.append("That asymmetry is a genuine signal about the data, not a decoding curiosity.\n")
    L.append("These recordings carry enough English that an English-forced decoder finds\n")
    L.append("real purchase on them, while the Sinhala-forced decoder — nominally the\n")
    L.append("\"correct\" setting for Sinhala-English speech — mostly produces nothing.\n")
    L.append("Any C1 evaluation that quotes WER alone will rank these configurations as\n")
    L.append("equivalent and miss this entirely, so downstream tasks should track words\n")
    L.append("recovered and error composition alongside WER.\n\n")

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

    L.append("## Ablation coverage — what is missing\n\n")
    L.append("The plan specified a two-axis ablation (model size × language forcing).\n")
    L.append("Only the **language-forcing axis** was run. The `medium` weights are not in\n")
    L.append("this host's HuggingFace cache and cannot be fetched: the machine has no\n")
    L.append("working outbound HTTPS route, and HuggingFace requests hang in `SYN-SENT`\n")
    L.append("rather than failing. The size axis therefore needs either a networked\n")
    L.append("machine or the weights copied in manually. See `docs/RUNNING_NOTES.md`.\n\n")
    L.append("Throughput on the RTX 5050 (8GB, float16), from `c1_ablation_runs.json`,\n")
    L.append("over 3070s of audio — useful as an early input to Task 7:\n\n")
    L.append("| config | wall clock | real-time factor |\n")
    L.append("|---|---|---|\n")
    runs_path = os.path.join(RESULTS_DIR, "c1_ablation_runs.json")
    if os.path.exists(runs_path):
        with open(runs_path, encoding="utf-8") as fh:
            for m in json.load(fh)["configs"]:
                L.append(f"| `{m['config']}` | {m['total_wall_s']:.0f}s "
                         f"| {m['real_time_factor']:.3f} |\n")
    L.append("\nThe forced-English arm is roughly 4x faster in wall-clock terms despite\n")
    L.append("producing *more* text, so the speed difference reflects decoding dynamics on\n")
    L.append("mismatched audio, not less work done.\n\n")

    L.append("## Motivating fine-tuning as future work\n\n")
    L.append("Removing the script artifact moves normalized WER by well under one point,\n")
    L.append("and no language-forcing setting brings it below ~0.98. The failure is\n")
    L.append("therefore not a scoring problem and not a decoding-parameter problem:\n")
    L.append("off-the-shelf Whisper-small has essentially no usable competence on this\n")
    L.append("audio, and a single language code cannot describe input that changes\n")
    L.append("language mid-utterance. The ablation's value is in showing how little\n")
    L.append("headroom decoding settings offer, which is the argument for domain\n")
    L.append("fine-tuning on a larger corpus (the TAF's 10–15hr target) rather than\n")
    L.append("further tuning of an off-the-shelf model.\n\n")
    L.append("The error-composition result sharpens what fine-tuning has to fix. The\n")
    L.append("dominant failure in Sinhala mode is *abstention*, not confusion — the model\n")
    L.append("emits a quarter of the expected words. That is a different target from\n")
    L.append("reducing substitutions, and it suggests the near-term gain may come from\n")
    L.append("segmentation and decoding behaviour (VAD settings, forced decoding, chunk\n")
    L.append("length) as much as from acoustic modelling. Worth testing before assuming\n")
    L.append("more data alone resolves it.\n\n")
    L.append("Fine-tuning stays scoped as post-proposal work, contingent on corpus growth.\n")

    out = os.path.join(RESULTS_DIR, "c1_report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(L)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
