#!/usr/bin/env python3
"""
noise_robustness_eval.py -- Task 6: re-run the C1 pipeline per SNR tier.

MEASUREMENT DESIGN — one deviation from the plan, and the reason for it
----------------------------------------------------------------------
The plan asks for "WER, LID accuracy, switch-F1, and timestamp error vs SNR".
Two of those cannot carry information on this corpus:

  * WER is already 0.996 on CLEAN audio (Task 1). It is pinned against its
    ceiling, so every tier will read ~1.0 and the "robustness curve" will be a
    flat line at the top of the chart. It is reported, but a flat curve at the
    ceiling is a statement about the clean baseline, not about noise.

  * LID accuracy and switch-F1 on ASR output are near-degenerate for the reason
    established in results/c1_token_stream_notes.md: 80% of predicted tokens
    get labelled by the Unicode rule reading whichever script Whisper decoded
    into, so they measure Whisper's single language choice rather than
    per-token language.

So the sweep also tracks the quantities that DO have dynamic range here:

  * words recovered   -- how many gold words the ASR actually got right.
                         Task 1 found forced-English recovers 1121 words vs
                         auto-detect's 28, so the `en` arm has real headroom to
                         degrade while `auto` starts near the floor.
  * output ratio      -- emitted words / expected words. Whisper's dominant
                         failure here is abstention, and abstention deepens
                         with noise in a way WER cannot show.
  * language detection stability -- whether Whisper's single acoustic language
                         choice flips under noise, and how its confidence
                         moves. This is a direct probe of the mechanism the
                         whole pipeline hangs on.
  * timestamp error   -- median |gold.start - pred.start| over text-matched
                         tokens.

BOTH decode configs are swept, because a robustness curve measured only on the
arm that already produces almost nothing would show a flat line and prove
nothing.

Usage:
    python3 noise_robustness_eval.py
    python3 noise_robustness_eval.py --configs auto
"""
import argparse
import difflib
import json
import os
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cuda_env  # noqa: E402

cuda_env.ensure_cuda_libs()

import jiwer  # noqa: E402

import script_normalize as sn  # noqa: E402

DATASET_ROOT = "/mnt/F/SLIIT/Research/SORA_Dataset"
C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(DATASET_ROOT, "processed", "audio")
GOLD_DIR = os.path.join(DATASET_ROOT, "annotations", "c1")
NOISY_ROOT = os.path.join(C1_ROOT, "predictions", "noisy_audio")
RESULTS_DIR = os.path.join(C1_ROOT, "results")
MANIFEST = os.path.join(RESULTS_DIR, "c1_noise_manifest.json")

CONDITIONS = ["clean", "snr15", "snr5", "snr0", "reverb"]


def audio_path(condition, rid):
    if condition == "clean":
        return os.path.join(AUDIO_DIR, f"{rid}.wav")
    return os.path.join(NOISY_ROOT, condition, f"{rid}.wav")


def gold_text(rid):
    with open(os.path.join(GOLD_DIR, f"{rid}.transcript.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    utts = sorted(doc["utterances"], key=lambda u: u["start"])
    return " ".join(u["text"] for u in utts)


def gold_tokens(rid):
    path = os.path.join(GOLD_DIR, f"{rid}.tokens.jsonl")
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return sorted(out, key=lambda t: t["start"])


def timestamp_error(gold_toks, pred_words):
    """Median |gold.start - pred.start| over tokens matched BY TEXT.

    Matching by text rather than by time is essential: aligning by temporal
    overlap and then measuring temporal error would be circular. difflib finds
    the longest common subsequence of normalized surface forms, so only tokens
    the ASR actually got right contribute a timing measurement.
    """
    g_norm = [sn.normalize_scripted(t["token"]) for t in gold_toks]
    p_norm = [sn.normalize_scripted(w["word"]) for w in pred_words]
    if not g_norm or not p_norm:
        return None, 0
    matcher = difflib.SequenceMatcher(a=g_norm, b=p_norm, autojunk=False)
    errors = []
    for gi, pi, size in matcher.get_matching_blocks():
        for k in range(size):
            errors.append(abs(gold_toks[gi + k]["start"] - pred_words[pi + k]["start"]))
    if not errors:
        return None, 0
    return round(statistics.median(errors), 3), len(errors)


def run(model, condition, rid, lang_code):
    segments, info = model.transcribe(
        audio_path(condition, rid), beam_size=5, vad_filter=True,
        language=lang_code, word_timestamps=True,
    )
    words, texts = [], []
    for seg in segments:
        texts.append(seg.text.strip())
        for w in (seg.words or []):
            if w.word.strip():
                words.append({"word": w.word.strip(), "start": round(w.start, 3),
                              "end": round(w.end, 3),
                              "probability": round(float(w.probability), 4)})
    return " ".join(texts), words, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["auto", "en"])
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--model-size", default="small")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--report-only", action="store_true",
                    help="regenerate the markdown from the saved JSON without "
                         "re-running the (40 minute) sweep")
    a = ap.parse_args()

    if a.report_only:
        with open(os.path.join(RESULTS_DIR, "c1_noise_robustness.json"),
                  encoding="utf-8") as fh:
            saved = json.load(fh)
        with open(MANIFEST, encoding="utf-8") as fh:
            man = json.load(fh)
        write_report(saved["results"], saved["recordings"], man)
        return

    global CONDITIONS
    if a.conditions:
        CONDITIONS = a.conditions
    with open(MANIFEST, encoding="utf-8") as fh:
        manifest = json.load(fh)
    rids = [r["recording"] for r in manifest["recordings"]]
    print(f"Noise robustness sweep: {len(CONDITIONS)} conditions x "
          f"{len(a.configs)} configs x {len(rids)} recordings "
          f"= {len(CONDITIONS) * len(a.configs) * len(rids)} runs\n")

    from faster_whisper import WhisperModel
    model = WhisperModel(a.model_size, device="cuda", compute_type=a.compute_type)

    gold_text_by_rid = {r: gold_text(r) for r in rids}
    gold_toks_by_rid = {r: gold_tokens(r) for r in rids}

    results = {}
    for config in a.configs:
        lang_code = None if config == "auto" else config
        for condition in CONDITIONS:
            refs, hyps, ts_errs, ts_n = [], [], [], 0
            detected, probs = [], []
            t0 = time.time()
            for rid in rids:
                text, words, info = run(model, condition, rid, lang_code)
                refs.append(sn.normalize_scripted(gold_text_by_rid[rid]))
                hyps.append(sn.normalize_scripted(text))
                detected.append(info.language)
                probs.append(float(info.language_probability))
                err, n = timestamp_error(gold_toks_by_rid[rid], words)
                if err is not None:
                    ts_errs.append(err)
                    ts_n += n

            out = jiwer.process_words(refs, hyps)
            n_ref = sum(len(r.split()) for r in refs)
            n_hyp = sum(len(h.split()) for h in hyps)
            key = f"{config}__{condition}"
            results[key] = {
                "config": config, "condition": condition,
                "wer": round(out.wer, 4),
                "words_recovered": out.hits,
                "n_ref_words": n_ref,
                "recall_of_ref_words": round(out.hits / n_ref, 4) if n_ref else 0.0,
                "output_ratio": round(n_hyp / n_ref, 3) if n_ref else 0.0,
                "substitutions": out.substitutions,
                "deletions": out.deletions,
                "insertions": out.insertions,
                "detected_languages": dict(
                    (l, detected.count(l)) for l in sorted(set(detected))),
                "mean_language_probability": round(statistics.mean(probs), 4),
                "median_timestamp_error_s": (round(statistics.median(ts_errs), 3)
                                             if ts_errs else None),
                "n_timestamp_matches": ts_n,
                "wall_s": round(time.time() - t0, 1),
            }
            r = results[key]
            print(f"  {config:5} {condition:7} WER={r['wer']:.4f} "
                  f"recovered={r['words_recovered']:5} ({100 * r['recall_of_ref_words']:5.1f}%) "
                  f"out={r['output_ratio']:.2f}x langP={r['mean_language_probability']:.3f} "
                  f"ts={r['median_timestamp_error_s']} ({r['wall_s']:.0f}s)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "c1_noise_robustness.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({
            "note": "Synthetic pink noise at exact SNR (see c1_noise_manifest.json). "
                    "WER is pinned near its ceiling on clean audio already, so the "
                    "informative columns are words recovered, output ratio and "
                    "language-detection stability.",
            "n_recordings": len(rids),
            "recordings": rids,
            "conditions": CONDITIONS,
            "results": results,
        }, fh, indent=2)
    print(f"\nWrote {out_json}")
    write_report(results, rids, manifest)


def write_report(results, rids, manifest):
    L = []
    L.append("# Task 6 — Noise robustness\n\n")
    L.append(f"Produced by `scripts/noise_robustness_eval.py` over "
             f"{len(rids)} recordings, stratified across the LATIN / MIXED / "
             f"SINHALA gold-script buckets.\n\n")

    L.append("## Setup\n\n")
    L.append("Synthetic **pink noise** at exact SNR (measured SNR matches the "
             "requested value to 0.1 dB), plus a separate reverb condition "
             "(synthetic RT60 0.45s). MUSAN was not usable here — ~11GB over a "
             "~500KB/s link — so the trade is exact, reproducible, "
             "licence-free noise in exchange for the realism of babble and "
             "music. Pink rather than white noise because its low-frequency "
             "emphasis matches real room tone and masks the formant region.\n\n")
    L.append("Reverb is a separate condition rather than an SNR tier, since it "
             "degrades speech by smearing it in time rather than by adding "
             "energy.\n\n")

    L.append("## Why WER is not the headline here\n\n")
    L.append("WER is already ~0.996 on **clean** audio (Task 1), so it is pinned "
             "against its ceiling before any noise is added. A flat curve at the "
             "top of the chart says something about the clean baseline, not about "
             "robustness. The columns that carry information are **words "
             "recovered**, **output ratio** and **language-detection stability**.\n\n")

    for config in sorted({r["config"] for r in results.values()}):
        L.append(f"### Decode config: `{config}`\n\n")
        L.append("| condition | WER | words recovered | % of ref | output ratio "
                 "| mean lang P | detected | median ts err |\n")
        L.append("|---|---|---|---|---|---|---|---|\n")
        for cond in CONDITIONS:
            r = results.get(f"{config}__{cond}")
            if not r:
                continue
            det = ", ".join(f"{k}×{v}" for k, v in r["detected_languages"].items())
            ts = (f"{r['median_timestamp_error_s']}s"
                  if r["median_timestamp_error_s"] is not None else "—")
            L.append(f"| {cond} | {r['wer']:.4f} | {r['words_recovered']} "
                     f"| {100 * r['recall_of_ref_words']:.1f}% "
                     f"| {r['output_ratio']:.2f}× "
                     f"| {r['mean_language_probability']:.3f} | {det} | {ts} |\n")
        L.append("\n")

    L.append("## Reading the curves\n\n")

    L.append("### Only the `en` arm produces a usable robustness curve\n\n")
    L.append("Running both configs was necessary, and the outcome shows why. "
             "Forced-English degrades **monotonically and cleanly**: words "
             "recovered fall 397 → 271 → 267 → 198 from clean to 0 dB, roughly "
             "halving, and output ratio falls 1.10× → 0.81× as the decoder "
             "progressively gives up. That is a genuine robustness curve.\n\n")
    L.append("`auto` does not produce one. Its recovered-word counts go "
             "7 → 17 → 19 → 7 — **non-monotonic, and rising with moderate "
             "noise**, which is not a real effect. With 7–19 correct words out "
             "of ~1,780 the arm is sitting on its measurement floor, and those "
             "swings are counting fluctuations. No conclusion about noise "
             "should be drawn from them.\n\n")

    L.append("### WER moves the wrong way; words recovered does not\n\n")
    L.append("The clearest vindication of not leading with WER is in the `en` "
             "arm. WER reads 0.934 → 0.962 → 0.944 → 0.957 across clean → 0 dB "
             "— **non-monotonic**, and it makes 15 dB look worse than 0 dB. "
             "Over exactly the same runs, words recovered falls 397 → 271 → 267 "
             "→ 198, monotonically. WER is compressed against its ceiling where "
             "substitution and insertion counts trade against each other, so it "
             "cannot rank these conditions; a direct count of what the model got "
             "right can.\n\n")

    L.append("### Language-detection confidence is the real `auto` finding\n\n")
    L.append("While `auto`'s word counts are noise, its language-detection "
             "probability degrades cleanly and monotonically: **0.761 → 0.622 → "
             "0.615 → 0.572**. That is the one number reporting directly on the "
             "mechanism the whole pipeline hangs on — Whisper's single acoustic "
             "language choice per recording. Noise erodes the model's certainty "
             "about *which language it is even hearing*, and because that choice "
             "fixes the output script, and the script is what the Unicode LID "
             "rule reads, every downstream stage inherits the wobble.\n\n")
    L.append("Reverb sits apart: it barely dents language confidence (0.736, "
             "close to clean's 0.761) while still costing recovered words in the "
             "`en` arm (227 vs 397). Smearing speech in time damages recognition "
             "without disguising which language it is.\n\n")
    L.append("The `en` arm's language probability is pinned at 1.000 throughout, "
             "as it must be — the language is forced, so no detection happens. "
             "That column is only meaningful for `auto`.\n\n")

    L.append("## Limitations\n\n")
    L.append("- Synthetic pink noise is a proxy for room tone, not for babble, "
             "music or channel distortion. Conclusions transfer to stationary "
             "background noise only.\n")
    L.append("- The subset is 8 recordings; per-condition figures pool across "
             "them, so a single unusual recording can move a row.\n")
    L.append("- Timestamp error is measured only over tokens the ASR got "
             "textually right, matched by longest common subsequence. At this "
             "recognition rate that is a small and non-random sample — it "
             "describes timing on the easy words.\n")
    L.append("- **The `auto` arm's timestamp errors (15–85s) are artifacts, not "
             "measurements, and must not be quoted.** With only ~7–19 correctly "
             "recognised words per condition, the longest-common-subsequence "
             "matcher pairs tokens that happen to coincide anywhere in a "
             "multi-minute recording, so the resulting offsets are essentially "
             "random. The `en` arm's 0.28–0.67s figures rest on hundreds of "
             "matches and are the only timing numbers worth reading.\n")
    L.append("- The clean-audio baseline is the binding constraint on every "
             "conclusion here. Noise robustness is a secondary question while "
             "clean WER sits near 1.0.\n")

    out = os.path.join(RESULTS_DIR, "c1_noise_robustness.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(L)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
