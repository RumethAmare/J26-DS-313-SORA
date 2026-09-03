#!/usr/bin/env python3
"""
Phase 0 scoring: WER/CER for every decode config, raw and script-folded.

Three normalizations are reported per config, and the differences between them
carry the finding:

  raw     as-annotated, script preserved. The honest "did the model produce the
          reference" number, and the headline figure.
  folded  both sides mapped through src.script_fold into a common romanized
          space. Script-agnostic, so it measures RECOGNITION quality with the
          Sinhala-vs-Latin mismatch removed. raw minus folded is the cost of
          script mismatch, in WER points.
  legacy  bug-compatible with SORA_Dataset/scripts/c1/evaluate.py, whose regex
          strips Sinhala vowel signs. Reported only to prove this harness reads
          the same data as the prior 0.937 baseline. Never a headline number.

Everything is broken down by split (train/eval) and by script_group, because
"forced-si is better" is only meaningful if it holds on the native-script
recordings AND does not wreck the romanized or English-only ones.

Writes eval/baseline_decisions.json — the machine-readable contract every later
phase reads — and eval/baseline_report.md alongside it.

Usage:
    python -m src.evaluate_asr
"""
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jiwer

from src import sora_paths
from src.script_fold import fold, normalize_legacy, normalize_raw
from src.split_guard import load_manifest_rows

NORMALIZERS = {"raw": normalize_raw, "folded": fold, "legacy": normalize_legacy}

# The prior baseline this harness must reproduce, from
# SORA_Dataset/results/c1_report.md (faster-whisper small, auto, legacy regex).
PRIOR_BASELINE_WER = 0.9372
REPRO_TOLERANCE = 0.05


def concat_text(payload):
    """Whole-recording text, in time order. Whisper segments never align to gold
    utt_ids, so recording-level concatenation is the only fair comparison."""
    utts = sorted(payload["utterances"], key=lambda u: u["start"])
    return " ".join(u["text"] for u in utts)


def safe_score(ref, hyp):
    if not ref.strip():
        return None, None
    return jiwer.wer(ref, hyp or " "), jiwer.cer(ref, hyp or " ")


def error_composition(refs, hyps):
    """
    Split the errors into substitutions / deletions / insertions.

    This separates two very different failure modes that both read as WER near
    1.0: a model that MISHEARS produces substitutions, a model that stays
    SILENT produces deletions. They call for opposite fixes, so the headline
    number on its own cannot direct the work.
    """
    usable = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not usable:
        return None
    out = jiwer.process_words([r for r, _ in usable], [h or " " for _, h in usable])
    n_ref = out.substitutions + out.deletions + out.hits
    if not n_ref:
        return None
    return {
        "hits": out.hits,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "n_ref_words": n_ref,
        "hit_rate": round(out.hits / n_ref, 4),
        "sub_rate": round(out.substitutions / n_ref, 4),
        "del_rate": round(out.deletions / n_ref, 4),
        "ins_rate": round(out.insertions / n_ref, 4),
    }


def aggregate(rows, key):
    """Micro (pooled, length-weighted) and macro (per-recording mean) for one
    normalization. Micro is the number to quote; macro exposes whether a few
    short recordings dominate."""
    usable = [r for r in rows if r[f"{key}_wer"] is not None]
    if not usable:
        return None
    refs = [r[f"{key}_ref"] for r in usable]
    hyps = [r[f"{key}_hyp"] for r in usable]
    return {
        "n_recordings": len(usable),
        "n_ref_words": sum(len(r.split()) for r in refs),
        "micro_wer": round(jiwer.wer(refs, hyps), 4),
        "micro_cer": round(jiwer.cer(refs, hyps), 4),
        "macro_wer": round(sum(r[f"{key}_wer"] for r in usable) / len(usable), 4),
        "macro_cer": round(sum(r[f"{key}_cer"] for r in usable) / len(usable), 4),
        "errors": error_composition(refs, hyps),
    }


def score_config(config_dir_path, manifest):
    """Score one decode config. Returns None if it has no predictions."""
    meta_path = os.path.join(config_dir_path, "_config_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    rows = []
    for rid, info in sorted(manifest.items()):
        pred_path = os.path.join(config_dir_path, f"{rid}.transcript.json")
        if not os.path.exists(pred_path):
            continue
        with open(pred_path, encoding="utf-8") as fh:
            pred = json.load(fh)

        gold_text = concat_text(sora_paths.load_transcript(rid))
        hyp_text = concat_text(pred)

        row = {
            "recording": rid,
            "split": info["split"],
            "script_group": info["script_group"],
            "detected_language": pred.get("detected_language"),
        }
        for key, norm in NORMALIZERS.items():
            ref, hyp = norm(gold_text), norm(hyp_text)
            wer, cer = safe_score(ref, hyp)
            row[f"{key}_ref"] = ref
            row[f"{key}_hyp"] = hyp
            row[f"{key}_wer"] = round(wer, 4) if wer is not None else None
            row[f"{key}_cer"] = round(cer, 4) if cer is not None else None
        rows.append(row)

    if not rows:
        return None

    breakdowns = {}
    for key in NORMALIZERS:
        breakdowns[key] = {
            "overall": aggregate(rows, key),
            "by_split": {
                s: aggregate([r for r in rows if r["split"] == s], key)
                for s in sorted({r["split"] for r in rows})
            },
            "by_script_group": {
                g: aggregate([r for r in rows if r["script_group"] == g], key)
                for g in sorted({r["script_group"] for r in rows})
            },
        }

    return {
        "model_size": meta["model_size"],
        "decode_mode": meta["decode_mode"],
        "real_time_factor": meta.get("real_time_factor"),
        "total_decode_s": meta.get("total_decode_s"),
        "n_scored": len(rows),
        "metrics": breakdowns,
        "per_recording": [
            {k: v for k, v in r.items() if not k.endswith(("_ref", "_hyp"))}
            for r in rows
        ],
    }


def eval_micro(config, key="folded", metric="micro_wer"):
    """The eval-split number a config is judged on. Falls back to overall if the
    eval split somehow has no scorable rows."""
    by_split = config["metrics"][key]["by_split"]
    agg = by_split.get("eval") or config["metrics"][key]["overall"]
    return agg[metric] if agg else float("inf")


def finite(value):
    """inf/nan -> None, so the value survives a round-trip through strict JSON."""
    return value if isinstance(value, (int, float)) and value == value \
        and value not in (float("inf"), float("-inf")) else None


def fmt(value, width=6):
    if not isinstance(value, (int, float)) or finite(value) is None:
        return " " * (width - 1) + "-"
    return f"{value:{width}.3f}"


def main():
    manifest = {
        r["recording_id"]: {"split": r["split"], "script_group": r["script_group"]}
        for r in load_manifest_rows()
    }

    pred_root = sora_paths.predictions_dir()
    if not os.path.isdir(pred_root):
        print(f"No predictions at {pred_root}. Run `python -m src.decode_compare` first.")
        return 1

    configs = []
    for name in sorted(os.listdir(pred_root)):
        scored = score_config(os.path.join(pred_root, name), manifest)
        if scored:
            configs.append(scored)

    if not configs:
        print("No scorable configs found. Run `python -m src.decode_compare` first.")
        return 1

    # ── harness reproduction check ────────────────────────────
    repro = next((c for c in configs
                  if c["model_size"] == "small" and c["decode_mode"] == "auto"), None)
    repro_note = None
    if repro:
        got = repro["metrics"]["legacy"]["overall"]["micro_wer"]
        delta = abs(got - PRIOR_BASELINE_WER)
        ok = delta <= REPRO_TOLERANCE
        repro_note = {
            "prior_baseline_micro_wer": PRIOR_BASELINE_WER,
            "this_harness_legacy_micro_wer": got,
            "delta": round(delta, 4),
            "reproduced": ok,
            "source": "SORA_Dataset/results/c1_report.md",
        }
        print("=" * 78)
        print("HARNESS CHECK — small/auto under the legacy normalizer")
        print(f"  prior baseline : {PRIOR_BASELINE_WER:.4f}")
        print(f"  this harness   : {got:.4f}  (delta {delta:.4f})")
        print(f"  -> {'REPRODUCED' if ok else 'MISMATCH — investigate before trusting anything below'}")
        print("=" * 78)

    # ── per-config table ──────────────────────────────────────
    print(f"\n{'config':18} {'raw WER':>8} {'fold WER':>9} {'raw CER':>8} "
          f"{'fold CER':>9} {'evalfWER':>9} {'RTF':>6}")
    print("-" * 78)
    for c in sorted(configs, key=lambda c: eval_micro(c)):
        raw, folded = c["metrics"]["raw"]["overall"], c["metrics"]["folded"]["overall"]
        label = f"{c['model_size']}/{c['decode_mode']}"
        print(f"{label:18} {fmt(raw['micro_wer'],8)} {fmt(folded['micro_wer'],9)} "
              f"{fmt(raw['micro_cer'],8)} {fmt(folded['micro_cer'],9)} "
              f"{fmt(eval_micro(c),9)} {fmt(c['real_time_factor'],6)}")

    # ── script-mismatch cost ──────────────────────────────────
    print(f"\nScript-mismatch cost (raw WER - folded WER), by script group:")
    groups = sorted({g for c in configs for g in c["metrics"]["raw"]["by_script_group"]})
    print(f"{'config':18} " + " ".join(f"{g:>13}" for g in groups))
    print("-" * 78)
    for c in sorted(configs, key=lambda c: eval_micro(c)):
        cells = []
        for g in groups:
            r = c["metrics"]["raw"]["by_script_group"].get(g)
            f_ = c["metrics"]["folded"]["by_script_group"].get(g)
            cells.append(f"{r['micro_wer'] - f_['micro_wer']:>13.3f}" if r and f_ else f"{'-':>13}")
        print(f"{c['model_size'] + '/' + c['decode_mode']:18} " + " ".join(cells))

    # ── error composition ─────────────────────────────────────
    # WER near 1.0 has two very different causes. Deletions dominating means
    # the model produced nothing; substitutions dominating means it produced
    # the wrong words. Only the second is a "recognition quality" problem that
    # a bigger model or fine-tuning straightforwardly improves.
    print(f"\nError composition (folded, whole corpus) — where the WER comes from:")
    print(f"{'config':18} {'hit':>7} {'sub':>7} {'del':>7} {'ins':>7}   reading")
    print("-" * 78)
    for c in sorted(configs, key=lambda c: eval_micro(c)):
        e = c["metrics"]["folded"]["overall"].get("errors")
        if not e:
            continue
        if e["del_rate"] > e["sub_rate"]:
            reading = f"silent — {e['del_rate']:.0%} of words never emitted"
        else:
            reading = f"mishearing — {e['sub_rate']:.0%} of words wrong"
        print(f"{c['model_size'] + '/' + c['decode_mode']:18} "
              f"{e['hit_rate']:>7.3f} {e['sub_rate']:>7.3f} {e['del_rate']:>7.3f} "
              f"{e['ins_rate']:>7.3f}   {reading}")

    # ── decision ──────────────────────────────────────────────
    # Judged on eval-split FOLDED WER: script-agnostic, so it measures
    # recognition quality rather than which script the corpus happens to use.
    # Fine-tuning can move the output script; it cannot conjure words the model
    # never recognised.
    best = min(configs, key=lambda c: eval_micro(c))
    by_mode = {}
    for c in configs:
        by_mode.setdefault(c["decode_mode"], []).append(eval_micro(c))
    best_mode = min(by_mode, key=lambda m: min(by_mode[m]))
    by_size = {}
    for c in configs:
        by_size.setdefault(c["model_size"], []).append(eval_micro(c))
    best_size = min(by_size, key=lambda s: min(by_size[s]))

    decisions = {
        "phase": 0,
        "decision_metric": "eval-split micro WER under script-folded normalization",
        "winning_decode_mode": best_mode,
        "winning_model_size": best_size,
        "winning_config": f"{best['model_size']}/{best['decode_mode']}",
        # inf would serialize as the non-standard JSON literal Infinity, so it
        # becomes null: "no eval-split rows" rather than "infinitely bad".
        "winning_eval_folded_wer": finite(eval_micro(best)),
        "winning_eval_raw_wer": finite(eval_micro(best, key="raw")),
        "harness_reproduction_check": repro_note,
        "eval_split_recordings": sorted(
            r for r, i in manifest.items() if i["split"] == "eval"),
        "configs": configs,
        "caveats": [
            "Scored on 52 min of audio; C1 §6.2 assumes ~1 h eval plus 15-20 h "
            "train. Treat every figure as provisional until the corpus grows.",
            "Folded WER is a floor, not a headline: script_fold is deliberately "
            "many-to-one and merges genuinely distinct words.",
            "Whisper segmentation does not align to gold utt_ids, so scoring is "
            "recording-level concatenation, which absorbs segmentation error "
            "into WER.",
        ],
    }

    sora_paths.ensure_dir(sora_paths.eval_dir())
    decisions_path = os.path.join(sora_paths.eval_dir(), "baseline_decisions.json")
    with open(decisions_path, "w", encoding="utf-8") as fh:
        json.dump(decisions, fh, indent=2, ensure_ascii=False)

    write_report(decisions, configs, groups)

    print(f"\nDECISION — decode mode: {best_mode}   model size: {best_size}")
    print(f"  best config {best['model_size']}/{best['decode_mode']}: "
          f"eval folded WER {fmt(eval_micro(best)).strip()}, eval raw WER "
          f"{fmt(eval_micro(best, key='raw')).strip()}")
    print(f"\nWrote {decisions_path}")
    print(f"Wrote {os.path.join(sora_paths.eval_dir(), 'baseline_report.md')}")
    return 0


def write_report(decisions, configs, groups):
    L = ["# Phase 0 — Baseline decisions\n\n",
         "Decoding-mode and model-size comparison for C1, measured on the real "
         "corpus. This file and its companion `baseline_decisions.json` are the "
         "single source of truth every later phase reads from "
         "(C1_FINAL_IMPLEMENTATION.md §8).\n\n",
         "## Decision\n\n",
         f"- **Decode mode:** `{decisions['winning_decode_mode']}`\n",
         f"- **Model size:** `{decisions['winning_model_size']}`\n",
         f"- **Best config:** `{decisions['winning_config']}` — eval-split folded "
         f"WER **{fmt(decisions['winning_eval_folded_wer'], 0).strip()}**, raw WER "
         f"**{fmt(decisions['winning_eval_raw_wer'], 0).strip()}**\n",
         f"- **Criterion:** {decisions['decision_metric']}. Script-folded scoring "
         "measures recognition quality independently of which script the corpus "
         "uses, because fine-tuning can change the output script but cannot "
         "recover words the model never heard.\n\n"]

    rep = decisions.get("harness_reproduction_check")
    if rep:
        verdict = "reproduced" if rep["reproduced"] else "**MISMATCH**"
        L += ["## Harness check\n\n",
              f"`small`/`auto` under the legacy normalizer scores "
              f"{rep['this_harness_legacy_micro_wer']:.4f} against the prior "
              f"baseline's {rep['prior_baseline_micro_wer']:.4f} "
              f"(delta {rep['delta']:.4f}) — {verdict}. Source: "
              f"`{rep['source']}`.\n\n"]

    L += ["## All configs\n\n",
          "| config | raw WER | folded WER | raw CER | folded CER | eval folded WER | RTF |\n",
          "|---|---|---|---|---|---|---|\n"]
    for c in sorted(configs, key=lambda c: eval_micro(c)):
        raw, fo = c["metrics"]["raw"]["overall"], c["metrics"]["folded"]["overall"]
        L.append(f"| `{c['model_size']}/{c['decode_mode']}` | {raw['micro_wer']:.4f} | "
                 f"{fo['micro_wer']:.4f} | {raw['micro_cer']:.4f} | "
                 f"{fo['micro_cer']:.4f} | {eval_micro(c):.4f} | "
                 f"{c['real_time_factor']} |\n")

    L += ["\n## Error composition (folded, whole corpus)\n\n",
          "Where the WER actually comes from. Deletions dominating means the "
          "model emitted nothing; substitutions dominating means it emitted the "
          "wrong words. The two call for opposite fixes.\n\n",
          "| config | hit | sub | del | ins |\n", "|---|---|---|---|---|\n"]
    for c in sorted(configs, key=lambda c: eval_micro(c)):
        e = c["metrics"]["folded"]["overall"].get("errors")
        if e:
            L.append(f"| `{c['model_size']}/{c['decode_mode']}` | {e['hit_rate']:.3f} | "
                     f"{e['sub_rate']:.3f} | {e['del_rate']:.3f} | {e['ins_rate']:.3f} |\n")

    L += ["\n## Script-mismatch cost (raw WER - folded WER)\n\n",
          "How many WER points each config loses purely to script mismatch "
          "rather than to misrecognition.\n\n",
          "| config | " + " | ".join(groups) + " |\n",
          "|---" * (len(groups) + 1) + "|\n"]
    for c in sorted(configs, key=lambda c: eval_micro(c)):
        cells = []
        for g in groups:
            r = c["metrics"]["raw"]["by_script_group"].get(g)
            f_ = c["metrics"]["folded"]["by_script_group"].get(g)
            cells.append(f"{r['micro_wer'] - f_['micro_wer']:.3f}" if r and f_ else "-")
        L.append(f"| `{c['model_size']}/{c['decode_mode']}` | " + " | ".join(cells) + " |\n")

    L += ["\n## Caveats\n\n"] + [f"- {c}\n" for c in decisions["caveats"]]
    L.append(f"\nEval split ({len(decisions['eval_split_recordings'])} recordings, "
             f"locked in `data/manifest.csv`): "
             f"{', '.join(decisions['eval_split_recordings'])}\n")

    with open(os.path.join(sora_paths.eval_dir(), "baseline_report.md"),
              "w", encoding="utf-8") as fh:
        fh.writelines(L)


if __name__ == "__main__":
    sys.exit(main())
