"""
Step 5: final scores on the held-out TEST split, for the report and PP1.

    Translation   base NLLB (zero-shot) vs fine-tuned NLLB, en and si
                  chrF++ (headline) + BLEU, overall and romanized / Sinhala-script
    Summary       base Qwen vs QLoRA direct vs QLoRA translate-first
                  ROUGE-L, action-item P/R/F1 (intent, intent+owner)

Needs a GPU (Colab T4) and data/test.jsonl. Run it once per final model set:
looking at test scores while still tuning turns test into a second dev set.

    python src/evaluate.py
    python src/evaluate.py --google    # + Google Translate baseline (sends test
                                       #   transcripts to Google: check consent first)

Writes eval/test_results.json and eval/c2_report.md (scores only, no text).
Predictions go to predictions/test/ (gitignored: they contain PII).
"""
import argparse
import json
from pathlib import Path

from metrics import score, score_actions, score_summaries
from normalize import fix_sinhala, prepare
from run_c2 import (BASE_NLLB, DRIVE_MODELS, LANG_ENGLISH, LANG_SINGLISH, summarise,
                    translate)
from train_translation import LANG, SPEAKER, SRC_LANG, is_romanized

C2_ROOT = Path(__file__).resolve().parents[1]


def by_script(hyps, refs, srcs):
    out = {"all": score(hyps, refs)}
    for name, rom in (("romanized", True), ("sinhala_script", False)):
        idx = [i for i, s in enumerate(srcs) if is_romanized(s) == rom]
        if idx:
            out[name] = score([hyps[i] for i in idx], [refs[i] for i in idx])
    return out


def google_translate(texts, target):
    from deep_translator import GoogleTranslator
    gt = GoogleTranslator(source="auto", target=target)
    return [gt.translate(t) or "" for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(DRIVE_MODELS))
    ap.add_argument("--en-model", default="nllb600m_en_frac100")
    ap.add_argument("--si-model", default="nllb600m_si_frac100")
    ap.add_argument("--google", action="store_true", help="add a Google Translate baseline")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()
    models = Path(args.models)
    pred_dir = C2_ROOT / "predictions" / "test"
    pred_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in open(C2_ROOT / "data" / "test.jsonl", encoding="utf-8")]
    utts = [u for r in records for u in r["utterances"]]
    srcs = [prepare(SPEAKER.sub("", u["src"])) for u in utts]
    print(f"test: {len(records)} recordings, {len(utts)} utterances")
    results = {"split": "test", "recordings": [r["rid"] for r in records], "translation": {}}
    saved = {}

    # ---- Task A: translation -------------------------------------------------
    for tgt in ("en", "si"):
        keep = [i for i, u in enumerate(utts) if u[f"clean_{tgt}"]]
        src = [srcs[i] for i in keep]
        refs = [utts[i][f"clean_{tgt}"] for i in keep]
        systems = {
            "nllb600m_zeroshot": lambda: translate(BASE_NLLB, src, SRC_LANG, LANG[tgt]),
            "nllb600m_finetuned": lambda: translate(
                models / (args.en_model if tgt == "en" else args.si_model), src, SRC_LANG, LANG[tgt]),
        }
        if args.google:
            systems["google_translate"] = lambda: google_translate(src, tgt)
        for name, run in systems.items():
            hyps = [fix_sinhala(h) for h in run()]
            saved[(tgt, name)] = dict(zip([utts[i]["utt_id"] for i in keep], hyps))
            results["translation"].setdefault(tgt, {})[name] = by_script(hyps, refs, src)
            print(tgt, name, results["translation"][tgt][name]["all"])
            with open(pred_dir / f"translation_{tgt}_{name}.jsonl", "w", encoding="utf-8") as f:
                for i, h in zip(keep, hyps):
                    f.write(json.dumps({"utt_id": utts[i]["utt_id"], "src": srcs[i], "hyp": h,
                                        "ref": utts[i][f"clean_{tgt}"]}, ensure_ascii=False) + "\n")

    # ---- Task B: summary + action items ---------------------------------------
    if not args.no_summary:
        recs = [r for r in records if r["summary_en"]]
        en_hyp = saved[("en", "nllb600m_finetuned")]
        direct_in = [r["transcript"] for r in recs]
        # Translate-first uses OUR translation of the test transcript, as in real use.
        cascade_in = ["\n".join(f"{u['speaker']}: {en_hyp.get(u['utt_id'], '')}" for u in r["utterances"])
                      for r in recs]
        systems = {
            "qwen3b_zeroshot": (None, direct_in, LANG_SINGLISH),
            "qwen3b_direct_qlora": (models / "qwen3b_direct_qlora" / "adapter", direct_in, LANG_SINGLISH),
            "qwen3b_cascade_qlora": (models / "qwen3b_cascade_qlora" / "adapter", cascade_in, LANG_ENGLISH),
        }
        results["summary"] = {}
        for name, (adapter, inputs, lang) in systems.items():
            out = summarise(adapter, inputs, lang)
            results["summary"][name] = {
                "summary": score_summaries([o[1] for o in out], [r["summary_en"] for r in recs]),
                "actions": score_actions([o[2] for o in out], [r["action_items"] for r in recs])}
            print(name, results["summary"][name])
            with open(pred_dir / f"summary_{name}.jsonl", "w", encoding="utf-8") as f:
                for r, (raw, summ, items) in zip(recs, out):
                    f.write(json.dumps({"rid": r["rid"], "raw": raw, "summary": summ, "action_items": items,
                                        "ref_summary": r["summary_en"],
                                        "ref_action_items": r["action_items"]}, ensure_ascii=False) + "\n")

    (C2_ROOT / "eval").mkdir(exist_ok=True)
    (C2_ROOT / "eval" / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (C2_ROOT / "eval" / "c2_report.md").write_text(report(results), encoding="utf-8")
    print("\n" + report(results))


def report(res):
    lines = [f"# C2 test results\n",
             f"Held-out test split: {len(res['recordings'])} recordings "
             f"({', '.join(res['recordings'])}).\n",
             "## Translation (chrF++ / BLEU)\n",
             "| Target | System | All | Romanized | Sinhala script |",
             "|---|---|---|---|---|"]
    fmt = lambda s: f"{s['chrf++']} / {s['bleu']}" if s else "-"
    for tgt, systems in res["translation"].items():
        for name, s in systems.items():
            lines.append(f"| {tgt} | {name} | {fmt(s['all'])} | {fmt(s.get('romanized'))} "
                         f"| {fmt(s.get('sinhala_script'))} |")
    if "summary" in res:
        lines += ["\n## Summary and action items\n",
                  "| System | ROUGE-L | Actions F1 (intent) | Actions F1 (intent+owner) |",
                  "|---|---|---|---|"]
        for name, s in res["summary"].items():
            lines.append(f"| {name} | {s['summary']['rougeL']} | {s['actions']['intent']['f1']} "
                         f"| {s['actions']['intent+owner']['f1']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
