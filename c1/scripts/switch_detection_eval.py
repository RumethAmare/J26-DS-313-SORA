#!/usr/bin/env python3
"""
switch_detection_eval.py -- Task 4: switch-point F1 + code-mixing index.

RE-SCOPED FROM THE PLAN, for two measured reasons.

1. DO NOT SCORE AGAINST THE ANNOTATED `switch` FIELD.
   The plan says switch detection "matches the gold `switch` field already in
   the schema -- audit whether it's populated correctly". The audit
   (docs/SWITCH_FIELD_AUDIT.md) says it is not. At utterance boundaries where
   the language genuinely changes, gold marks switch=true 102 times and
   switch=false 100 times -- a coin flip -- plus 27 tokens marked as switches
   with no language change at all. Scoring against that field measures
   annotator inconsistency, not model quality.

   So both sides are RE-DERIVED from the `lang` field, which is the primary
   annotation and far more reliable, using one explicit convention applied
   identically to gold and predictions.

2. THE END-TO-END NUMBER IS ASR-BOUNDED AND MUST BE REPORTED SEPARATELY.
   The token-stream analysis (results/c1_token_stream_notes.md) showed the
   ASR emits 1,280 tokens against gold's 5,818 and only 86 switch points
   against gold's 2,012 -- a recall ceiling near 4% before switch detection
   does anything at all. Quoting one end-to-end switch F1 would attribute an
   ASR failure to the switch-detection work. The primary metric is therefore
   computed on GOLD tokens, isolating the LID + switch-derivation logic, with
   the end-to-end figure reported alongside and labelled for what it is.

Outputs results/c1_switch_eval.json and results/c1_switch_report.md.

Usage:
    python3 switch_detection_eval.py
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lid_data
import lid_rules

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(C1_ROOT, "results")
TOKEN_STREAM_DIR = os.path.join(C1_ROOT, "predictions", "token_stream")


# --- switch derivation -------------------------------------------------------

def derive_switches(labels, utt_ids, cross_utterance=False):
    """Positions where the language changes.

    Returns a list of booleans, one per token. A switch is a property of the
    label *sequence*, so it is derived identically for gold and predictions --
    that is what makes the comparison meaningful.

    `cross_utterance=False` (default, the plan's rule): the first token of each
    utterance is never a switch. Well-defined and needs no speaker information.

    `cross_utterance=True`: carry the previous token's language across
    utterance boundaries. Linguistically closer to what code-switching means,
    but conflates a speaker change with a code-switch, because tokens.jsonl
    carries no speaker field. Reported as a sensitivity check only.
    """
    out = []
    prev_lang = None
    prev_utt = None
    for lang, utt in zip(labels, utt_ids):
        if prev_utt is not None and utt != prev_utt and not cross_utterance:
            prev_lang = None          # reset at utterance boundary
        out.append(prev_lang is not None and lang != prev_lang)
        prev_lang = lang
        prev_utt = utt
    return out


def switch_prf(gold_switches, pred_switches):
    """Precision/recall/F1 over switch POSITIONS, not token labels.

    Token accuracy is the wrong metric for this task: a sequence can be 90%
    correct token-wise and still miss every switch, because switches are rare
    and sit exactly where the labels are hardest.
    """
    tp = sum(1 for g, p in zip(gold_switches, pred_switches) if g and p)
    fp = sum(1 for g, p in zip(gold_switches, pred_switches) if not g and p)
    fn = sum(1 for g, p in zip(gold_switches, pred_switches) if g and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_gold_switches": tp + fn,
        "n_pred_switches": tp + fp,
    }


# --- code-mixing index -------------------------------------------------------

def cmi(labels, exclude_numeric_tokens=None):
    """Code-Mixing Index for one utterance (Das & Gambäck 2014).

        CMI = 100 * (N - max_L) / N

    where N is the number of language-dependent tokens and max_L the count of
    the dominant language. 0 means monolingual; higher means more evenly mixed.

    `exclude_numeric_tokens` optionally drops purely-numeric tokens. That
    matters here: Section D.3 found gold's numeric tagging is inconsistent
    (192 EN, 61 OTHER, 1 SI for the same kinds of token), so numerics inject
    arbitrary label mass into the mix. Both variants are reported.
    """
    if exclude_numeric_tokens is not None:
        labels = [l for l, tok in zip(labels, exclude_numeric_tokens)
                  if lid_data.token_script(tok) != "numeric"]
    n = len(labels)
    if n == 0:
        return None
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return round(100.0 * (n - max(counts.values())) / n, 2)


# --- end-to-end alignment ----------------------------------------------------

def align_by_time(gold_tokens, pred_tokens):
    """Match predicted tokens to gold tokens by maximum temporal overlap.

    Needed only for the end-to-end figure: predicted and gold token sequences
    have different lengths and contents, so switch positions do not correspond
    index-to-index the way they do on gold tokens. Each gold token takes the
    predicted token it overlaps most in time; gold tokens with no overlapping
    prediction are unmatched, which is the common case at 22% ASR coverage.
    """
    matches = []
    j = 0
    for g in gold_tokens:
        best, best_ov = None, 0.0
        # predictions are time-ordered, so only scan the local window
        while j < len(pred_tokens) and pred_tokens[j]["end"] < g["start"]:
            j += 1
        k = j
        while k < len(pred_tokens) and pred_tokens[k]["start"] <= g["end"]:
            ov = (min(g["end"], pred_tokens[k]["end"])
                  - max(g["start"], pred_tokens[k]["start"]))
            if ov > best_ov:
                best, best_ov = pred_tokens[k], ov
            k += 1
        matches.append(best)
    return matches


def build_out_of_fold_predictions(by_rec, recordings, k=5):
    """Label every token with a fastText model that never saw its recording.

    Mirrors the fold construction in train_lid_fasttext.py so the switch
    numbers and the LID numbers describe the same system under the same
    protocol. Non-Latin tokens go to the deterministic rules, exactly as
    lid_hybrid.HybridLID routes them at inference time.
    """
    import tempfile

    import fasttext

    import train_lid_fasttext as trainer

    def en_ratio(rid):
        latin = [t for t in by_rec[rid] if t["script"] == "latin"]
        if not latin:
            return 0.0
        return sum(1 for t in latin if t["lang"] == "EN") / len(latin)

    folds = lid_data.make_folds(recordings, k=k, key=en_ratio)
    params = {"epoch": 50, "lr": 0.5, "dim": 50, "minn": 2, "maxn": 5,
              "wordNgrams": 1, "minCount": 1, "loss": "softmax",
              "thread": 1, "seed": 13, "bucket": 50000}  # thread=1 for determinism

    workdir = tempfile.mkdtemp(prefix="c1_switch_oof_")
    out = {}
    for test_recs in folds:
        train_recs = [r for r in recordings if r not in set(test_recs)]
        train_latin = [t for r in train_recs for t in by_rec[r]
                       if t["script"] == "latin"]
        model = trainer.train_fold(train_latin, params, workdir)
        for rid in test_recs:
            for tok in by_rec[rid]:
                if tok["script"] == "latin":
                    out[(rid, tok["token"])] = trainer.predict_ft(model, tok["token"])[0]
                else:
                    out[(rid, tok["token"])] = lid_rules.predict(tok["token"])[0]
    return out


def main():
    tokens = lid_data.load_gold_tokens()
    recordings = sorted({t["recording"] for t in tokens})
    print(f"Gold tokens: {len(tokens)} across {len(recordings)} recordings "
          f"(excluded {sorted(lid_data.EXCLUDED_RECORDINGS)})\n")

    by_rec = {r: [t for t in tokens if t["recording"] == r] for r in recordings}

    # --- primary: switch F1 on gold tokens, per LID method -------------------
    # The hybrid's predictions MUST be out-of-fold. The shipped model
    # (models/lid_latin.ftz) is trained on every Latin token in the corpus, so
    # asking it to label those same tokens would score it on its own training
    # data and inflate switch F1. Each token is instead labelled by a model
    # trained without its recording, matching train_lid_fasttext.py's protocol.
    methods = {"heuristic": lambda tok, rid: lid_rules.predict(tok)[0]}
    try:
        oof = build_out_of_fold_predictions(by_rec, recordings)
        methods["hybrid"] = lambda tok, rid, _o=oof: _o[(rid, tok)]
        print("Hybrid predictions are out-of-fold (model never saw the "
              "recording it labels).\n")
    except Exception as exc:                                  # noqa: BLE001
        print(f"NOTE: hybrid LID unavailable ({exc}); heuristic only.\n")

    results = {}
    for conv_name, cross in (("within_utterance", False), ("cross_utterance", True)):
        for meth_name, predict in methods.items():
            agg_gold, agg_pred = [], []
            per_rec = []
            for rid in recordings:
                toks = by_rec[rid]
                utt_ids = [t["utt_id"] for t in toks]
                gold_sw = derive_switches([t["lang"] for t in toks], utt_ids, cross)
                pred_sw = derive_switches([predict(t["token"], rid) for t in toks],
                                          utt_ids, cross)
                agg_gold.extend(gold_sw)
                agg_pred.extend(pred_sw)
                per_rec.append({"recording": rid, **switch_prf(gold_sw, pred_sw)})
            results[f"{meth_name}__{conv_name}"] = {
                "overall": switch_prf(agg_gold, agg_pred),
                "per_recording": per_rec,
            }

    # Token-level LID accuracy under the same protocol, so the amplification
    # from LID errors to switch errors can be quantified rather than asserted.
    lid_accuracy = {}
    for meth_name, predict in methods.items():
        correct = sum(1 for rid in recordings for t in by_rec[rid]
                      if predict(t["token"], rid) == t["lang"])
        lid_accuracy[meth_name] = round(correct / len(tokens), 4)

    amplification = None
    if len(lid_accuracy) == 2:
        d_lid = lid_accuracy["hybrid"] - lid_accuracy["heuristic"]
        d_sw = (results["hybrid__within_utterance"]["overall"]["f1"]
                - results["heuristic__within_utterance"]["overall"]["f1"])
        amplification = {
            "lid_accuracy": lid_accuracy,
            "lid_accuracy_delta": round(d_lid, 4),
            "switch_f1_delta": round(d_sw, 4),
            "amplification_ratio": round(d_sw / d_lid, 2) if d_lid else None,
        }
        print(f"\nLID token accuracy -> switch F1 (same out-of-fold predictions)")
        print(f"  {'method':12} {'LID acc':>9} {'switch F1':>11} {'gap':>8}")
        for m in ("heuristic", "hybrid"):
            acc = lid_accuracy[m]
            f1 = results[f"{m}__within_utterance"]["overall"]["f1"]
            print(f"  {m:12} {acc:>9.4f} {f1:>11.4f} {acc - f1:>+8.4f}")
        print(f"  improving LID by {d_lid:+.4f} moved switch F1 by {d_sw:+.4f} "
              f"({amplification['amplification_ratio']}x)")

    print("SWITCH-POINT F1 on GOLD tokens (gold switches re-derived from gold `lang`)")
    print(f"{'method':12} {'convention':18} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'gold_sw':>8} {'pred_sw':>8}")
    for key, res in results.items():
        meth, conv = key.split("__")
        o = res["overall"]
        print(f"{meth:12} {conv:18} {o['precision']:>7.4f} {o['recall']:>7.4f} "
              f"{o['f1']:>7.4f} {o['n_gold_switches']:>8} {o['n_pred_switches']:>8}")

    # --- CMI: corpus characterization ---------------------------------------
    utt_cmis, utt_cmis_nonum = [], []
    per_rec_cmi = []
    for rid in recordings:
        toks = by_rec[rid]
        by_utt = {}
        for t in toks:
            by_utt.setdefault(t["utt_id"], []).append(t)
        vals, vals_nn = [], []
        for utt, ts in by_utt.items():
            c = cmi([t["lang"] for t in ts])
            if c is not None:
                vals.append(c)
            c2 = cmi([t["lang"] for t in ts], [t["token"] for t in ts])
            if c2 is not None:
                vals_nn.append(c2)
        utt_cmis.extend(vals)
        utt_cmis_nonum.extend(vals_nn)
        per_rec_cmi.append({
            "recording": rid,
            "n_utterances": len(by_utt),
            "mean_cmi": round(statistics.mean(vals), 2) if vals else None,
            "mean_cmi_excl_numeric": round(statistics.mean(vals_nn), 2) if vals_nn else None,
            "pct_monolingual_utterances": round(
                100 * sum(1 for v in vals if v == 0) / len(vals), 1) if vals else None,
        })

    cmi_summary = {
        "n_utterances": len(utt_cmis),
        "mean_cmi": round(statistics.mean(utt_cmis), 2),
        "median_cmi": round(statistics.median(utt_cmis), 2),
        "std_cmi": round(statistics.stdev(utt_cmis), 2),
        "mean_cmi_excl_numeric": round(statistics.mean(utt_cmis_nonum), 2),
        "pct_monolingual_utterances": round(
            100 * sum(1 for v in utt_cmis if v == 0) / len(utt_cmis), 1),
        "pct_utterances_cmi_over_25": round(
            100 * sum(1 for v in utt_cmis if v > 25) / len(utt_cmis), 1),
    }
    print(f"\nCODE-MIXING INDEX (gold, {cmi_summary['n_utterances']} utterances)")
    print(f"  mean CMI                 {cmi_summary['mean_cmi']:.2f}  "
          f"(median {cmi_summary['median_cmi']:.2f}, sd {cmi_summary['std_cmi']:.2f})")
    print(f"  mean CMI excl. numerics  {cmi_summary['mean_cmi_excl_numeric']:.2f}")
    print(f"  monolingual utterances   {cmi_summary['pct_monolingual_utterances']:.1f}%")
    print(f"  utterances with CMI>25   {cmi_summary['pct_utterances_cmi_over_25']:.1f}%")

    # --- end-to-end: ASR-bounded --------------------------------------------
    e2e = None
    if os.path.isdir(TOKEN_STREAM_DIR):
        agg_gold, agg_pred = [], []
        n_matched = n_gold = 0
        for rid in recordings:
            path = os.path.join(TOKEN_STREAM_DIR, f"{rid}.tokens.jsonl")
            if not os.path.exists(path):
                continue
            preds = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            preds.sort(key=lambda t: t["start"])
            toks = sorted(by_rec[rid], key=lambda t: t["start"])
            matched = align_by_time(toks, preds)
            n_gold += len(toks)
            n_matched += sum(1 for m in matched if m is not None)
            # Unmatched gold tokens get a sentinel label so they can never
            # accidentally register as a correct switch.
            pred_labels = [m["lang"] if m else "__NONE__" for m in matched]
            utt_ids = [t["utt_id"] for t in toks]
            agg_gold.extend(derive_switches([t["lang"] for t in toks], utt_ids))
            agg_pred.extend(derive_switches(pred_labels, utt_ids))
        e2e = switch_prf(agg_gold, agg_pred)
        e2e["gold_tokens"] = n_gold
        e2e["gold_tokens_matched_to_asr"] = n_matched
        e2e["pct_gold_tokens_matched"] = round(100 * n_matched / n_gold, 1) if n_gold else 0
        print(f"\nEND-TO-END (ASR -> LID -> switch), time-aligned to gold")
        print(f"  gold tokens matched to an ASR token: {n_matched}/{n_gold} "
              f"({e2e['pct_gold_tokens_matched']:.1f}%)")
        print(f"  P={e2e['precision']:.4f} R={e2e['recall']:.4f} F1={e2e['f1']:.4f}")
        print("  ^ ASR-bounded: see results/c1_token_stream_notes.md")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "c1_switch_eval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "note": "Gold switches are RE-DERIVED from gold `lang`, not read from "
                    "the annotated `switch` field, which is unreliable at utterance "
                    "boundaries (see docs/SWITCH_FIELD_AUDIT.md). Primary metric is "
                    "on gold tokens; the end-to-end figure is ASR-bounded.",
            "excluded_recordings": sorted(lid_data.EXCLUDED_RECORDINGS),
            "switch_f1_gold_tokens": results,
            "cmi": {"summary": cmi_summary, "per_recording": per_rec_cmi},
            "end_to_end_asr_bounded": e2e,
            "lid_to_switch_amplification": amplification,
        }, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")

    write_report(results, cmi_summary, per_rec_cmi, e2e, amplification)


def write_report(results, cmi_summary, per_rec_cmi, e2e, amplification=None):
    L = []
    L.append("# Task 4 — Switch-point detection and code-mixing index\n\n")
    L.append("Produced by `scripts/switch_detection_eval.py`.\n\n")

    L.append("## Two deviations from the plan, both forced by the data\n\n")
    L.append("**Gold switches are re-derived from gold `lang`, not read from the\n")
    L.append("annotated `switch` field.** The plan expected that field to serve as the\n")
    L.append("target. It cannot: at utterance boundaries where the language actually\n")
    L.append("changes, gold marks `switch=true` 102 times and `switch=false` 100 times,\n")
    L.append("plus 27 tokens flagged as switches with no language change at all. Scoring\n")
    L.append("against it would measure annotator inconsistency. Full analysis in\n")
    L.append("[SWITCH_FIELD_AUDIT.md](../docs/SWITCH_FIELD_AUDIT.md).\n\n")
    L.append("**The primary metric is computed on gold tokens.** End-to-end switch F1 is\n")
    L.append("bounded by the ASR before switch detection contributes anything — the\n")
    L.append("baseline emits 1,280 tokens against gold's 5,818. It is reported below,\n")
    L.append("separately and labelled, so an ASR failure is not attributed to this task.\n\n")

    L.append("## Switch-point F1 on gold tokens\n\n")
    L.append("Positions, not token labels: a sequence can be 90% correct token-wise and\n")
    L.append("still miss every switch, because switches are rare and sit exactly where\n")
    L.append("the labels are hardest.\n\n")
    L.append("| LID method | convention | P | R | F1 | gold switches | predicted |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for key, res in results.items():
        meth, conv = key.split("__")
        o = res["overall"]
        L.append(f"| {meth} | {conv.replace('_', '-')} | {o['precision']:.4f} "
                 f"| {o['recall']:.4f} | {o['f1']:.4f} | {o['n_gold_switches']} "
                 f"| {o['n_pred_switches']} |\n")
    L.append("\nThe `cross-utterance` rows are a sensitivity check. That convention is\n")
    L.append("linguistically closer to what code-switching means, but it conflates a\n")
    L.append("speaker change with a code-switch, because `tokens.jsonl` carries no\n")
    L.append("speaker field. Resolving it properly needs a join against the C3 RTTM.\n\n")

    if amplification:
        a = amplification
        L.append("### Switch detection amplifies LID errors\n\n")
        L.append("Switch detection trains nothing of its own — it is a deterministic\n")
        L.append("function of the language sequence — so all of its error comes from LID.\n")
        L.append("But it does not inherit that error one-for-one. A single mislabelled\n")
        L.append("token in the middle of a monolingual run creates **two** spurious\n")
        L.append("switches, one entering the error and one leaving it.\n\n")
        L.append("The LID accuracies below are pooled over all tokens, so they differ\n")
        L.append("slightly from `c1_lid_report.md`'s 0.8994 / 0.9318, which average the\n")
        L.append("five per-fold accuracies. Same predictions, micro vs macro averaging.\n\n")
        L.append("| method | LID token accuracy | switch F1 | gap |\n")
        L.append("|---|---|---|---|\n")
        for m in ("heuristic", "hybrid"):
            acc = a["lid_accuracy"][m]
            f1 = results[f"{m}__within_utterance"]["overall"]["f1"]
            L.append(f"| {m} | {acc:.4f} | {f1:.4f} | {acc - f1:+.4f} |\n")
        L.append(f"\nSwitch F1 sits roughly 6–11 points below LID accuracy for both\n")
        L.append(f"methods, and the amplification cuts both ways: improving LID accuracy\n")
        L.append(f"by **{a['lid_accuracy_delta']:+.4f}** moved switch F1 by\n")
        L.append(f"**{a['switch_f1_delta']:+.4f}** — about **{a['amplification_ratio']}×**\n")
        L.append("the token-level gain.\n\n")
        L.append("That ratio is the practical argument for Task 3's classifier. Judged on\n")
        L.append("token accuracy alone the hybrid looks like a modest three-point\n")
        L.append("improvement; judged on the switch points that actually define\n")
        L.append("code-switching, it is worth several times that. Switch F1, not token\n")
        L.append("accuracy, is the honest headline metric for this sub-objective.\n\n")

    L.append("## Code-mixing index (corpus characterization)\n\n")
    L.append("Das & Gambäck (2014): `CMI = 100 × (N − max_L) / N` per utterance, where\n")
    L.append("N is language-dependent tokens and `max_L` the dominant language's count.\n")
    L.append("0 is monolingual; higher is more evenly mixed.\n\n")
    L.append(f"- utterances measured: **{cmi_summary['n_utterances']}**\n")
    L.append(f"- mean CMI: **{cmi_summary['mean_cmi']:.2f}** "
             f"(median {cmi_summary['median_cmi']:.2f}, sd {cmi_summary['std_cmi']:.2f})\n")
    L.append(f"- mean CMI excluding numeric tokens: "
             f"**{cmi_summary['mean_cmi_excl_numeric']:.2f}**\n")
    L.append(f"- monolingual utterances (CMI = 0): "
             f"**{cmi_summary['pct_monolingual_utterances']:.1f}%**\n")
    L.append(f"- utterances with CMI > 25: "
             f"**{cmi_summary['pct_utterances_cmi_over_25']:.1f}%**\n\n")
    L.append("Both variants are given because Section D.3 found gold's numeric tagging\n")
    L.append("inconsistent (192 EN / 61 OTHER / 1 SI for comparable tokens), so numerics\n")
    L.append("inject arbitrary label mass into the mix.\n\n")

    L.append("### Most and least code-mixed recordings\n\n")
    ranked = [r for r in per_rec_cmi if r["mean_cmi"] is not None]
    ranked.sort(key=lambda r: r["mean_cmi"], reverse=True)
    L.append("| recording | utterances | mean CMI | excl. numeric | monolingual utts |\n")
    L.append("|---|---|---|---|---|\n")
    for r in ranked[:5] + [None] + ranked[-5:]:
        if r is None:
            L.append("| … | | | | |\n")
            continue
        L.append(f"| {r['recording']} | {r['n_utterances']} | {r['mean_cmi']:.2f} "
                 f"| {r['mean_cmi_excl_numeric']:.2f} "
                 f"| {r['pct_monolingual_utterances']:.1f}% |\n")
    L.append("\n")

    if e2e:
        L.append("## End-to-end (ASR → LID → switch) — ASR-bounded\n\n")
        L.append("Predicted tokens are aligned to gold by maximum temporal overlap, since\n")
        L.append("the two sequences differ in length and content.\n\n")
        L.append(f"- gold tokens matched to any ASR token: "
                 f"**{e2e['gold_tokens_matched_to_asr']}/{e2e['gold_tokens']} "
                 f"({e2e['pct_gold_tokens_matched']:.1f}%)**\n")
        L.append(f"- precision {e2e['precision']:.4f}, recall {e2e['recall']:.4f}, "
                 f"**F1 {e2e['f1']:.4f}**\n\n")
        L.append("This number describes the ASR, not the switch-detection logic. Roughly\n")
        L.append("four fifths of gold tokens have no corresponding ASR token at all, so\n")
        L.append("most gold switches are unreachable regardless of how good the LID is.\n")
        L.append("Read the gold-token table above for the quality of this task's own\n")
        L.append("contribution, and `c1_report.md` for why the ASR is the bottleneck.\n\n")

    L.append("## What this says about the sub-objective\n\n")
    L.append("Switch detection needs no model of its own — it is a deterministic function\n")
    L.append("of the language sequence, so its accuracy is entirely inherited from LID.\n")
    L.append("That makes it a useful diagnostic rather than a component to optimise: it\n")
    L.append("amplifies LID errors, because one mislabelled token in a monolingual run\n")
    L.append("creates *two* spurious switches. That amplification is why switch F1 sits\n")
    L.append("well below LID token accuracy and is the honest metric to quote for\n")
    L.append("code-switching capability.\n")

    out = os.path.join(RESULTS_DIR, "c1_switch_report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(L)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
