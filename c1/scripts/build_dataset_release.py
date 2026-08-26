#!/usr/bin/env python3
"""
build_dataset_release.py -- Task 8: merged C1 dataset release.

RE-SCOPED FROM THE PLAN: the release has TWO LAYERS, not one
------------------------------------------------------------
The plan says to merge "Task 2's token stream + Task 3's LID labels + Task 4's
switch markers + Task 5's OTHER flags into one canonical
processed/c1_dataset/<rid>.jsonl".

Merging onto Task 2's token stream would produce a near-useless artifact.
That stream is off-the-shelf ASR output measured at ~0.98-1.00 WER with 22%
token coverage (Task 1, Task 2): 1,280 tokens against gold's 5,805, most of
them hallucinated. Enriching hallucinated tokens with language labels does not
make them a dataset.

So the release ships two clearly separated layers:

  gold/  -- PRIMARY. The 5,805 human-annotated tokens, enriched with everything
            C1 produced: derived switch points, out-of-fold hybrid-LID
            predictions, Tamil candidate flags, and per-recording quality
            flags. This is the artifact downstream work should consume, and
            the only one on which the reported metrics were measured.

  asr/   -- SECONDARY. The Task 2 end-to-end token stream, carried through so
            the pipeline output is inspectable and reproducible. Marked
            unreliable in every record. Not a substitute for gold.

Predictions on the gold layer are OUT-OF-FOLD: each token is labelled by a
fastText model trained without that token's recording. The shipped model
(models/lid_latin.ftz) is trained on everything, so using it here would leak
its training data into the release and inflate any metric computed from it.

Excluded recordings (R0017, R0008) are INCLUDED but flagged, not silently
dropped. A consumer filtering on `quality.usable_for_training` gets the clean
5,545-token set; a consumer auditing the corpus can still see what was
excluded and why.

Usage:
    python3 build_dataset_release.py
"""
import json
import os
import statistics
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lid_data
import lid_rules
import otherlang_flag as otf
import switch_detection_eval as switching

DATASET_ROOT = "/mnt/F/SLIIT/Research/SORA_Dataset"
C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(DATASET_ROOT, "annotations", "c1")
TOKEN_STREAM_DIR = os.path.join(C1_ROOT, "predictions", "token_stream")
RELEASE_DIR = os.path.join(C1_ROOT, "release", "c1_dataset")
RESULTS_DIR = os.path.join(C1_ROOT, "results")

EXCLUSION_REASONS = {
    "J26DS313_R0017": "every token tagged EN including obvious romanized "
                      "Sinhala (gold-data bug, Section D.2)",
    "J26DS313_R0008": "token file structurally malformed - junk in timestamps "
                      "(Section D.1)",
}


def load_all_gold():
    """Every gold token INCLUDING the excluded recordings, which are flagged."""
    out = []
    for fname in sorted(os.listdir(GOLD_DIR)):
        if not fname.endswith(".tokens.jsonl"):
            continue
        rid = fname[: -len(".tokens.jsonl")]
        with open(os.path.join(GOLD_DIR, fname), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    tok = json.loads(line)
                    tok["recording"] = rid
                    tok["script"] = lid_data.token_script(tok["token"])
                    out.append(tok)
    return out


def build_tamil_flagger():
    """Task 5's detector, with the thresholds it was calibrated at."""
    gt = otf.load_ground_truth()
    lexicon = otf.load_lexicon()
    tamil_set = {e["token"] for e in gt["genuinely_tamil"]}
    si_words, en_words = otf.build_corpus_wordlists(tamil_set)
    corpus_words = set(si_words) | set(en_words)
    lexicon = [w for w in lexicon if w not in corpus_words]
    flagger = otf.TamilFlagger(lexicon, si_words, en_words)

    tokens = lid_data.load_gold_tokens()
    negatives = [t["token"] for t in tokens
                 if t["token"] not in tamil_set
                 and otf.is_candidate(otf.normalize_token(t["token"]))]
    margins = sorted(flagger.score(tok)["tamil_margin"] for tok in negatives)
    threshold = margins[int(0.99 * len(margins))]
    return flagger, threshold, {e["token"] for e in gt["genuinely_tamil"]}


def main():
    tokens = load_all_gold()
    recordings = sorted({t["recording"] for t in tokens})
    usable = [r for r in recordings if r not in EXCLUSION_REASONS]
    print(f"Gold: {len(tokens)} tokens across {len(recordings)} recordings "
          f"({len(usable)} usable, {len(EXCLUSION_REASONS)} flagged)\n")

    by_rec = {r: [t for t in tokens if t["recording"] == r] for r in recordings}

    # --- out-of-fold LID predictions on the usable recordings ---------------
    print("Building out-of-fold LID predictions (model never sees the "
          "recording it labels)...")
    clean_by_rec = {r: [t for t in by_rec[r]] for r in usable}
    oof = switching.build_out_of_fold_predictions(clean_by_rec, usable)

    # Excluded recordings get predictions from a model trained on all usable
    # data. They are never scored, so there is no leakage concern - but the
    # provenance differs and the record says so.
    import train_lid_fasttext as trainer
    params = {"epoch": 50, "lr": 0.5, "dim": 50, "minn": 2, "maxn": 5,
              "wordNgrams": 1, "minCount": 1, "loss": "softmax",
              "thread": 1, "seed": 13, "bucket": 50000}
    workdir = tempfile.mkdtemp(prefix="c1_release_")
    full_model = trainer.train_fold(
        [t for r in usable for t in by_rec[r] if t["script"] == "latin"],
        params, workdir)

    flagger, tamil_threshold, tamil_gt = build_tamil_flagger()
    print(f"Tamil flagger calibrated at margin >= {tamil_threshold:+.4f}\n")

    os.makedirs(os.path.join(RELEASE_DIR, "gold"), exist_ok=True)
    os.makedirs(os.path.join(RELEASE_DIR, "asr"), exist_ok=True)

    corpus = {"n_tokens": 0, "gold_lang": Counter(), "pred_lang": Counter(),
              "script": Counter(), "n_gold_switches": 0, "n_pred_switches": 0,
              "n_tamil_flagged": 0, "agreement": 0}
    per_recording, all_cmi = [], []

    for rid in recordings:
        toks = sorted(by_rec[rid], key=lambda t: (t["utt_id"], t["tok_id"]))
        is_usable = rid not in EXCLUSION_REASONS
        utt_ids = [t["utt_id"] for t in toks]

        # Predicted labels
        preds = []
        for t in toks:
            if is_usable:
                preds.append(oof[(rid, t["token"])])
            elif t["script"] == "latin":
                preds.append(trainer.predict_ft(full_model, t["token"])[0])
            else:
                preds.append(lid_rules.predict(t["token"])[0])

        # Switches derived identically for gold and prediction (Task 4). The
        # annotated `switch` field is NOT used - it is unreliable at utterance
        # boundaries (docs/SWITCH_FIELD_AUDIT.md).
        gold_sw = switching.derive_switches([t["lang"] for t in toks], utt_ids)
        pred_sw = switching.derive_switches(preds, utt_ids)

        records = []
        for t, pred, gsw, psw in zip(toks, preds, gold_sw, pred_sw):
            _, conf, method = lid_rules.predict(t["token"])
            flagged, reason, sig = flagger.flag(t["token"], tamil_threshold)
            records.append({
                "recording": rid,
                "utt_id": t["utt_id"],
                "tok_id": t["tok_id"],
                "token": t["token"],
                "start": t["start"],
                "end": t["end"],
                "script": t["script"],
                "gold": {
                    "lang": t["lang"],
                    "switch": gsw,
                    "switch_annotated": bool(t.get("switch")),
                },
                "predicted": {
                    "lang": pred,
                    "switch": psw,
                    "lang_confidence": conf,
                    "lang_method": method,
                    "provenance": "out_of_fold" if is_usable else "full_model",
                },
                "tamil_flag": {
                    "flagged": flagged,
                    "reason": reason if flagged else None,
                    "margin": sig["tamil_margin"],
                    "is_ground_truth_tamil": t["token"] in tamil_gt,
                },
                "quality": {
                    "usable_for_training": is_usable,
                    "exclusion_reason": EXCLUSION_REASONS.get(rid),
                },
            })

        out_path = os.path.join(RELEASE_DIR, "gold", f"{rid}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # per-utterance CMI
        by_utt = {}
        for t in toks:
            by_utt.setdefault(t["utt_id"], []).append(t)
        cmis = [c for c in (switching.cmi([x["lang"] for x in ts])
                            for ts in by_utt.values()) if c is not None]

        agree = sum(1 for t, p in zip(toks, preds) if t["lang"] == p)
        rec_row = {
            "recording": rid,
            "usable_for_training": is_usable,
            "exclusion_reason": EXCLUSION_REASONS.get(rid),
            "n_tokens": len(toks),
            "n_utterances": len(by_utt),
            "gold_lang_counts": dict(Counter(t["lang"] for t in toks)),
            "n_gold_switches": sum(gold_sw),
            "n_pred_switches": sum(pred_sw),
            "lid_agreement": round(agree / len(toks), 4) if toks else None,
            "mean_cmi": round(statistics.mean(cmis), 2) if cmis else None,
            "n_tamil_flagged": sum(1 for r in records if r["tamil_flag"]["flagged"]),
        }
        per_recording.append(rec_row)

        if is_usable:
            corpus["n_tokens"] += len(toks)
            corpus["gold_lang"].update(t["lang"] for t in toks)
            corpus["pred_lang"].update(preds)
            corpus["script"].update(t["script"] for t in toks)
            corpus["n_gold_switches"] += sum(gold_sw)
            corpus["n_pred_switches"] += sum(pred_sw)
            corpus["n_tamil_flagged"] += rec_row["n_tamil_flagged"]
            corpus["agreement"] += agree
            all_cmi.extend(cmis)

        flag = "" if is_usable else "  [EXCLUDED]"
        print(f"  {rid}  {len(toks):4} tokens  {sum(gold_sw):3} switches  "
              f"agree={rec_row['lid_agreement']:.3f}  "
              f"cmi={rec_row['mean_cmi']}{flag}")

    # --- ASR layer ----------------------------------------------------------
    n_asr = 0
    for fname in sorted(os.listdir(TOKEN_STREAM_DIR)) if os.path.isdir(TOKEN_STREAM_DIR) else []:
        if not fname.endswith(".tokens.jsonl"):
            continue
        rid = fname[: -len(".tokens.jsonl")]
        src = os.path.join(TOKEN_STREAM_DIR, fname)
        dst = os.path.join(RELEASE_DIR, "asr", f"{rid}.jsonl")
        with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["source"] = "asr_baseline"
                rec["reliability"] = ("UNRELIABLE: off-the-shelf faster-whisper "
                                      "small, ~0.98-1.00 WER, 22% token coverage. "
                                      "Not a substitute for the gold layer.")
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_asr += 1

    manifest = {
        "name": "J26-DS-313 C1 code-switched token dataset",
        "layers": {
            "gold": {
                "path": "gold/<recording>.jsonl",
                "description": "Human-annotated tokens enriched with C1 outputs. "
                               "PRIMARY layer; all reported metrics measured here.",
                "n_recordings": len(recordings),
                "n_tokens_total": sum(r["n_tokens"] for r in per_recording),
                "n_tokens_usable": corpus["n_tokens"],
            },
            "asr": {
                "path": "asr/<recording>.jsonl",
                "description": "End-to-end pipeline output. SECONDARY, unreliable, "
                               "retained for reproducibility only.",
                "n_tokens": n_asr,
            },
        },
        "excluded_recordings": EXCLUSION_REASONS,
        "corpus_statistics_usable_only": {
            "n_tokens": corpus["n_tokens"],
            "gold_language_counts": dict(corpus["gold_lang"]),
            "gold_language_share": {
                k: round(v / corpus["n_tokens"], 4)
                for k, v in corpus["gold_lang"].items()},
            "predicted_language_counts": dict(corpus["pred_lang"]),
            "script_counts": dict(corpus["script"]),
            "class_imbalance_ratio_SI_to_OTHER": round(
                corpus["gold_lang"]["SI"] / max(1, corpus["gold_lang"]["OTHER"]), 1),
            "n_gold_switches": corpus["n_gold_switches"],
            "n_predicted_switches": corpus["n_pred_switches"],
            "switch_rate": round(corpus["n_gold_switches"] / corpus["n_tokens"], 4),
            "lid_agreement_out_of_fold": round(
                corpus["agreement"] / corpus["n_tokens"], 4),
            "n_tamil_flagged": corpus["n_tamil_flagged"],
            "cmi": {
                "n_utterances": len(all_cmi),
                "mean": round(statistics.mean(all_cmi), 2),
                "median": round(statistics.median(all_cmi), 2),
                "std": round(statistics.stdev(all_cmi), 2),
                "pct_monolingual": round(
                    100 * sum(1 for c in all_cmi if c == 0) / len(all_cmi), 1),
            },
        },
        "provenance": {
            "switch_field": "DERIVED from gold `lang`, not read from the annotated "
                            "`switch` field, which is a coin flip at utterance "
                            "boundaries (docs/SWITCH_FIELD_AUDIT.md).",
            "predicted_lang": "Hybrid LID: Unicode rules for Sinhala/Tamil script "
                              "and numerics, fastText for Latin script. Out-of-fold "
                              "on usable recordings.",
            "tamil_flag": "Task 5 detector; see docs/C1_DATASET_CARD.md for why "
                          "gold OTHER is not a usable Tamil label.",
        },
        "per_recording": per_recording,
    }
    man_path = os.path.join(RELEASE_DIR, "MANIFEST.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"\nGold layer : {manifest['layers']['gold']['n_tokens_total']} tokens "
          f"({corpus['n_tokens']} usable) -> {RELEASE_DIR}/gold/")
    print(f"ASR layer  : {n_asr} tokens -> {RELEASE_DIR}/asr/")
    print(f"Language mix (usable): {dict(corpus['gold_lang'])}")
    print(f"SI:OTHER imbalance    : "
          f"{manifest['corpus_statistics_usable_only']['class_imbalance_ratio_SI_to_OTHER']}:1")
    print(f"Out-of-fold LID agreement: "
          f"{manifest['corpus_statistics_usable_only']['lid_agreement_out_of_fold']:.4f}")
    print(f"Wrote {man_path}")

    with open(os.path.join(RESULTS_DIR, "c1_release_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
