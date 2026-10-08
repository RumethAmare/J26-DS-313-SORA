#!/usr/bin/env python3
"""
build_dataset_release.py -- Task 8: assemble the shared C1 dataset.

Merges everything C1 produces into one release C2/C3/C4 can consume:

  c1/release/
    tokens/<rid>.jsonl     one record per GOLD token, enriched with C1's
                           predictions (language, switch, Tamil flag)
    asr/<rid>.tokens.jsonl the Task 2 ASR token stream, copied as-is
    MANIFEST.json          per-recording status and statistics, corpus
                           totals, provenance

The dataset card is docs/C1_DATASET_CARD.md.

WHY GOLD TOKENS ARE THE BASE, NOT THE ASR STREAM
------------------------------------------------
The plan merges "token stream + LID + switch + OTHER" into the release. The
ASR token stream covers about 17% of the gold token count (off-the-shelf
Whisper is at WER ~0.99 here), so a release built on it would be mostly
empty and mostly wrong. Gold tokens are the base; C1's predictions sit
alongside the human labels, and the ASR stream ships as its own layer.

WHY PREDICTIONS ARE OUT-OF-FOLD
-------------------------------
`lang_pred` for a usable recording comes from a fastText model trained on
the other folds only (same folds as train_lid_fasttext.py), so its
confidence means what it would on unseen data. Labelling with the shipped
model would score recordings it was trained on. Excluded recordings were
never in training, so the final model labels them directly.

EXCLUDED RECORDINGS ARE INCLUDED, FLAGGED
-----------------------------------------
They carry `status: "excluded"` and the reason from corpus.py. Their gold
`lang` is known to be wrong in bulk or in part; `lang_pred` is the useful
field there (see also predictions/pretag/).

Usage:
    python build_dataset_release.py
"""
import datetime
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audit_gold
import corpus
import lid_data
import lid_rules
import otherlang_flag as olf
import switch_detection_eval as sde
import train_lid_fasttext as trainer

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(C1_ROOT, "results")
ASR_STREAM_DIR = os.path.join(C1_ROOT, "predictions", "token_stream")
MODEL_PATH = os.path.join(C1_ROOT, "models", "lid_latin.ftz")
RELEASE_DIR = os.path.join(C1_ROOT, "release")

RELEASE_VERSION = "0.1.0"

# Same fastText settings as train_lid_fasttext.py's defaults.
FT_PARAMS = {"epoch": 50, "lr": 0.5, "dim": 50, "minn": 2, "maxn": 5,
             "wordNgrams": 1, "minCount": 1, "loss": "softmax",
             "seed": 13, "thread": 1, "bucket": 50000}


def git_head(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                         # noqa: BLE001
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- language predictions ----------------------------------------------------

def predict_token(model, token, script):
    """(label, confidence, method) via the hybrid routing in lid_hybrid.py."""
    if script == "latin":
        label, prob = trainer.predict_ft(model, token)
        return label, round(prob, 4), "fasttext_latin"
    return lid_rules.predict(token)


def out_of_fold_predictions(usable_by_rec):
    """{(rid, tok_index): (label, conf, method)} from models that never saw rid."""
    recordings = sorted(usable_by_rec)

    def en_ratio(rid):
        latin = [t for t in usable_by_rec[rid] if t["script"] == "latin"]
        return sum(1 for t in latin if t["lang"] == "EN") / len(latin) if latin else 0.0

    folds = lid_data.make_folds(recordings, k=5, key=en_ratio)
    seen = [r for f in folds for r in f]
    assert len(seen) == len(set(seen)) == len(recordings), "fold leakage!"
    workdir = tempfile.mkdtemp(prefix="c1_release_")
    out = {}
    for i, test_recs in enumerate(folds):
        train_latin = [t for r in recordings if r not in set(test_recs)
                       for t in usable_by_rec[r] if t["script"] == "latin"]
        model = trainer.train_fold(train_latin, FT_PARAMS, workdir)
        for rid in test_recs:
            for j, t in enumerate(usable_by_rec[rid]):
                out[(rid, j)] = predict_token(model, t["token"], t["script"])
        print(f"  fold {i}: {len(test_recs)} recordings labelled out-of-fold", flush=True)
    return out


# --- Tamil flag --------------------------------------------------------------

def build_tamil_flagger():
    """Task 5's detector at its reported operating point (99th percentile).

    The outlier rule is left off: Task 5 measured it at 1/16 recall for ~9%
    of the corpus flagged, so it would bury the real signal.
    """
    gt = olf.load_ground_truth()
    tamil_set = {e["token"] for e in gt["genuinely_tamil"]}
    si_words, en_words = olf.build_corpus_wordlists(tamil_set)
    lexicon = [w for w in olf.load_lexicon() if w not in set(si_words) | set(en_words)]
    with open(os.path.join(RESULTS_DIR, "c1_otherlang_eval.json"), encoding="utf-8") as fh:
        threshold = json.load(fh)["thresholds"]["tamil_margin"]
    return olf.TamilFlagger(lexicon, si_words, en_words), threshold


# --- per recording -----------------------------------------------------------

def read_tokens(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def audio_duration(rid):
    path = os.path.join(corpus.AUDIO_DIR, f"{rid}.wav")
    if not os.path.exists(path):
        return None
    import soundfile as sf
    return round(sf.info(path).duration, 2)


def main():
    print(f"Dataset root: {corpus.DATASET_ROOT}")
    files = corpus.gold_token_files()
    manifest_ids = audit_gold.read_manifest_ids()

    gold, audits = {}, {}
    for rid, path in files:
        toks = read_tokens(path)
        for t in toks:
            t["script"] = lid_data.token_script(t.get("token", ""))
        gold[rid] = toks
        audits[rid] = audit_gold.audit_file(rid, path, manifest_ids)[0]

    usable = {r: t for r, t in gold.items() if r not in corpus.EXCLUDED_RECORDINGS}
    print(f"{len(gold)} gold recordings, {len(usable)} usable, "
          f"{len(gold) - len(usable)} excluded")

    print("Language predictions (out-of-fold for usable recordings):")
    preds = out_of_fold_predictions(usable)
    import fasttext
    final_model = fasttext.load_model(MODEL_PATH)
    for rid, toks in gold.items():
        if rid in usable:
            continue
        for j, t in enumerate(toks):
            preds[(rid, j)] = predict_token(final_model, t["token"], t["script"])

    flagger, tamil_thr = build_tamil_flagger()

    tok_dir = os.path.join(RELEASE_DIR, "tokens")
    asr_dir = os.path.join(RELEASE_DIR, "asr")
    for d in (tok_dir, asr_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    rows = []
    totals = {"tokens": 0, "SI": 0, "EN": 0, "OTHER": 0, "switches": 0,
              "tamil_flags": 0, "audio_s": 0.0}
    for rid in sorted(gold):
        toks = gold[rid]
        excluded = rid in corpus.EXCLUDED_RECORDINGS
        derived = sde.derive_switches([t.get("lang") for t in toks],
                                      [t.get("utt_id") for t in toks])
        records = []
        for j, t in enumerate(toks):
            label, conf, method = preds[(rid, j)]
            flagged, reason, _ = flagger.flag(t["token"], tamil_thr, None)
            records.append({
                "recording": rid,
                "utt_id": t.get("utt_id"),
                "tok_id": t.get("tok_id"),
                "token": t.get("token"),
                "start": t.get("start"),
                "end": t.get("end"),
                "lang": t.get("lang"),
                "switch": derived[j],
                "switch_annotated": t.get("switch"),
                "script": t["script"],
                "numeric": t["script"] == "numeric",
                "lang_pred": label,
                "lang_pred_confidence": conf,
                "lang_pred_method": method,
                "lang_pred_source": "final_model" if excluded else "out_of_fold",
                "tamil_flag": flagged,
                "tamil_reason": reason if flagged else None,
            })
        with open(os.path.join(tok_dir, f"{rid}.jsonl"), "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        asr_src = os.path.join(ASR_STREAM_DIR, f"{rid}.tokens.jsonl")
        asr_tokens = None
        if os.path.exists(asr_src):
            shutil.copy2(asr_src, os.path.join(asr_dir, f"{rid}.tokens.jsonl"))
            asr_tokens = sum(1 for l in open(asr_src, encoding="utf-8") if l.strip())

        by_utt = {}
        for t in toks:
            by_utt.setdefault(t.get("utt_id"), []).append(t.get("lang"))
        cmis = [c for c in (sde.cmi(v) for v in by_utt.values()) if c is not None]
        dur = audio_duration(rid)
        a = audits[rid]
        lang_counts = {l: sum(1 for t in toks if t.get("lang") == l)
                       for l in ("SI", "EN", "OTHER")}
        agree = sum(1 for r in records if r["lang_pred"] == r["lang"])
        row = {
            "recording": rid,
            "status": "excluded" if excluded else ("usable" if dur else "usable_no_audio"),
            "exclusion_reason": corpus.EXCLUDED_RECORDINGS.get(rid),
            "n_tokens": len(toks),
            "n_utterances": len(by_utt),
            "lang_counts": lang_counts,
            "n_switches": sum(derived),
            "mean_cmi": round(statistics.mean(cmis), 2) if cmis else None,
            "pct_monolingual_utterances": (round(100 * sum(1 for c in cmis if c == 0)
                                                 / len(cmis), 1) if cmis else None),
            "lang_pred_agreement": round(agree / len(toks), 4) if toks else None,
            "tamil_flags": sum(1 for r in records if r["tamil_flag"]),
            "null_timestamps": a["null_timestamps"],
            "sinhala_script_convention": a["sinhala_script_convention"],
            "audit_issues": [i for i in a["issues"].split() if i != "EXCLUDED"],
            "has_audio": dur is not None,
            "audio_duration_s": dur,
            "asr_tokens": asr_tokens,
            "gold_source_file": os.path.basename(corpus.gold_tokens_path(rid)),
        }
        rows.append(row)
        if not excluded:
            totals["tokens"] += len(toks)
            for l in ("SI", "EN", "OTHER"):
                totals[l] += lang_counts[l]
            totals["switches"] += row["n_switches"]
            totals["tamil_flags"] += row["tamil_flags"]
            totals["audio_s"] += dur or 0.0

    usable_rows = [r for r in rows if r["status"] != "excluded"]
    manifest = {
        "name": "SORA C1 code-switching token dataset",
        "version": RELEASE_VERSION,
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset_card": "docs/C1_DATASET_CARD.md",
        "provenance": {
            "c1_repo_commit": git_head(C1_ROOT),
            "sora_dataset_commit": git_head(corpus.DATASET_ROOT),
            "lid_model": os.path.relpath(MODEL_PATH, C1_ROOT).replace(os.sep, "/"),
            "lid_model_sha256": sha256(MODEL_PATH),
            "lid_cv_eval": "results/c1_lid_eval.json",
            "tamil_threshold": tamil_thr,
            "tamil_eval": "results/c1_otherlang_eval.json",
            "asr_stream_config": "results/c1_token_stream_summary.json",
        },
        "totals_usable": {
            "recordings": len(usable_rows),
            "recordings_with_audio": sum(1 for r in usable_rows if r["has_audio"]),
            "tokens": totals["tokens"],
            "lang_counts": {l: totals[l] for l in ("SI", "EN", "OTHER")},
            "switches": totals["switches"],
            "tamil_flags": totals["tamil_flags"],
            "audio_hours": round(totals["audio_s"] / 3600, 2),
        },
        "totals_all": {
            "recordings": len(rows),
            "excluded": len(rows) - len(usable_rows),
            "tokens": sum(r["n_tokens"] for r in rows),
        },
        "recordings": rows,
    }
    with open(os.path.join(RELEASE_DIR, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    t = manifest["totals_usable"]
    print(f"\nRelease {RELEASE_VERSION} -> {RELEASE_DIR}")
    print(f"  usable: {t['recordings']} recordings ({t['recordings_with_audio']} with audio, "
          f"{t['audio_hours']} h), {t['tokens']:,} tokens, {t['switches']:,} switches, "
          f"{t['tamil_flags']} Tamil flags")
    print(f"  all:    {manifest['totals_all']['recordings']} recordings, "
          f"{manifest['totals_all']['excluded']} excluded, "
          f"{manifest['totals_all']['tokens']:,} tokens")
    print(f"  ASR stream files copied: {sum(1 for r in rows if r['asr_tokens'] is not None)}")


if __name__ == "__main__":
    main()
