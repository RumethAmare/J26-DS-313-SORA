#!/usr/bin/env python3
"""
train_lid_fasttext.py -- Task 3: fastText LID with recording-level 5-fold CV.

Trains "Method B" (fastText supervised) on the Latin-script tokens only and
compares it against "Method A" (the rule-based heuristic in lid_rules.py) and
against the hybrid that uses rules for deterministic scripts and fastText for
the ambiguous Latin ones.

WHY LATIN-SCRIPT ONLY
---------------------
Sinhala and Tamil script are resolved by Unicode range with no ambiguity, so a
classifier there would be learning to imitate a rule that is already exact.
Measured on the 5,545 usable gold tokens, the heuristic scores 97.7% on
Sinhala script but only 83.9% on Latin script -- 430 errors, and every one of
the remaining gains available to Task 3 lives there.

Those 430 break down as:
  273  EN -> SI   proper nouns and acronyms wordfreq does not know
                  (Nawaloka, Dehiwala, Sampath, Galle, NIC, 20th)
  142  SI -> EN   romanized Sinhala colliding with real English words
                  (me, da, ah, one, Aa) -- the 'mama' problem
   15  OTHER -> * see the OTHER note below

So 97% of the headroom is SI/EN confusion on Latin script, which is exactly
what a subword model should be able to attack.

THE `OTHER` CLASS DOES NOT SURVIVE CONTACT WITH THE DATA
--------------------------------------------------------
The plan assumes OTHER means Tamil. In this corpus it does not:

  * There are **zero Tamil-script tokens**. The Unicode Tamil rule -- credited
    in the plan with ~100% precision -- never fires on real data.
  * Of 78 OTHER tokens (after exclusions), 58 are purely numeric, i.e. the
    annotation drift Section D.3 already documented, not a language at all.
  * Of the remaining 20, most are identifiers and English ordinals
    (`8812304567V.`, `MASIT2024/0892.`, `LN2024NG00445`,
    `hasitha.94@gmail.com.`, `21st,`, `1st,`, `No.`, `1%`).
  * The genuinely Tamil items are written in **Sinhala** script
    (`පෙරිය` periya, `ඉල්ලයි` illai) or romanized (`puriyam.`, `sari`) --
    never in Tamil script.

That leaves 15 Latin-script OTHER examples spread across an incoherent set of
concepts. Three-way training is still run, because the deliverable schema has
three classes and silently dropping one would hide the problem -- but OTHER F1
is expected to be ~0 and that is a property of the labels, not the model.
Reporting it honestly is the useful output here, and it is what Task 5 needs
to know before it tries to build an OTHER detector.

Usage:
    python3 train_lid_fasttext.py
    python3 train_lid_fasttext.py --folds 5 --epoch 50
"""
import argparse
import json
import os
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fasttext

import lid_data
import lid_rules

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(C1_ROOT, "results")
MODEL_DIR = os.path.join(C1_ROOT, "models")


def train_fold(train_tokens, params, workdir):
    """Train one fastText model on the Latin-script tokens of the train split."""
    path = os.path.join(workdir, "train.txt")
    with open(path, "w", encoding="utf-8") as fh:
        for tok in train_tokens:
            fh.write(lid_data.to_fasttext_line(tok["token"], tok["lang"]) + "\n")
    return fasttext.train_supervised(input=path, verbose=0, **params)


def predict_ft(model, token):
    """Predict via the raw binding, bypassing fastText's numpy conversion.

    `model.predict()` ends in `np.array(probs, copy=False)`, which NumPy 2.x
    rejects outright ("Unable to avoid copy while creating an array as
    requested") -- fasttext-wheel predates the NumPy 2 copy-semantics change
    and this venv has NumPy 2.5. `model.f.predict` is the underlying C++ call
    and returns plain (prob, label) tuples, so going one level down avoids the
    problem without patching a third-party package or pinning NumPy back for
    the whole project.
    """
    line = lid_data.to_fasttext_line(token) + "\n"
    predictions = model.f.predict(line, 1, 0.0, "strict")
    if not predictions:
        return "SI", 0.0
    prob, label = predictions[0]
    return label.replace("__label__", ""), float(prob)


def hybrid_predict(model, token, script):
    """Rules for deterministic scripts, fastText for ambiguous Latin ones."""
    if script in ("sinhala", "tamil", "numeric"):
        return lid_rules.predict(token)[0]
    return predict_ft(model, token)[0]


def summarize(fold_metrics, labels=None):
    """mean +/- std across folds, per class."""
    labels = labels or lid_data.LABELS
    out = {}
    for lbl in labels:
        f1s = [m[lbl]["f1"] for m in fold_metrics]
        ps = [m[lbl]["precision"] for m in fold_metrics]
        rs = [m[lbl]["recall"] for m in fold_metrics]
        out[lbl] = {
            "f1_mean": round(statistics.mean(f1s), 4),
            "f1_std": round(statistics.stdev(f1s), 4) if len(f1s) > 1 else 0.0,
            "precision_mean": round(statistics.mean(ps), 4),
            "recall_mean": round(statistics.mean(rs), 4),
            "support_total": sum(m[lbl]["support"] for m in fold_metrics),
        }
    accs = [m["_accuracy"] for m in fold_metrics]
    macros = [m["_macro_f1"] for m in fold_metrics]
    out["_accuracy_mean"] = round(statistics.mean(accs), 4)
    out["_accuracy_std"] = round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0
    out["_macro_f1_mean"] = round(statistics.mean(macros), 4)
    out["_macro_f1_std"] = round(statistics.stdev(macros), 4) if len(macros) > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--minn", type=int, default=2)
    ap.add_argument("--maxn", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    # fastText defaults to 2,000,000 hash buckets, sized for web-scale corpora.
    # With 2,674 training tokens that produces a 400MB model (50MB quantized)
    # to hold a few thousand distinct character n-grams -- a nearly empty
    # table. Sweeping the bucket count over a 400x range moves CV accuracy by
    # less than half a standard deviation (0.906-0.918), so the capacity is
    # simply unused. 50k buckets scored highest of the values tried AND ships a
    # 1.3MB quantized model, a 37x reduction for no measured cost -- the
    # difference between fitting the offline deployment budget and not.
    ap.add_argument("--bucket", type=int, default=50000)
    a = ap.parse_args()

    params = {
        "epoch": a.epoch, "lr": a.lr, "dim": a.dim,
        "minn": a.minn, "maxn": a.maxn, "wordNgrams": 1,
        "minCount": 1, "loss": "softmax", "seed": a.seed,
        # thread=1: fastText's parallel SGD updates weights in a
        # nondeterministic order, so multi-thread runs differ by ~0.2pp between
        # invocations even with a fixed seed. At this corpus size training takes
        # about a second either way, so determinism is free and reported numbers
        # stay reproducible.
        "thread": 1,
        "bucket": a.bucket,
    }

    tokens = lid_data.load_gold_tokens()
    recordings = sorted({t["recording"] for t in tokens})
    print(f"Gold tokens after excluding {sorted(lid_data.EXCLUDED_RECORDINGS)}: "
          f"{len(tokens)} across {len(recordings)} recordings")

    by_rec = {r: [t for t in tokens if t["recording"] == r] for r in recordings}

    def en_ratio(rid):
        latin = [t for t in by_rec[rid] if t["script"] == "latin"]
        if not latin:
            return 0.0
        return sum(1 for t in latin if t["lang"] == "EN") / len(latin)

    folds = lid_data.make_folds(recordings, k=a.folds, key=en_ratio)
    print(f"\n{a.folds}-fold split, folded by RECORDING (no token leakage):")
    for i, f in enumerate(folds):
        n_lat = sum(1 for r in f for t in by_rec[r] if t["script"] == "latin")
        print(f"  fold {i}: {len(f)} recordings, {n_lat:4} latin tokens  "
              f"{[r.replace('J26DS313_', '') for r in f]}")

    # Leakage guard: a recording must appear in exactly one fold.
    seen = [r for f in folds for r in f]
    assert len(seen) == len(set(seen)) == len(recordings), "fold leakage!"
    print("  leakage check: every recording appears in exactly one fold — OK")

    heur_all, hyb_all, ft_latin, heur_latin = [], [], [], []
    workdir = tempfile.mkdtemp(prefix="c1_lid_")

    for i, test_recs in enumerate(folds):
        train_recs = [r for r in recordings if r not in set(test_recs)]
        train_tokens = [t for r in train_recs for t in by_rec[r]]
        test_tokens = [t for r in test_recs for t in by_rec[r]]

        train_latin = [t for t in train_tokens if t["script"] == "latin"]
        test_latin = [t for t in test_tokens if t["script"] == "latin"]

        model = train_fold(train_latin, params, workdir)

        heur_all.append(lid_data.per_class_metrics(
            [(t["lang"], lid_rules.predict(t["token"])[0]) for t in test_tokens]))
        hyb_all.append(lid_data.per_class_metrics(
            [(t["lang"], hybrid_predict(model, t["token"], t["script"]))
             for t in test_tokens]))
        ft_latin.append(lid_data.per_class_metrics(
            [(t["lang"], predict_ft(model, t["token"])[0]) for t in test_latin]))
        heur_latin.append(lid_data.per_class_metrics(
            [(t["lang"], lid_rules.predict(t["token"])[0]) for t in test_latin]))

        print(f"  fold {i}: train_latin={len(train_latin):4} test_latin={len(test_latin):4}"
              f"  heuristic_acc={heur_all[-1]['_accuracy']:.4f}"
              f"  hybrid_acc={hyb_all[-1]['_accuracy']:.4f}")

    results = {
        "method_a_heuristic_all_tokens": summarize(heur_all),
        "method_b_fasttext_latin_only": summarize(ft_latin),
        "heuristic_latin_only": summarize(heur_latin),
        "hybrid_all_tokens": summarize(hyb_all),
    }

    def show(name, s, note=""):
        print(f"\n{name}{note}")
        print(f"  accuracy  {s['_accuracy_mean']:.4f} +/- {s['_accuracy_std']:.4f}"
              f"    macro-F1 {s['_macro_f1_mean']:.4f} +/- {s['_macro_f1_std']:.4f}")
        print(f"  {'class':6} {'F1 mean':>9} {'F1 std':>8} {'P':>8} {'R':>8} {'support':>8}")
        for lbl in lid_data.LABELS:
            m = s[lbl]
            print(f"  {lbl:6} {m['f1_mean']:>9.4f} {m['f1_std']:>8.4f} "
                  f"{m['precision_mean']:>8.4f} {m['recall_mean']:>8.4f} "
                  f"{m['support_total']:>8}")

    print("\n" + "=" * 72)
    print(f"{a.folds}-FOLD CROSS-VALIDATION RESULTS (mean +/- std over folds)")
    print("=" * 72)
    show("METHOD A — rule-based heuristic", results["method_a_heuristic_all_tokens"],
         "  [all tokens]")
    show("HYBRID — rules + fastText on Latin", results["hybrid_all_tokens"],
         "  [all tokens]")
    print("\n--- Latin-script subset only (where the two methods actually differ) ---")
    show("  heuristic", results["heuristic_latin_only"])
    show("  METHOD B — fastText", results["method_b_fasttext_latin_only"])

    # Paired per-fold comparison. Comparing mean +/- std across folds
    # understates the evidence here: the two methods are evaluated on the
    # *same* folds, and fold difficulty varies a lot (per-fold heuristic
    # accuracy ranges 0.84-0.92). What matters is whether the hybrid wins on
    # each fold individually, not whether the error bars overlap.
    per_fold = [(h["_accuracy"], y["_accuracy"]) for h, y in zip(heur_all, hyb_all)]
    deltas = [y - h for h, y in per_fold]
    wins = sum(1 for d in deltas if d > 0)
    print("\nPaired per-fold comparison (same folds, both methods):")
    print(f"  {'fold':>4} {'heuristic':>10} {'hybrid':>9} {'delta':>9}")
    for i, ((h, y), d) in enumerate(zip(per_fold, deltas)):
        print(f"  {i:>4} {h:>10.4f} {y:>9.4f} {d:>+9.4f}")
    print(f"  hybrid wins on {wins}/{len(deltas)} folds; "
          f"mean delta {statistics.mean(deltas):+.4f}, "
          f"min {min(deltas):+.4f}, max {max(deltas):+.4f}")

    results["paired_fold_comparison"] = {
        "heuristic_accuracy_per_fold": [round(h, 4) for h, _ in per_fold],
        "hybrid_accuracy_per_fold": [round(y, 4) for _, y in per_fold],
        "delta_per_fold": [round(d, 4) for d in deltas],
        "folds_won_by_hybrid": wins,
        "mean_delta": round(statistics.mean(deltas), 4),
        "min_delta": round(min(deltas), 4),
        "max_delta": round(max(deltas), 4),
    }

    # Final model on all usable Latin tokens, for lid_hybrid.py to load.
    os.makedirs(MODEL_DIR, exist_ok=True)
    all_latin = [t for t in tokens if t["script"] == "latin"]
    final = train_fold(all_latin, params, workdir)
    model_path = os.path.join(MODEL_DIR, "lid_latin.ftz")
    final.quantize(input=os.path.join(workdir, "train.txt"), retrain=False)
    final.save_model(model_path)
    size_kb = os.path.getsize(model_path) / 1024
    print(f"\nFinal model trained on all {len(all_latin)} Latin tokens")
    print(f"  quantized -> {model_path} ({size_kb:.0f} KB)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "c1_lid_eval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "note": "Folded by recording, not token, so no conversation appears in "
                    "both train and test. fastText sees Latin-script tokens only; "
                    "Sinhala/Tamil script and numerics are handled by deterministic "
                    "rules. OTHER F1 near zero is a property of the labels (15 "
                    "Latin-script examples, incoherent membership), not the model.",
            "n_tokens": len(tokens),
            "n_recordings": len(recordings),
            "excluded_recordings": sorted(lid_data.EXCLUDED_RECORDINGS),
            "folds": [[r for r in f] for f in folds],
            "fasttext_params": params,
            "results": results,
            "final_model": {"path": model_path, "size_kb": round(size_kb, 1),
                            "n_train_tokens": len(all_latin)},
        }, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
