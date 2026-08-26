#!/usr/bin/env python3
"""
otherlang_flag.py -- Task 5: flag Tamil (`OTHER`) tokens.

RE-SCOPED FROM THE PLAN. The plan's Task 5 says:

    "Unicode rule (already in lid_baseline.py) handles native-script Tamil at
     ~100% precision - keep as-is. Real gap is romanized Tamil [...] flag a
     Latin-script token as candidate-OTHER if its likelihood under both [SI and
     EN] is below a percentile threshold."

Three of those premises do not hold against the corpus:

1. THERE IS NO NATIVE-SCRIPT TAMIL. Zero tokens in the Tamil Unicode block.
   The rule being "kept as-is" never fires, so its ~100% precision is vacuous.

2. ROMANIZED TAMIL IS THE MINORITY CASE. Of 9 genuinely-Tamil tokens, only 3
   are romanized. FIVE are Tamil written in SINHALA SCRIPT -- one entire
   sentence in R0015, "ඉදු ඔරුව පෙරිය ප්‍රචන ඉල්ලයි"
   = idhu oru periya pirachanai illai = "this is not a big problem". The plan
   does not consider this case at all, and it is the dominant one. Worse, the
   `unicode_sinhala` rule currently labels every one of those tokens SI at 0.99
   confidence -- confidently wrong.

3. GOLD `OTHER` IS NOT A USABLE EVALUATION TARGET. Of 78 OTHER tokens, 58 are
   purely numeric and 11 are identifiers or English ordinals (Section D.3
   annotation drift). Scoring "precision/recall against gold OTHER tokens" as
   the plan proposes would measure a detector against a class that is 88%
   not-Tamil. Ground truth is therefore curated in
   data/tamil_ground_truth.json and evaluated against that.

THE DESIGN THAT FOLLOWS FROM THIS
---------------------------------
Romanize first, detect second. script_normalize.normalize_scripted() (built in
Task 1 for WER scoring) maps Sinhala script onto the same Latin skeleton that
romanized text already lives in: පෙරිය -> periya, සරි -> sari. So ONE Tamil
detector, operating in romanized space, covers both the Latin-script and the
Sinhala-script cases instead of needing two separate models.

Three signals are combined:
  * script anomaly -- a token in neither Sinhala, Latin nor Tamil block is
    flagged outright (this catches the Kannada/Malayalam-encoded `ಪುರಿಯമ്.`)
  * gazetteer     -- exact match against a romanized Tamil lexicon
  * char n-gram   -- discriminative Tamil-vs-(SI+EN) log-likelihood ratio

The plan's outlier variant (low likelihood under BOTH SI and EN) is also
implemented and reported side by side, since it needs no Tamil lexicon at all.

EVALUATION USES RAW COUNTS, NOT F1. With 9 positives, an F1 moves ~0.1 per
token and communicates false precision.

Usage:
    python3 otherlang_flag.py
"""
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lid_data
import script_normalize as sn

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(C1_ROOT, "data")
RESULTS_DIR = os.path.join(C1_ROOT, "results")
LEXICON_PATH = os.path.join(DATA_DIR, "tamil_romanized_lexicon.txt")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "tamil_ground_truth.json")

NGRAM_ORDER = 3


# --- character n-gram language model ----------------------------------------

def ngrams(word, n=NGRAM_ORDER):
    padded = "^" + word + "$"
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


def train_lm(words):
    counts = Counter()
    for w in words:
        counts.update(ngrams(w))
    return counts, sum(counts.values()), len(counts)


def lm_score(word, model):
    """Mean log-probability per n-gram, add-1 smoothed.

    Normalising per n-gram matters: without it, longer words score lower purely
    for being long, and token length would drive the decision instead of
    orthography.
    """
    counts, total, vocab = model
    grams = ngrams(word)
    if not grams:
        return -99.0
    return sum(math.log((counts[g] + 1) / (total + vocab)) for g in grams) / len(grams)


# --- loading ----------------------------------------------------------------

def load_lexicon():
    words = []
    with open(LEXICON_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)
    # Normalise into the same skeleton space as everything else.
    return sorted({sn.normalize_scripted(w) for w in words} - {""})


def load_ground_truth():
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def normalize_token(token):
    return sn.normalize_scripted(token)


def script_of(token):
    """Extends lid_data.token_script with an 'other-script' category.

    lid_data collapses anything non-Sinhala/non-Tamil/non-numeric into 'latin',
    which silently swallows the Kannada/Malayalam-encoded token. Task 5 needs
    that case visible, because a token in an unexpected script is a strong
    OTHER signal on its own.
    """
    base = lid_data.token_script(token)
    if base != "latin":
        return base
    stripped = token.strip(".,:;!?\"'()")
    for ch in stripped:
        cp = ord(ch)
        if cp < 128 or ch.isspace() or not ch.isalpha():
            continue
        # A letter outside ASCII that is not Sinhala or Tamil -> foreign script.
        if not (0x0D80 <= cp <= 0x0DFF or 0x0B80 <= cp <= 0x0BFF):
            return "other_script"
    return "latin"


# --- detector ---------------------------------------------------------------

MIN_TOKEN_LEN = 3


def is_candidate(norm):
    """Only alphabetic tokens of a plausible word length are scoreable.

    Without this the detector drowns in digits and single letters. A token like
    "9" has n-grams that appear in NO language model, so every model falls back
    to its add-1 floor -- and because the Tamil model is trained on the fewest
    words it has the smallest (total + vocab), which makes that floor the
    HIGHEST. The result is that every unseen short token scores as maximally
    Tamil. That is a smoothing artifact, not evidence, and it accounted for
    ~380 of 426 false positives before this filter existed.
    """
    return len(norm) >= MIN_TOKEN_LEN and norm.isalpha()


class TamilFlagger:
    def __init__(self, tamil_words, si_words, en_words):
        self.gazetteer = set(tamil_words)
        self.lm_ta = train_lm(tamil_words)
        self.lm_si = train_lm(si_words)
        self.lm_en = train_lm(en_words)

    def _seen_fraction(self, word):
        """Share of the token's n-grams the Tamil model has actually observed."""
        counts, _, _ = self.lm_ta
        grams = ngrams(word)
        if not grams:
            return 0.0
        return sum(1 for g in grams if counts[g] > 0) / len(grams)

    def score(self, token):
        """Return a dict of signals for one token."""
        script = script_of(token)
        norm = normalize_token(token)
        s_ta = lm_score(norm, self.lm_ta)
        s_si = lm_score(norm, self.lm_si)
        s_en = lm_score(norm, self.lm_en)
        return {
            "token": token,
            "normalized": norm,
            "script": script,
            "candidate": is_candidate(norm),
            "foreign_script": script == "other_script",
            "in_gazetteer": norm in self.gazetteer,
            # discriminative: Tamil vs the better of SI/EN
            "tamil_margin": round(s_ta - max(s_si, s_en), 4),
            # the plan's outlier signal: unlikely under BOTH SI and EN
            "native_max": round(max(s_si, s_en), 4),
            # positive Tamil evidence, independent of the smoothing floor
            "tamil_seen_frac": round(self._seen_fraction(norm), 4),
        }

    def flag(self, token, margin_threshold, outlier_threshold=None,
             min_seen_frac=0.5):
        sig = self.score(token)
        # A foreign script is evidence on its own and needs no word model.
        if sig["foreign_script"]:
            return True, "foreign_script", sig
        if not sig["candidate"]:
            return False, "not_candidate", sig
        if sig["in_gazetteer"]:
            return True, "gazetteer", sig
        # Require POSITIVE Tamil evidence: the margin alone can be high simply
        # because a token is unlike Sinhala and English, which is a statement
        # about what it is not.
        if (sig["tamil_margin"] >= margin_threshold
                and sig["tamil_seen_frac"] >= min_seen_frac):
            return True, "ngram_margin", sig
        if outlier_threshold is not None and sig["native_max"] <= outlier_threshold:
            return True, "outlier", sig
        return False, "none", sig


def build_corpus_wordlists(exclude_tokens):
    """Romanized SI and EN word lists from the corpus, minus the target tokens."""
    tokens = lid_data.load_gold_tokens()
    si, en = set(), set()
    for t in tokens:
        if t["token"] in exclude_tokens:
            continue
        norm = normalize_token(t["token"])
        if not norm or not norm.isalpha() or len(norm) < 2:
            continue
        if t["lang"] == "SI":
            si.add(norm)
        elif t["lang"] == "EN":
            en.add(norm)
    return sorted(si), sorted(en)


def main():
    gt = load_ground_truth()
    tamil_tokens = [e["token"] for e in gt["genuinely_tamil"]]
    tamil_set = set(tamil_tokens)
    lexicon = load_lexicon()

    print("=" * 74)
    print("Task 5 — Tamil (OTHER) flagging")
    print("=" * 74)
    print(f"Ground truth: {len(tamil_tokens)} genuinely-Tamil tokens "
          f"(of {gt['counts']['other_tokens_total']} labelled OTHER)")
    print(f"  by script: {gt['counts']['tamil_by_script']}")
    print(f"Romanized Tamil lexicon: {len(lexicon)} normalized entries\n")

    si_words, en_words = build_corpus_wordlists(tamil_set)
    print(f"Negative training data from corpus (target tokens excluded): "
          f"{len(si_words)} SI, {len(en_words)} EN")

    # A gazetteer entry that is also a common Sinhala or English word in this
    # corpus cannot serve as a high-precision signal: Tamil "namma" (our)
    # normalises to the same skeleton as Sinhala "nama" (name). Drop the
    # collisions rather than accept guaranteed false positives.
    corpus_words = set(si_words) | set(en_words)
    collisions = sorted(set(lexicon) & corpus_words)
    lexicon = [w for w in lexicon if w not in corpus_words]
    print(f"Gazetteer: dropped {len(collisions)} entries colliding with corpus "
          f"SI/EN words -> {len(lexicon)} usable")
    if collisions:
        print(f"  collisions: {collisions[:12]}")
    print()

    all_tokens = lid_data.load_gold_tokens()

    # --- calibrate thresholds on NON-target tokens only ---------------------
    # Thresholds must not be tuned on the 9 positives, or the evaluation is
    # circular. They are set from the negative distribution: allow roughly 1%
    # of ordinary corpus tokens through as false positives.
    calib = TamilFlagger(lexicon, si_words, en_words)
    # Calibrate over CANDIDATES only. Including digits and single letters would
    # set the threshold from tokens the detector never scores, which is what
    # pushed it to an unreachable +1.58 before the candidate filter existed.
    negatives = [t["token"] for t in all_tokens
                 if t["token"] not in tamil_set
                 and is_candidate(normalize_token(t["token"]))]
    neg_margins = sorted(calib.score(tok)["tamil_margin"] for tok in negatives)
    neg_natives = sorted(calib.score(tok)["native_max"] for tok in negatives)
    margin_threshold = neg_margins[int(0.99 * len(neg_margins))]
    outlier_threshold = neg_natives[int(0.01 * len(neg_natives))]
    print(f"Thresholds calibrated on {len(negatives)} non-target CANDIDATE tokens "
          f"(99th pct margin, 1st pct native):")
    print(f"  tamil_margin >= {margin_threshold:+.4f}")
    print(f"  native_max   <= {outlier_threshold:+.4f}  (plan's outlier rule)\n")

    # --- leave-one-out evaluation on the 9 positives ------------------------
    # Each target token is scored by a detector whose Tamil lexicon and n-gram
    # model have had that word (and its normalized form) removed. Without this
    # the gazetteer trivially recognises every positive, since common Tamil
    # words are exactly what a hand-written lexicon contains.
    print("LEAVE-ONE-OUT detection on the 9 genuinely-Tamil tokens")
    print(f"  {'token':22} {'script':12} {'norm':12} {'margin':>8} {'flagged by':14}")
    loo_hits = []
    for entry in gt["genuinely_tamil"]:
        tok = entry["token"]
        norm = normalize_token(tok)
        held_lex = [w for w in lexicon if w != norm]
        det = TamilFlagger(held_lex, si_words, en_words)
        flagged, reason, sig = det.flag(tok, margin_threshold, outlier_threshold)
        loo_hits.append((tok, flagged, reason, sig))
        mark = reason if flagged else "-- MISSED --"
        print(f"  {tok:22} {sig['script']:12} {sig['normalized']:12} "
              f"{sig['tamil_margin']:>+8.3f} {mark:14}")
    n_found = sum(1 for _, f, _, _ in loo_hits if f)
    print(f"\n  recovered {n_found}/{len(loo_hits)} Tamil tokens (leave-one-out)\n")

    # --- false positives over the whole corpus ------------------------------
    full = TamilFlagger(lexicon, si_words, en_words)
    fps = []
    for t in all_tokens:
        if t["token"] in tamil_set:
            continue
        flagged, reason, sig = full.flag(t["token"], margin_threshold, outlier_threshold)
        if flagged:
            fps.append((t["recording"], t["token"], t["lang"], reason, sig))
    print(f"FALSE POSITIVES across {len(all_tokens)} corpus tokens: {len(fps)} "
          f"({100 * len(fps) / len(all_tokens):.2f}%)")
    by_reason = Counter(r for _, _, _, r, _ in fps)
    print(f"  by trigger: {dict(by_reason)}")
    print("  sample:")
    for rec, tok, lang, reason, sig in fps[:12]:
        print(f"    {tok!r:24} gold={lang:6} via {reason:14} margin={sig['tamil_margin']:+.3f}")

    # --- operating curve ----------------------------------------------------
    # One threshold is not a useful summary when there are 9 positives. Sweeping
    # it shows whether the misses are near the boundary (worth tuning) or far
    # from it (needs more lexicon, not a different threshold).
    print("\nOPERATING CURVE — margin threshold vs recall / false positives")
    print(f"  {'pctile':>7} {'threshold':>10} {'LOO recall':>11} {'full-lex':>9} "
          f"{'false pos':>10} {'FP %':>7}")
    curve = []
    for pct in (0.999, 0.99, 0.98, 0.95, 0.90, 0.80):
        thr = neg_margins[min(int(pct * len(neg_margins)), len(neg_margins) - 1)]
        loo_r = 0
        for entry in gt["genuinely_tamil"]:
            norm = normalize_token(entry["token"])
            det = TamilFlagger([w for w in lexicon if w != norm], si_words, en_words)
            if det.flag(entry["token"], thr)[0]:
                loo_r += 1
        full_r = sum(1 for e in gt["genuinely_tamil"]
                     if full.flag(e["token"], thr)[0])
        n_fp = sum(1 for t in all_tokens if t["token"] not in tamil_set
                   and full.flag(t["token"], thr)[0])
        curve.append({"percentile": pct, "threshold": round(thr, 4),
                      "loo_recall": loo_r, "full_lexicon_recall": full_r,
                      "false_positives": n_fp,
                      "fp_pct": round(100 * n_fp / len(all_tokens), 3)})
        print(f"  {pct:>7.3f} {thr:>+10.3f} {loo_r:>8}/9 {full_r:>7}/9 "
              f"{n_fp:>10} {100 * n_fp / len(all_tokens):>6.2f}%")
    print("  LOO = the word is absent from the lexicon (pessimistic bound)")
    print("  full-lex = the word is present, as it would be with a real Tamil")
    print("             dictionary — these are all common Tamil words")

    # --- the plan's outlier-only variant, for comparison --------------------
    print("\nPLAN'S OUTLIER-ONLY RULE (no Tamil lexicon; low likelihood under SI and EN)")
    out_hits = sum(1 for e in gt["genuinely_tamil"]
                   if full.score(e["token"])["native_max"] <= outlier_threshold)
    out_fps = sum(1 for t in all_tokens if t["token"] not in tamil_set
                  and full.score(t["token"])["native_max"] <= outlier_threshold)
    print(f"  recovered {out_hits}/{len(tamil_tokens)} Tamil tokens")
    print(f"  false positives: {out_fps} ({100 * out_fps / len(all_tokens):.2f}% of corpus)")

    # --- the mislabelling this exposes --------------------------------------
    sinhala_script_tamil = [e for e in gt["genuinely_tamil"] if e["script"] == "sinhala"]
    print(f"\nCONFIDENCE BUG: {len(sinhala_script_tamil)} Tamil tokens are in Sinhala "
          f"script.\n  lid_rules labels each of them SI with confidence 0.99 "
          f"(unicode_sinhala).\n  Confidently wrong is the worst failure mode for a "
          f"signal C2/C3/C4 rely on.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "c1_otherlang_eval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "note": "Evaluated against curated ground truth (data/tamil_ground_truth.json), "
                    "NOT the gold OTHER label, which is 88% non-Tamil. Raw counts are "
                    "reported instead of F1 because there are only 9 positives.",
            "n_ground_truth_tamil": len(tamil_tokens),
            "tamil_by_script": gt["counts"]["tamil_by_script"],
            "lexicon_size": len(lexicon),
            "thresholds": {
                "tamil_margin": round(margin_threshold, 4),
                "native_max_outlier": round(outlier_threshold, 4),
                "calibrated_on": "non-target corpus tokens only",
            },
            "leave_one_out": {
                "recovered": n_found,
                "total": len(loo_hits),
                "detail": [{"token": t, "flagged": f, "reason": r,
                            "margin": s["tamil_margin"], "script": s["script"]}
                           for t, f, r, s in loo_hits],
            },
            "false_positives": {
                "count": len(fps),
                "pct_of_corpus": round(100 * len(fps) / len(all_tokens), 3),
                "by_trigger": dict(by_reason),
                "examples": [{"recording": r, "token": t, "gold_lang": l, "reason": rs}
                             for r, t, l, rs, _ in fps[:40]],
            },
            "operating_curve": curve,
            "plan_outlier_only_variant": {
                "recovered": out_hits,
                "total": len(tamil_tokens),
                "false_positives": out_fps,
                "pct_of_corpus": round(100 * out_fps / len(all_tokens), 3),
            },
        }, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
