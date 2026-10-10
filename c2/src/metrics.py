"""
Translation scoring: chrF++ (headline) and BLEU, via sacrebleu.

chrF++ leads because Sinhala is morphologically rich: BLEU's word n-grams
punish correct inflection variants that chrF's character n-grams credit.
"""
import sacrebleu

# Zero-width joiner/non-joiner only change how Sinhala conjuncts render (e.g.
# the yansaya in "වෛද්‍ය"). ~1 in 4 references has one and model tokenizers
# often drop them, so they are removed from both sides before scoring.
_ZW = str.maketrans("", "", "\u200c\u200d")


def score(hypotheses, references):
    """Corpus-level scores for parallel lists of strings."""
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses vs {len(references)} references")
    hypotheses = [h.translate(_ZW) for h in hypotheses]
    references = [r.translate(_ZW) for r in references]
    return {
        "n": len(hypotheses),
        "chrf++": round(sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score, 2),
        "bleu": round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2),
    }


def score_summaries(hypotheses, references):
    """Mean ROUGE-L F1 (x100) over summaries."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    f = [scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hypotheses, references)]
    return {"n": len(f), "rougeL": round(100 * sum(f) / max(len(f), 1), 2)}


def _words(text):
    return set("".join(c if c.isalnum() else " " for c in (text or "").lower()).split())


def _overlap_f1(a, b):
    a, b = _words(a), _words(b)
    common = len(a & b)
    return 0.0 if not common else 2 * common / (len(a) + len(b))


def score_actions(predicted, gold, threshold=0.5):
    """Action-item precision/recall/F1 over all recordings.

    An item matches when its intent shares enough words with a gold intent
    (word-overlap F1 >= threshold; loose, since wording varies). "+owner" also
    needs the owners to share a word ("Amashi (agent)" ~ "Amashi").
    """
    n_pred = sum(len(p) for p in predicted)
    n_gold = sum(len(g) for g in gold)
    tp_intent = tp_owner = 0
    for preds, golds in zip(predicted, gold):
        free = list(preds)
        for g in golds:
            best = max(free, key=lambda p: _overlap_f1(p["intent"], g["intent"]), default=None)
            if best is None or _overlap_f1(best["intent"], g["intent"]) < threshold:
                continue
            free.remove(best)
            tp_intent += 1
            tp_owner += bool(_words(best.get("owner")) & _words(g.get("owner")))

    def prf(tp):
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gold if n_gold else 0.0
        return {"p": round(100 * p, 1), "r": round(100 * r, 1),
                "f1": round(200 * p * r / (p + r), 1) if p + r else 0.0}

    return {"pred": n_pred, "gold": n_gold, "intent": prf(tp_intent), "intent+owner": prf(tp_owner)}
