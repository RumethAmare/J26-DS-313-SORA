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
