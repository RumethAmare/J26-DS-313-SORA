"""
Translation scoring: chrF++ (headline) and BLEU, via sacrebleu.

chrF++ leads because Sinhala is morphologically rich: BLEU's word n-grams
punish correct inflection variants that chrF's character n-grams credit.
"""
import sacrebleu


def score(hypotheses, references):
    """Corpus-level scores for parallel lists of strings."""
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses vs {len(references)} references")
    return {
        "n": len(hypotheses),
        "chrf++": round(sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score, 2),
        "bleu": round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2),
    }
