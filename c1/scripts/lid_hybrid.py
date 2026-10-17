#!/usr/bin/env python3
"""
lid_hybrid.py -- Task 3 deliverable: rule-first, fastText-fallback LID.

The predictor Tasks 4, 5 and 8 should import. Routing:

    Sinhala script  -> Unicode rule    (deterministic, 97.7% against gold)
    Tamil script    -> Unicode rule    (deterministic; never fires in this
                                        corpus -- there is no Tamil script in
                                        it, see the note below)
    pure numeric    -> numeric rule    (kept for schema compatibility; gold's
                                        own numeric tagging is inconsistent,
                                        so confidence stays low)
    Latin script    -> fastText        (the only genuinely ambiguous case)

Measured by 5-fold cross-validation folded by recording
(`train_lid_fasttext.py`, 48 recordings), this lifts overall token accuracy
from 0.8807 +/- 0.0208 to 0.9232 +/- 0.0218 (0.9276 -> 0.9735 excluding
numerals, whose gold convention is inconsistent), and Latin-script accuracy
from 0.8487 to 0.9535.

It keeps the same `(lang, confidence, method)` signature as lid_rules.predict,
so it is a drop-in replacement in build_token_stream.py.

ON `OTHER`
----------
This predictor will essentially never return OTHER, and that is a property of
the training data rather than a bug to fix later. See
results/c1_lid_report.md: of 743 OTHER tokens, ~720 are numerals tagged under
an inconsistent convention, and only ~22 are in Latin script -- spread across
identifiers (`LN2024NG00445`, `hasitha.94@gmail.com.`), ordinals (`21st,`,
`No.`) and a handful of Tamil words. There is no coherent class to learn;
Tamil detection is Task 5's job (otherlang_flag.py).

Usage:
    from lid_hybrid import HybridLID
    lid = HybridLID()                 # loads models/lid_latin.ftz
    lang, conf, method = lid.predict("Nawaloka")
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lid_data
import lid_rules

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(C1_ROOT, "models", "lid_latin.ftz")


class HybridLID:
    def __init__(self, model_path=DEFAULT_MODEL):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No fastText model at {model_path}. Run train_lid_fasttext.py first.")
        import fasttext
        self.model = fasttext.load_model(model_path)
        self.model_path = model_path

    def _predict_latin(self, token):
        """Raw-binding predict; fastText's own wrapper breaks on NumPy 2.x.

        `model.predict()` finishes with `np.array(probs, copy=False)`, which
        NumPy 2 refuses. `model.f.predict` is the C++ call underneath and
        returns plain tuples.
        """
        line = lid_data.to_fasttext_line(token) + "\n"
        predictions = self.model.f.predict(line, 1, 0.0, "strict")
        if not predictions:
            # No prediction at all -- fall back to the rule rather than
            # inventing a label.
            return lid_rules.predict(token)
        prob, label = predictions[0]
        return label.replace("__label__", ""), round(float(prob), 4), "fasttext_latin"

    def predict(self, token):
        """Return (lang, confidence, method) -- same contract as lid_rules."""
        script = lid_data.token_script(token)
        if script == "latin":
            return self._predict_latin(token)
        return lid_rules.predict(token)


def _demo():
    lid = HybridLID()
    # Tokens the heuristic gets wrong, drawn from the measured error analysis.
    samples = [
        ("Nawaloka", "EN", "proper noun the heuristic calls SI"),
        ("Dehiwala", "EN", "place name"),
        ("Sampath", "EN", "bank name"),
        ("NIC", "EN", "acronym"),
        ("me", "SI", "romanized Sinhala colliding with English"),
        ("da", "SI", "same"),
        ("one", "SI", "the 'mama' problem"),
        ("thiyenawa", "SI", "unambiguous romanized Sinhala"),
        ("report", "EN", "ordinary English"),
        ("එකට", "SI", "Sinhala script -> rule"),
        ("075", "EN", "numeric -> rule"),
    ]
    print(f"{'token':14} {'gold':6} {'pred':6} {'conf':>6}  {'method':16} {'':3} note")
    for token, gold, note in samples:
        lang, conf, method = lid.predict(token)
        mark = "ok" if lang == gold else "MISS"
        print(f"{token:14} {gold:6} {lang:6} {conf:>6.2f}  {method:16} {mark:4} {note}")


if __name__ == "__main__":
    _demo()
