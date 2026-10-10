"""
Learned detection of unstructured identifiers: PERSON, ADDRESS, ORG, LOCATION
(SO4), and the hybrid detector that combines it with the rule layer.

Structured identifiers have fixed formats and are left to the rules. Names,
addresses and organisations have no shared surface pattern, so they are
learned. This module trains a CPU spaCy NER pipeline; it trains in minutes on a
laptop and runs fully offline. A multilingual transformer (xlm-roberta-base)
is the planned upgrade once GPU time is available -- it plugs in behind the
same Detection interface.

Training data comes from three sources, so the value of synthetic data can be
measured rather than assumed:

    real        complete real recordings outside the held-out set
    synthetic   the synthetic train split
    both        both of the above

Every model is scored on the same five held-out real recordings.

    python ner.py train --data both
    python evaluate.py --source real-eval --system hybrid:both --labels proposal

Trained models live in models/, which git ignores: they are rebuilt from this
script and the seed, not committed.
"""
from __future__ import annotations

import argparse
import random
import time
from functools import lru_cache
from pathlib import Path

from corpus import (C4_ROOT, EVAL_RECORDINGS, Recording, complete_recording_ids, load_real,
                    load_synthetic)
from rules import Detection
from rules import detect as rule_detect

MODEL_LABELS = ("PERSON", "ADDRESS", "ORG", "LOCATION")
MODEL_DIR = C4_ROOT / "models"

ROLE_OF = {"PERSON": "PRIVATE_INDIVIDUAL", "ADDRESS": "PRIVATE_INDIVIDUAL",
           "ORG": "ORGANISATION", "LOCATION": "PUBLIC_PLACE"}


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


def dev_fold() -> list[str]:
    """
    Every fifth complete tuning recording. A model trained without them can be
    analysed and improved on them, so the held-out five stay untouched until
    the final score.
    """
    tuning = [r for r in complete_recording_ids() if r not in EVAL_RECORDINGS]
    return tuning[::5]


def training_recordings(data: str, exclude_dev: bool = False) -> list[Recording]:
    recs: list[Recording] = []
    if data in ("synthetic", "both"):
        recs += load_synthetic("train")
    if data in ("real", "both"):
        skip = set(EVAL_RECORDINGS) | (set(dev_fold()) if exclude_dev else set())
        recs += [load_real(r) for r in complete_recording_ids() if r not in skip]
    return recs


def span_examples(recordings: list[Recording], labels=MODEL_LABELS) -> tuple[list, dict]:
    """
    (text, [(start, end, label)]) per document, keeping only spans that are
    usable as training targets: offsets that match their surface (R0008 has
    stale offsets) and no overlaps (the longer span wins, as in the schema).
    """
    examples, skipped = [], {"offset": 0, "overlap": 0}
    for rec in recordings:
        by_doc: dict[str, list] = {}
        for row in rec.rows:
            if row["label"] not in labels:
                continue
            text = rec.texts.get(row["doc_id"])
            s, e = row["start_char"], row["end_char"]
            if text is None or text[s:e] != row["surface"]:
                skipped["offset"] += 1
                continue
            by_doc.setdefault(row["doc_id"], []).append((s, e, row["label"]))
        for doc_id, text in rec.texts.items():
            kept: list = []
            for s, e, label in sorted(by_doc.get(doc_id, []), key=lambda x: x[0] - x[1]):
                if any(s < ke and ks < e for ks, ke, _ in kept):
                    skipped["overlap"] += 1
                    continue
                kept.append((s, e, label))
            examples.append((text, sorted(kept)))
    return examples, skipped


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(data: str, epochs: int = 30, seed: int = 13, dropout: float = 0.2,
          batch_size: int = 16, out_dir: Path | None = None, verbose: bool = True,
          exclude_dev: bool = False) -> Path:
    import spacy
    from spacy.training import Example
    from spacy.util import fix_random_seed, minibatch

    fix_random_seed(seed)
    rng = random.Random(seed)
    examples, skipped = span_examples(training_recordings(data, exclude_dev))

    nlp = spacy.blank("xx")               # multilingual tokenizer: Sinhala + Latin
    ner = nlp.add_pipe("ner")
    for label in MODEL_LABELS:
        ner.add_label(label)

    train_set, misaligned = [], 0
    for text, spans in examples:
        doc = nlp.make_doc(text)
        ents = []
        for s, e, label in spans:
            span = doc.char_span(s, e, label=label, alignment_mode="expand")
            if span is None:
                misaligned += 1
                continue
            ents.append((span.start_char, span.end_char, label))
        train_set.append(Example.from_dict(doc, {"entities": ents}))

    if verbose:
        n_spans = sum(len(s) for _, s in examples)
        print(f"training on '{data}': {len(train_set)} documents, {n_spans} spans "
              f"(skipped: {skipped['offset']} bad offsets, {skipped['overlap']} overlaps, "
              f"{misaligned} misaligned)")

    optimizer = nlp.initialize(lambda: train_set)
    started = time.time()
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_set)
        losses: dict = {}
        for batch in minibatch(train_set, size=batch_size):
            nlp.update(batch, sgd=optimizer, drop=dropout, losses=losses)
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:>2}  loss {losses.get('ner', 0):9.1f}  "
                  f"{time.time() - started:5.0f}s", flush=True)

    out = out_dir or MODEL_DIR / f"ner_{data}{'_devfold' if exclude_dev else ''}"
    out.mkdir(parents=True, exist_ok=True)
    nlp.meta["c4"] = {"data": data, "epochs": epochs, "seed": seed, "dropout": dropout,
                      "batch_size": batch_size}
    nlp.to_disk(out)
    if verbose:
        print(f"saved -> {out.relative_to(C4_ROOT)}")
    return out


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load(path: str):
    import spacy
    return spacy.load(path)


class ModelDetector:
    """PERSON / ADDRESS / ORG / LOCATION from a trained model, as Detections."""

    def __init__(self, name: str):
        self.path = MODEL_DIR / f"ner_{name}"
        if not self.path.exists():
            raise FileNotFoundError(f"no model at {self.path}; run: python ner.py train --data {name}")

    def __call__(self, text: str, preceding: str = "") -> list[Detection]:
        doc = _load(str(self.path))(text)
        out = []
        for ent in doc.ents:
            # Trim edge whitespace and sentence-final punctuation (§4).
            s, e = ent.start_char, ent.end_char
            while e > s and text[e - 1] in " .,;:?!":
                e -= 1
            while s < e and text[s] == " ":
                s += 1
            if e > s:
                out.append(Detection(s, e, ent.label_, text[s:e], role=ROLE_OF[ent.label_],
                                     source="model"))
        return out


class HybridDetector:
    """
    Rules for structured identifiers, the model for everything else. Where the
    two overlap the rule wins: a fixed format is stronger evidence than a
    learned guess, and the rules were validated span by span.
    """

    def __init__(self, name: str):
        self.model = ModelDetector(name)

    def __call__(self, text: str, preceding: str = "") -> list[Detection]:
        ruled = rule_detect(text, preceding)
        learned = [m for m in self.model(text, preceding)
                   if not any(m.start < r.end and r.start < m.end for r in ruled)]
        return sorted(ruled + learned, key=lambda d: d.start)


def main() -> None:
    ap = argparse.ArgumentParser(description="C4 learned NER (SO4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", choices=["real", "synthetic", "both"], default="both")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--seed", type=int, default=13)
    t.add_argument("--dev-fold", action="store_true",
                   help="leave the development fold out (for error analysis)")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.data, args.epochs, args.seed, exclude_dev=args.dev_fold)


if __name__ == "__main__":
    main()
