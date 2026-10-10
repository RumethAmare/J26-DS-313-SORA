"""
Multilingual transformer NER for PERSON / ADDRESS / ORG / LOCATION (SO4).

Fine-tunes xlm-roberta-base, whose pre-training covers Sinhala, for token
classification. It replaces or joins the CPU spaCy model behind the same
Detection interface (ner.ModelDetector).

Trained ON THIS LAPTOP: the recordings carry no signed consent yet, so the
training text must not leave the machine (NFR2). Two measures make CPU
training feasible:

  * the 250k-word embedding table is frozen (85M of 277M parameters train);
    updating it dominated the step time (75 s -> 4 s per step here)
  * every document with an entity is kept, plus a seeded 30% sample of the
    documents without one, so the model still learns what is NOT a name

    python transformer_ner.py train --data both --dev-fold   # for the decision
    python transformer_ner.py train --data both               # the final model
    python evaluate.py --source real-devfold --system pipeline:xlmr_both_devfold

Models are written to models/xlmr_<data>[_devfold]/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from functools import lru_cache

from corpus import C4_ROOT
from ner import MODEL_DIR, MODEL_LABELS, ROLE_OF, span_examples, training_recordings
from rules import Detection

BASE_MODEL = "xlm-roberta-base"
LABELS = ["O"] + [f"{p}-{l}" for l in MODEL_LABELS for p in ("B", "I")]
LABEL_ID = {l: i for i, l in enumerate(LABELS)}
MAX_LEN = 192
NEGATIVE_KEEP = 0.3


def model_dir(data: str, exclude_dev: bool):
    return MODEL_DIR / f"xlmr_{data}{'_devfold' if exclude_dev else ''}"


def encode(tokenizer, text: str, spans: list) -> dict:
    """Token ids and BIO labels aligned to character spans."""
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
    labels = []
    for s, e in enc["offset_mapping"]:
        if s == e:                                   # special tokens
            labels.append(-100)
            continue
        tag = "O"
        for a, b, label in spans:
            if s >= a and e <= b + 1 and s < b:      # token inside the span
                tag = ("B-" if s <= a else "I-") + label
                break
        labels.append(LABEL_ID[tag])
    return {"input_ids": enc["input_ids"], "labels": labels}


def train(data: str = "both", exclude_dev: bool = False, epochs: int = 3, lr: float = 5e-5,
          batch_size: int = 16, seed: int = 13) -> None:
    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(seed)
    torch.set_num_threads(8)
    rng = random.Random(seed)
    examples, skipped = span_examples(training_recordings(data, exclude_dev))
    examples = [(t, s) for t, s in examples if t.strip()]
    kept = [x for x in examples if x[1] or rng.random() < NEGATIVE_KEEP]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL_ID)
    model.roberta.embeddings.word_embeddings.weight.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]

    encoded = [encode(tokenizer, t, s) for t, s in kept]
    # Length-sorted buckets keep padding (and CPU time) low; buckets are shuffled.
    encoded.sort(key=lambda x: len(x["input_ids"]))
    batches = [encoded[i:i + batch_size] for i in range(0, len(encoded), batch_size)]
    steps = epochs * len(batches)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    print(f"training {BASE_MODEL} on '{data}'{' (dev fold excluded)' if exclude_dev else ''}: "
          f"{len(kept)} documents ({sum(1 for _, s in kept if s)} with entities), "
          f"{len(batches)} batches x {epochs} epochs; skipped {skipped}", flush=True)

    pad = tokenizer.pad_token_id
    started, step = time.time(), 0
    model.train()
    for epoch in range(1, epochs + 1):
        rng.shuffle(batches)
        total = 0.0
        for batch in batches:
            width = max(len(x["input_ids"]) for x in batch)
            ids = torch.tensor([x["input_ids"] + [pad] * (width - len(x["input_ids"])) for x in batch])
            lab = torch.tensor([x["labels"] + [-100] * (width - len(x["labels"])) for x in batch])
            loss = model(input_ids=ids, attention_mask=(ids != pad).long(), labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            total += loss.item()
            step += 1
            if step % 50 == 0:
                done = step / steps
                eta = (time.time() - started) / done * (1 - done)
                print(f"  epoch {epoch} step {step}/{steps}  loss {total / (step - (epoch - 1) * len(batches)):.4f}"
                      f"  elapsed {time.time() - started:5.0f}s  eta {eta / 60:4.0f} min", flush=True)

    out = model_dir(data, exclude_dev)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    with open(out / "c4_training.json", "w", encoding="utf-8") as f:
        json.dump({"base": BASE_MODEL, "data": data, "exclude_dev": exclude_dev, "epochs": epochs,
                   "lr": lr, "batch_size": batch_size, "seed": seed, "documents": len(kept),
                   "frozen": "word_embeddings", "negative_keep": NEGATIVE_KEEP,
                   "seconds": round(time.time() - started)}, f, indent=2)
    print(f"saved -> {out.relative_to(C4_ROOT)}", flush=True)


@lru_cache(maxsize=4)
def _load(path: str):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path).eval()
    return tok, model


# Span post-processing. Each switch is a general rule about text, not about
# any recording; each was kept only if it improved the development fold.
POSTPROCESS = {
    "whole_words": True,        # a span never ends or starts inside a word
    "split_sentences": True,    # a name never runs across ". " (except Mr./Dr.)
    "trim_case_endings": True,  # "නිමල්ට" -> "නිමල්", as in name propagation
    # label -> minimum mean token confidence. Chosen on the development fold:
    # F1 is flat for 0.8-0.9, so 0.85 is not a knife-edge value.
    "min_score": {"PERSON": 0.85, "ADDRESS": 0.85},
}
_HONORIFIC_END = ("mr", "mrs", "ms", "dr", "prof", "rev", "st", "no")


def _is_word_char(ch: str) -> bool:
    import unicodedata
    return unicodedata.category(ch)[0] in "LMN" or ch in "‍‌"


class TransformerDetector:
    """PERSON / ADDRESS / ORG / LOCATION from the fine-tuned transformer."""

    def __init__(self, name: str):
        self.path = MODEL_DIR / name
        if not self.path.exists():
            raise FileNotFoundError(f"no model at {self.path}")
        self._cache: dict[str, list] = {}

    def raw_spans(self, text: str) -> list[list]:
        """[start, end, label, confidence] straight from the model (cached)."""
        if text in self._cache:
            return self._cache[text]
        import torch
        tok, model = _load(str(self.path))
        enc = tok(text, truncation=True, max_length=512, return_offsets_mapping=True,
                  return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            probs = model(**enc).logits[0].softmax(-1)
        conf, pred = probs.max(-1)
        spans, cur = [], None
        for (s, e), p, c in zip(offsets, pred.tolist(), conf.tolist()):
            if s == e:
                continue
            tag = LABELS[p]
            if tag == "O":
                cur = None
                continue
            kind, label = tag.split("-", 1)
            # A word piece continuing the current word, or an I- tag, extends it.
            if cur and cur[2] == label and (kind == "I" or s == cur[1]):
                cur[1] = e
                cur[4].append(c)
            else:
                cur = [s, e, label, None, [c]]
                spans.append(cur)
        out = [[s, e, label, sum(cs) / len(cs)] for s, e, label, _, cs in spans]
        self._cache[text] = out
        return out

    def __call__(self, text: str, preceding: str = "") -> list[Detection]:
        if not text.strip():
            return []
        cfg = POSTPROCESS
        pieces = []
        for s, e, label, score in self.raw_spans(text):
            if score < cfg["min_score"].get(label, 0.0):
                continue
            if cfg["whole_words"]:
                while s > 0 and _is_word_char(text[s - 1]) and _is_word_char(text[s]):
                    s -= 1
                while e < len(text) and _is_word_char(text[e - 1]) and _is_word_char(text[e]):
                    e += 1
            parts = [(s, e)]
            if cfg["split_sentences"] and label == "PERSON":
                parts = _split_at_sentence(text, s, e)
            for ps, pe in parts:
                pieces.append((ps, pe, label, score))
        out = []
        for s, e, label, score in pieces:
            while e > s and text[e - 1] in " .,;:?!":
                e -= 1
            while s < e and text[s] == " ":
                s += 1
            if cfg["trim_case_endings"] and label == "PERSON":
                e = s + _without_case_ending(text[s:e])
            if e > s and not any(s < o.end and o.start < e for o in out):
                out.append(Detection(s, e, label, text[s:e], role=ROLE_OF[label],
                                     source="xlmr", score=round(score, 4)))
        return out


def _split_at_sentence(text: str, s: int, e: int) -> list[tuple[int, int]]:
    """Split a name span at ". " unless the word before the stop is a title."""
    parts, start = [], s
    i = text.find(". ", s, e)
    while i != -1:
        word = text[start:i].split()[-1].lower() if text[start:i].split() else ""
        if word not in _HONORIFIC_END:
            parts.append((start, i))
            start = i + 2
        i = text.find(". ", i + 2, e)
    parts.append((start, e))
    return [p for p in parts if p[1] > p[0]]


def _without_case_ending(span: str) -> int:
    """Length of a Sinhala name span with a trailing case ending removed."""
    from resolve import SINHALA_CASE_ENDINGS, _SINHALA
    words = span.split(" ")
    last = words[-1]
    if _SINHALA.search(last):
        for ending in SINHALA_CASE_ENDINGS:
            if last.endswith(ending) and len(last) > len(ending) + 1:
                return len(span) - len(ending)
    return len(span)


def main() -> None:
    ap = argparse.ArgumentParser(description="C4 transformer NER (SO4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", choices=["real", "synthetic", "both"], default="both")
    t.add_argument("--dev-fold", action="store_true")
    t.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    train(args.data, args.dev_fold, args.epochs)


if __name__ == "__main__":
    main()
