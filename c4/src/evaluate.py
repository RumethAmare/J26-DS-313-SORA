"""
Evaluation harness for C4 (FR10).

Entity-level precision, recall and F1 under STRICT span matching: a prediction
is correct only if its doc_id, start_char, end_char and label all equal a gold
span. This is the matching the 0.231 baseline was scored with, so numbers are
directly comparable. Recall is the primary metric (NFR3): a false negative is a
disclosure, a false positive only costs readability.

Reported per label, per script (LATIN / SINHALA -- the measurement behind
Contribution 2), per document kind (transcript / clean_en / clean_si /
summary), and as micro and macro averages.

Data sources:

    synthetic-test    held-out synthetic split (frozen; the development score)
    synthetic-train   synthetic train split (regression only -- not a result)
    real-dev          real recordings used for tuning
    real-eval         the five held-out real recordings (report, never tune)

    python evaluate.py --source synthetic-test
    python evaluate.py --source real-dev --labels structured
    python evaluate.py --source real-eval --save

Surfaces from real recordings are masked (digits -> 9, letters -> x / ස) in
every saved error list, so no personal data is written into this repository.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from corpus import (C4_ROOT, EVAL_RECORDINGS, Recording, load_real, load_synthetic,
                    preceding_texts, real_recording_ids)

# ---------------------------------------------------------------------------
# Label sets
# ---------------------------------------------------------------------------

LABEL_SETS = {
    # The seven labels of the proposal (SO1) -- the headline comparison.
    "proposal": ("PERSON", "NIC", "PHONE", "ADDRESS", "ORG", "ACCOUNT", "DOB"),
    # What the rule layer (SO3) is responsible for.
    "structured": ("NIC", "PHONE", "ACCOUNT", "DOB", "EMAIL"),
    # Everything the schema defines that occurs in the corpus.
    "all": ("PERSON", "NIC", "PHONE", "ADDRESS", "ORG", "ACCOUNT", "DOB",
            "EMAIL", "LOCATION"),
}

_SINHALA = re.compile(r"[඀-෿]")


def doc_kind(doc_id: str) -> str:
    if "_transcript_" in doc_id:
        return "transcript"
    if "_summary_" in doc_id:
        return "summary"
    if doc_id.endswith("_si_v1"):
        return "clean_si"
    return "clean_en"


def span_script(surface: str, doc_id: str) -> str:
    """Script of a span; a digit-only span takes its document's script."""
    if _SINHALA.search(surface):
        return "SINHALA"
    return "SINHALA" if doc_kind(doc_id) == "clean_si" else "LATIN"


def mask(text: str) -> str:
    """Shape of a real surface without its content."""
    text = re.sub(r"\d", "9", text)
    text = re.sub(r"[A-Za-z]", "x", text)
    return _SINHALA.sub("ස", text)


# ---------------------------------------------------------------------------
# Spans and scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int
    label: str
    surface: str
    script: str
    noise: tuple = ()

    @property
    def key(self) -> tuple:
        return self.doc_id, self.start, self.end, self.label


def _strip_edge_punct(row: dict) -> dict:
    """Lenient gold: drop trailing punctuation the schema says to exclude."""
    surface = row["surface"]
    trimmed = surface.rstrip(".,;:")
    if trimmed == surface or not trimmed:
        return row
    return {**row, "surface": trimmed, "end_char": row["end_char"] - (len(surface) - len(trimmed))}


def gold_spans(rec: Recording, labels: Iterable[str], lenient: bool = False) -> list[Span]:
    spans = []
    for row in rec.rows:
        if row["label"] not in labels or row["doc_id"] not in rec.texts:
            continue
        if lenient:
            row = _strip_edge_punct(row)
        spans.append(Span(row["doc_id"], row["start_char"], row["end_char"], row["label"],
                          row["surface"], row.get("script") or
                          span_script(row["surface"], row["doc_id"]),
                          tuple(row.get("noise", ()))))
    return spans


Predictor = Callable[[str, str], list]   # (text, preceding) -> objects with start/end/label


def predicted_spans(rec: Recording, predict: Predictor, labels: Iterable[str]) -> list[Span]:
    # A recording-level system (one that looks across documents, such as
    # propagation) exposes .recording(texts) -> {doc_id: detections}.
    if hasattr(predict, "recording"):
        per_doc = predict.recording(rec.texts)
    else:
        previous = preceding_texts(rec.texts)
        per_doc = {d: predict(t, previous[d]) for d, t in rec.texts.items()}
    spans = []
    for doc_id, text in rec.texts.items():
        for p in per_doc[doc_id]:
            if p.label in labels:
                surface = text[p.start:p.end]
                spans.append(Span(doc_id, p.start, p.end, p.label, surface,
                                  span_script(surface, doc_id)))
    return spans


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}


def score(gold: list[Span], pred: list[Span], labels: Iterable[str]) -> dict:
    """Strict-match metrics, broken down by label, script and document kind."""
    labels = list(labels)
    gold_keys = {g.key for g in gold}
    pred_keys = {p.key for p in pred}

    counts = {axis: defaultdict(Counter) for axis in ("label", "script", "doc_kind")}

    def add(span: Span, outcome: str) -> None:
        counts["label"][span.label][outcome] += 1
        counts["script"][span.script][outcome] += 1
        counts["doc_kind"][doc_kind(span.doc_id)][outcome] += 1

    for g in gold:
        add(g, "tp" if g.key in pred_keys else "fn")
    for p in pred:
        if p.key not in gold_keys:
            add(p, "fp")

    def table(axis: str, keys: Iterable[str]) -> dict:
        return {k: prf(counts[axis][k]["tp"], counts[axis][k]["fp"], counts[axis][k]["fn"])
                for k in keys}

    per_label = table("label", labels)
    tp = sum(v["tp"] for v in per_label.values())
    fp = sum(v["fp"] for v in per_label.values())
    fn = sum(v["fn"] for v in per_label.values())
    # Macro averages only over labels that occur in the gold data: a label with
    # no gold spans has no defined recall and would drag the mean to 0.
    present = [v for v in per_label.values() if v["support"]]
    macro = {m: round(sum(v[m] for v in present) / len(present), 4) if present else 0.0
             for m in ("precision", "recall", "f1")}

    return {
        "micro": prf(tp, fp, fn),
        "macro": macro,
        "per_label": per_label,
        "per_script": table("script", sorted(counts["script"])),
        "per_doc_kind": table("doc_kind", sorted(counts["doc_kind"])),
    }


def classify_errors(gold: list[Span], pred: list[Span], masked: bool) -> dict:
    """
    Sort every error into one bucket, for error analysis:

      boundary   same label, overlapping, different offsets (partly found)
      label      same offsets, different label (found, misnamed)
      missed     no overlapping prediction at all (a disclosure)
      spurious   no overlapping gold span at all
    """
    gold_keys = {g.key for g in gold}
    pred_keys = {p.key for p in pred}
    by_doc_pred, by_doc_gold = defaultdict(list), defaultdict(list)
    for p in pred:
        by_doc_pred[p.doc_id].append(p)
    for g in gold:
        by_doc_gold[g.doc_id].append(g)

    def show(s: Span) -> str:
        return mask(s.surface) if masked else s.surface

    buckets: dict[str, list] = defaultdict(list)
    for g in gold:
        if g.key in pred_keys:
            continue
        overlaps = [p for p in by_doc_pred[g.doc_id] if p.start < g.end and p.end > g.start]
        if any(p.label == g.label for p in overlaps):
            kind = "boundary"
        elif any((p.start, p.end) == (g.start, g.end) for p in overlaps):
            kind = "label"
        else:
            kind = "missed"
        buckets[kind].append({"doc_id": g.doc_id, "label": g.label, "gold": show(g),
                              "predicted": [f"{p.label}:{show(p)}" for p in overlaps],
                              **({"noise": list(g.noise)} if g.noise else {})})
    for p in pred:
        if p.key in gold_keys:
            continue
        if not any(g.start < p.end and g.end > p.start for g in by_doc_gold[p.doc_id]):
            buckets["spurious"].append({"doc_id": p.doc_id, "label": p.label,
                                        "predicted": show(p)})
    return dict(buckets)


def recall_by_noise(gold: list[Span], pred: list[Span]) -> dict:
    """Synthetic only: recall on spans carrying each kind of speech noise."""
    pred_keys = {p.key for p in pred}
    hit, total = Counter(), Counter()
    for g in gold:
        for n in g.noise or ("clean",):
            total[n] += 1
            hit[n] += g.key in pred_keys
    return {n: {"recall": round(hit[n] / total[n], 4), "support": total[n]}
            for n in sorted(total)}


# ---------------------------------------------------------------------------
# Running an evaluation
# ---------------------------------------------------------------------------

SOURCES = ("synthetic-test", "synthetic-train", "real-dev", "real-eval")


def load_source(source: str) -> list[Recording]:
    if source.startswith("synthetic-"):
        return load_synthetic(source.removeprefix("synthetic-"))
    ids = real_recording_ids()
    if source == "real-eval":
        ids = [r for r in ids if r in EVAL_RECORDINGS]
    else:
        ids = [r for r in ids if r not in EVAL_RECORDINGS]
    return [load_real(r) for r in ids]


def get_system(name: str) -> Predictor:
    if name == "rules":
        from rules import detect
        return detect
    if name == "rules+propagation":
        from redact import detect_recording

        class Propagating:
            recording = staticmethod(detect_recording)
        return Propagating()
    # model:<data> / hybrid:<data>, where <data> is real, synthetic or both
    if name.startswith(("model:", "hybrid:")):
        from ner import HybridDetector, ModelDetector
        kind, data = name.split(":", 1)
        return ModelDetector(data) if kind == "model" else HybridDetector(data)
    raise SystemExit(f"unknown system {name!r}")


def evaluate(recordings: list[Recording], predict: Predictor, labels: Iterable[str],
             lenient: bool = False, masked: bool = False) -> dict:
    labels = tuple(labels)
    gold, pred = [], []
    for rec in recordings:
        gold += gold_spans(rec, labels, lenient)
        pred += predicted_spans(rec, predict, labels)
    result = score(gold, pred, labels)
    result["errors"] = classify_errors(gold, pred, masked)
    if any(g.noise for g in gold):
        result["recall_by_noise"] = recall_by_noise(gold, pred)
    return result


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C4_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(source: str, system: str, label_set: str, lenient: bool = False) -> dict:
    recordings = load_source(source)
    labels = LABEL_SETS[label_set]
    result = evaluate(recordings, get_system(system), labels, lenient,
                      masked=source.startswith("real-"))
    return {
        "config": {
            "system": system, "source": source, "labels": label_set,
            "matching": "strict" + (" (gold trailing punctuation stripped)" if lenient else ""),
            "recordings": [r.rid for r in recordings],
            "commit": _git_commit(),
            "date": dt.date.today().isoformat(),
        },
        **result,
    }


def format_report(report: dict) -> str:
    cfg = report["config"]
    lines = [f"C4 evaluation  system={cfg['system']}  source={cfg['source']}  "
             f"labels={cfg['labels']}  matching={cfg['matching']}",
             f"{len(cfg['recordings'])} recordings, commit {cfg['commit']}", ""]

    def row(name: str, m: dict) -> str:
        return (f"  {name:<12} {m['recall']:>7.3f} {m['precision']:>9.3f} {m['f1']:>7.3f}"
                f" {m['tp']:>6} {m['fp']:>6} {m['fn']:>6}")

    header = f"  {'':<12} {'recall':>7} {'precision':>9} {'f1':>7} {'tp':>6} {'fp':>6} {'fn':>6}"
    for title, key in (("per label", "per_label"), ("per script", "per_script"),
                       ("per document", "per_doc_kind")):
        lines += [title, header]
        lines += [row(k, v) for k, v in report[key].items()]
        lines.append("")
    lines.append(row("MICRO", report["micro"]))
    mac = report["macro"]
    lines.append(f"  {'MACRO':<12} {mac['recall']:>7.3f} {mac['precision']:>9.3f} {mac['f1']:>7.3f}")
    if "recall_by_noise" in report:
        lines += ["", "recall by speech noise (synthetic)"]
        lines += [f"  {k:<20} {v['recall']:.3f}  (n={v['support']})"
                  for k, v in report["recall_by_noise"].items()]
    lines += ["", "errors: " + ", ".join(f"{k} {len(v)}" for k, v in report["errors"].items())]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="C4 strict-span evaluation")
    ap.add_argument("--source", choices=SOURCES, default="synthetic-test")
    ap.add_argument("--system", default="rules")
    ap.add_argument("--labels", choices=list(LABEL_SETS), default="structured")
    ap.add_argument("--lenient", action="store_true",
                    help="strip trailing punctuation from gold spans (defect analysis only)")
    ap.add_argument("--save", action="store_true", help="write the report to eval/")
    args = ap.parse_args()

    if args.source == "real-eval":
        print("NOTE: real-eval is the held-out test set -- report it, never tune on it.\n",
              file=sys.stderr)
    report = run(args.source, args.system, args.labels, args.lenient)
    print(format_report(report))
    if args.save:
        # ":" is not a legal filename character on Windows (model:both -> model-both).
        out = C4_ROOT / "eval" / (f"{args.system.replace(':', '-')}_{args.source}_{args.labels}"
                                  f"{'_lenient' if args.lenient else ''}.json")
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nsaved -> {out.relative_to(C4_ROOT)}")


if __name__ == "__main__":
    main()
