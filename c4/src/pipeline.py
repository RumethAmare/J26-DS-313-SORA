"""
C4 end-to-end: C2 output in, redacted shareable output and a re-identification
map out (FR1, FR9, FR8).

    Input   the single-language record and summary from C2
              <rid>.translation.json  (clean_en / clean_si per utterance)
              <rid>.summary.json
            and, when present, the C1 transcript <rid>.transcript.json
    Output  out/<rid>.redacted.json     shareable: every document redacted,
                                         one placeholder per entity everywhere
            out/<rid>.detections.jsonl  every detected span, contract fields
            reid_map/<rid>.reid.json    the reversal map; git-ignored storage
                                         only (NFR2)

Stages: detect (rules + model) -> propagate known surfaces -> link names
across scripts -> classify person roles -> redact -> leak check.

    python pipeline.py --recording J26DS313_R0022
    python pipeline.py --c2-dir path/to/c2 --rid J26DS313_R0022 --c1-dir path/to/c1
    python pipeline.py --recording J26DS313_R0022 --offline-check --measure

--offline-check runs with every network connection blocked (NFR1).
--measure records per-document latency and peak memory (NFR4, NFR5).
"""
from __future__ import annotations

import argparse
import json
import socket
import time
import tracemalloc
from pathlib import Path

from corpus import C4_ROOT, dataset_root, load_texts
from redact import redact_recording, write_reid_map

OUT_DIR = C4_ROOT / "out"


class NetworkBlocked(RuntimeError):
    pass


def block_network() -> list:
    """Make any outbound connection fail loudly; returns the attempts seen."""
    attempts: list = []

    def refuse(self, address, *a, **k):
        attempts.append(address)
        raise NetworkBlocked(f"network access attempted: {address}")

    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = lambda address, *a, **k: refuse(None, address)
    return attempts


def load_inputs(rid: str, c2_dir: Path | None, c1_dir: Path | None) -> dict[str, str]:
    if c2_dir is None:
        return load_texts(rid)
    # A loose directory of C2 (and optional C1) files, laid out like the corpus.
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="c4_in_"))
    for sub, src in (("c2", c2_dir), ("c1", c1_dir)):
        if src is None:
            continue
        (tmp / "annotations" / sub).mkdir(parents=True, exist_ok=True)
        for p in Path(src).glob(f"{rid}.*.json"):
            shutil.copy(p, tmp / "annotations" / sub / p.name)
    try:
        return load_texts(rid, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def detector(model: str | None):
    if model:
        from ner import HybridDetector
        return HybridDetector(model)
    from rules import detect
    return detect


def _shown(path: Path) -> str:
    """A path relative to the component when it lies inside it."""
    try:
        return str(path.relative_to(C4_ROOT))
    except ValueError:
        return str(path)


def run(rid: str, texts: dict[str, str], model: str | None = "xlmr_both",
        save_map: bool = True, measure: bool = False) -> dict:
    if not texts:
        raise SystemExit(f"no C1/C2 text found for {rid}")
    det = detector(model)
    if measure:
        det("warm-up", "")                 # load the models before timing
        from roles import MODEL_PATH, _model
        if MODEL_PATH.exists():
            _model()
        tracemalloc.start()
    started = time.perf_counter()
    result = redact_recording(texts, rid, detect=det)
    elapsed = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1] if measure else None
    if measure:
        tracemalloc.stop()

    OUT_DIR.mkdir(exist_ok=True)
    redacted = {
        "recording_id": rid,
        "policy": "Every personal entity replaced by one consistent pseudonym in every "
                  "document; ORG, LOCATION and organisation-owned identifiers kept visible.",
        "entities": [{"pseudonym": e["placeholder"], "label": e["label"], "role": e["role"],
                      "mentions": len(e["mentions"])} for e in result.entities],
        "documents": [{"doc_id": doc_id.replace("_v1", "_redacted_v1"), "source_doc_id": doc_id,
                       "text": text,
                       "redacted_entities": sorted({r["placeholder"] for r in result.reid.get(doc_id, [])})}
                      for doc_id, text in result.texts.items()],
    }
    with open(OUT_DIR / f"{rid}.redacted.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(redacted, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / f"{rid}.detections.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for e in result.entities:
            for m in e["mentions"]:
                f.write(json.dumps({"doc_id": m["doc_id"], "field": "text", "entity_id": e["placeholder"],
                                    "label": e["label"], "start_char": m["start"], "end_char": m["end"],
                                    "role": e["role"], "redact": True, "source": m["source"]},
                                   ensure_ascii=False) + "\n")
    report = {
        "recording": rid,
        "documents": len(texts),
        "entities_redacted": len(result.entities),
        "by_label": {lab: sum(1 for e in result.entities if e["label"] == lab)
                     for lab in sorted({e["label"] for e in result.entities})},
        "leaks": len(result.leaks),
        "outputs": [_shown(OUT_DIR / f"{rid}.redacted.json"),
                    _shown(OUT_DIR / f"{rid}.detections.jsonl")],
    }
    if save_map:
        report["reid_map"] = _shown(write_reid_map(result))
    if measure:
        report["seconds_total"] = round(elapsed, 3)
        report["ms_per_document"] = round(1000 * elapsed / len(texts), 1)
        report["peak_python_memory_mb"] = round(peak / 2**20, 1)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="C4 end-to-end redaction (FR1, FR9)")
    ap.add_argument("--recording", help="a recording id in SORA_Dataset")
    ap.add_argument("--c2-dir", type=Path, help="directory holding <rid>.translation.json / .summary.json")
    ap.add_argument("--c1-dir", type=Path, help="optional directory holding <rid>.transcript.json")
    ap.add_argument("--rid", help="recording id when using --c2-dir")
    ap.add_argument("--model", default=None,
                    help="NER model: xlmr_both (transformer), both (spaCy), ...; 'none' = rules only. "
                         "Default: the transformer if trained, else spaCy")
    ap.add_argument("--no-map", action="store_true", help="do not write the re-identification map")
    ap.add_argument("--offline-check", action="store_true", help="block all network access (NFR1)")
    ap.add_argument("--measure", action="store_true", help="report latency and peak memory (NFR4/5)")
    args = ap.parse_args()

    rid = args.recording or args.rid
    if not rid:
        ap.error("give --recording, or --c2-dir with --rid")
    attempts = block_network() if args.offline_check else None
    texts = load_inputs(rid, args.c2_dir, args.c1_dir)
    from ner import default_model
    model = default_model() if args.model is None else None if args.model == "none" else args.model
    report = run(rid, texts, model,
                 save_map=not args.no_map, measure=args.measure)
    if attempts is not None:
        report["network_attempts"] = len(attempts)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
