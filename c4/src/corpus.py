"""
Read-only access to the shared SORA_Dataset corpus, and to the synthetic set.

Resolves every doc_id used by the C4 annotation layer to the exact text its
offsets index into (PII_SCHEMA.md §7):

    <rid>_transcript_<utt>_v1   annotations/c1/<rid>.transcript.json  utterances[].text
    <rid>_c2_<utt>_en_v1        annotations/c2/<rid>.translation.json clean_en
    <rid>_c2_<utt>_si_v1        annotations/c2/<rid>.translation.json clean_si
    <rid>_c2_summary_en_v1      annotations/c2/<rid>.summary.json     summaries.en.text

Nothing is ever written to the dataset repository, and nothing read from it is
copied into this one (.gitignore, NFR2).

    SORA_DATASET_ROOT   default ../../SORA_Dataset relative to this component
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from validate import normalise_label

C4_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = C4_ROOT / "data" / "synthetic"

# Whole recordings held out for final evaluation. Identical to the set the
# 0.231 baseline was scored on (SORA_Dataset scripts/c4/prepare_data.py), so
# results are directly comparable. Rules and thresholds are tuned on the other
# recordings only -- never on these.
EVAL_RECORDINGS = frozenset({
    "J26DS313_R0011", "J26DS313_R0019", "J26DS313_R0021",
    "J26DS313_R0022", "J26DS313_R0023",
})


def dataset_root() -> Path:
    return Path(os.environ.get("SORA_DATASET_ROOT", C4_ROOT.parents[1] / "SORA_Dataset"))


@dataclass
class Recording:
    rid: str
    texts: dict[str, str]          # doc_id -> text
    rows: list[dict]               # pii rows, labels normalised (§6)
    registry: dict | None


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _utt_suffix(utt_id: str, rid: str) -> str:
    """'J26DS313_R0002_u001' -> 'u001'."""
    return utt_id[len(rid) + 1 :] if utt_id.startswith(rid + "_") else utt_id


def load_texts(rid: str, root: Path | None = None) -> dict[str, str]:
    ann = (root or dataset_root()) / "annotations"
    texts: dict[str, str] = {}

    transcript = ann / "c1" / f"{rid}.transcript.json"
    if transcript.exists():
        for u in _read_json(transcript)["utterances"]:
            texts[f"{rid}_transcript_{_utt_suffix(u['utt_id'], rid)}_v1"] = u["text"]

    translation = ann / "c2" / f"{rid}.translation.json"
    if translation.exists():
        for u in _read_json(translation)["utterance_records"]:
            utt = _utt_suffix(u["utt_id"], rid)
            texts[f"{rid}_c2_{utt}_en_v1"] = u.get("clean_en") or ""
            texts[f"{rid}_c2_{utt}_si_v1"] = u.get("clean_si") or ""

    summary = ann / "c2" / f"{rid}.summary.json"
    if summary.exists():
        summaries = _read_json(summary).get("summaries", {})
        for lang in ("en", "si"):
            if summaries.get(lang, {}).get("text"):
                texts[f"{rid}_c2_summary_{lang}_v1"] = summaries[lang]["text"]
    return texts


_UTT_DOC = re.compile(r"^(?P<stream>.+?_(?:transcript|c2))_u(?P<n>\d+)(?P<tail>_(?:en_|si_)?v\d+)$")


def preceding_texts(texts: dict[str, str]) -> dict[str, str]:
    """
    doc_id -> text of the previous utterance in the same stream (transcript,
    clean_en or clean_si). Summaries and first utterances map to "".
    """
    streams: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for doc_id in texts:
        m = _UTT_DOC.match(doc_id)
        if m:
            streams.setdefault((m["stream"], m["tail"]), []).append((int(m["n"]), doc_id))
    previous = {doc_id: "" for doc_id in texts}
    for docs in streams.values():
        docs.sort()
        for (_, before), (_, doc_id) in zip(docs, docs[1:]):
            previous[doc_id] = texts[before]
    return previous


def real_recording_ids(root: Path | None = None) -> list[str]:
    c4 = (root or dataset_root()) / "annotations" / "c4"
    return sorted(p.name.removesuffix(".pii.jsonl") for p in c4.glob("*.pii.jsonl"))


def load_real(rid: str, root: Path | None = None) -> Recording:
    c4 = (root or dataset_root()) / "annotations" / "c4"
    registry_path = c4 / f"{rid}.entities.json"
    return Recording(
        rid=rid,
        texts=load_texts(rid, root),
        rows=[normalise_label(r) for r in _read_jsonl(c4 / f"{rid}.pii.jsonl")],
        registry=_read_json(registry_path) if registry_path.exists() else None,
    )


def load_synthetic(directory: Path = SYNTHETIC_DIR) -> list[Recording]:
    recordings = []
    for pii in sorted(directory.glob("*.pii.jsonl")):
        rid = pii.name.removesuffix(".pii.jsonl")
        docs = _read_jsonl(directory / f"{rid}.docs.jsonl")
        recordings.append(Recording(
            rid=rid,
            texts={d["doc_id"]: d["text"] for d in docs},
            rows=_read_jsonl(pii),
            registry=_read_json(directory / f"{rid}.entities.json"),
        ))
    return recordings
