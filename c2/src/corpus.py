"""
Read-only loader for the shared SORA_Dataset corpus.

Pairs each code-mixed transcript utterance (C1) with its gold clean English and
clean Sinhala text (C2). Handles the different translation.json layouts found
in the corpus. Nothing is ever written to the dataset.

    SORA_DATASET_ROOT   overrides the dataset location
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def dataset_root():
    env = os.environ.get("SORA_DATASET_ROOT")
    if env:
        return Path(env)
    for candidate in (REPO_ROOT.parent / "SORA_Dataset",
                      REPO_ROOT.parent / "GitHub" / "SORA_Dataset"):
        if (candidate / "annotations").is_dir():
            return candidate
    raise FileNotFoundError("SORA_Dataset not found; set SORA_DATASET_ROOT.")


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def translation_records(data):
    """{"utterance_records": [...]} / {"utterances": [...]} / [...]"""
    if isinstance(data, list):
        return data
    return data.get("utterance_records") or data.get("utterances") or []


def recording_ids(root=None):
    """Recordings that have both a C1 transcript and a C2 translation."""
    ann = Path(root or dataset_root()) / "annotations"
    c1 = {p.name.removesuffix(".transcript.json") for p in (ann / "c1").glob("*.transcript.json")}
    c2 = {p.name.removesuffix(".translation.json") for p in (ann / "c2").glob("*.translation.json")}
    return sorted(c1 & c2)


def load_pairs(rid, root=None):
    """List of {utt_id, speaker, text, clean_en, clean_si} for one recording."""
    ann = Path(root or dataset_root()) / "annotations"
    transcript = _read_json(ann / "c1" / f"{rid}.transcript.json")["utterances"]
    gold = {r.get("utt_id"): r for r in
            translation_records(_read_json(ann / "c2" / f"{rid}.translation.json"))}
    pairs = []
    for u in transcript:
        g = gold.get(u["utt_id"], {})
        pairs.append({
            "utt_id": u["utt_id"],
            "speaker": u.get("speaker"),
            "text": u.get("text") or "",
            "clean_en": g.get("clean_en") or None,
            "clean_si": g.get("clean_si") or None,
        })
    return pairs


if __name__ == "__main__":
    ids = recording_ids()
    n_utts = n_missing = 0
    for rid in ids:
        pairs = load_pairs(rid)
        n_utts += len(pairs)
        n_missing += sum(1 for p in pairs if not p["clean_en"] or not p["clean_si"])
    print(f"dataset     : {dataset_root()}")
    print(f"recordings  : {len(ids)}")
    print(f"utterances  : {n_utts}")
    print(f"missing gold: {n_missing}")
