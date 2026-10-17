"""
Step 2: split the corpus by recording into train / dev / test.

Writes data/train.jsonl, data/dev.jsonl and data/test.jsonl. One line per
recording, carrying everything both C2 tasks need:

    utterances    one per transcript line, with the code-mixed input
                  ("S1: ...") and the clean_en / clean_si targets (Task A)
    transcript    the whole conversation as "S1: ... / S2: ..." lines (Task B)
    summary_en, summary_si, action_items                          (Task B)

The split is by recording, never by utterance, so no conversation leaks
between train and test.

    python src/prepare_data.py
"""
import json
from pathlib import Path

from corpus import dataset_root, load_pairs, recording_ids

C2_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = C2_ROOT / "data"

# Same 5 recordings C4 holds out, so the full pipeline can later be scored on
# the same conversations. Do not look at these while developing.
TEST = {"J26DS313_R0011", "J26DS313_R0019", "J26DS313_R0021",
        "J26DS313_R0022", "J26DS313_R0023"}

# For tuning. A mix of romanized and Sinhala-script transcripts, short calls,
# multi-speaker meetings and one long podcast chunk.
DEV = {"J26DS313_R0007", "J26DS313_R0014", "J26DS313_R0033",
       "J26DS313_R0060", "J26DS313_R0102"}

# Deadline/owner values annotators used to mean "none".
EMPTY = {"", "n/a", "na", "none", "not specified", "null"}


def split_of(rid):
    if rid in TEST:
        return "test"
    if rid in DEV:
        return "dev"
    return "train"


def _clean(value):
    if value is None:
        return None
    value = str(value).strip()
    return None if value.lower() in EMPTY else value


def load_summary(rid):
    """(summary_en, summary_si, action_items) for one recording."""
    path = dataset_root() / "annotations" / "c2" / f"{rid}.summary.json"
    if not path.exists():
        return None, None, []
    data = json.loads(path.read_text(encoding="utf-8"))

    summaries = data.get("summaries")
    if isinstance(summaries, dict):
        texts = {k: (v.get("text") if isinstance(v, dict) else v) for k, v in summaries.items()}
    elif isinstance(summaries, list):
        texts = {s.get("lang", "en"): s.get("text") for s in summaries}
    else:
        texts = {"en": data.get("summary"), "si": data.get("summary_si")}

    items = []
    for raw in data.get("action_items") or []:
        item = {"intent": raw} if isinstance(raw, str) else dict(raw)
        intent = _clean(item.get("intent"))
        # "None — discussion only" placeholders mean there are no action items.
        if not intent or intent.lower().startswith("none"):
            continue
        items.append({
            "intent": intent,
            "owner": _clean(item.get("owner")),
            "receiver": _clean(item.get("receiver")),
            "deadline": _clean(item.get("deadline")),
        })
    return _clean(texts.get("en")), _clean(texts.get("si")), items


def build_record(rid):
    utterances = []
    for p in load_pairs(rid):
        if not p["text"].strip():
            continue
        speaker = p["speaker"] or "S?"
        utterances.append({
            "utt_id": p["utt_id"],
            "speaker": speaker,
            "src": f"{speaker}: {p['text'].strip()}",
            "clean_en": p["clean_en"],
            "clean_si": p["clean_si"],
        })
    summary_en, summary_si, items = load_summary(rid)
    return {
        "rid": rid,
        "split": split_of(rid),
        "utterances": utterances,
        "transcript": "\n".join(u["src"] for u in utterances),
        "summary_en": summary_en,
        "summary_si": summary_si,
        "action_items": items,
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    records = [build_record(rid) for rid in recording_ids()]

    missing = (TEST | DEV) - {r["rid"] for r in records}
    if missing:
        raise SystemExit(f"split recordings not found in the corpus: {sorted(missing)}")

    print(f"{'split':<6} {'recs':>5} {'utts':>6} {'en pairs':>9} {'si pairs':>9} {'summaries':>10} {'actions':>8}")
    for split in ("train", "dev", "test"):
        rows = [r for r in records if r["split"] == split]
        with open(OUT_DIR / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        utts = [u for r in rows for u in r["utterances"]]
        print(f"{split:<6} {len(rows):>5} {len(utts):>6}"
              f" {sum(1 for u in utts if u['clean_en']):>9}"
              f" {sum(1 for u in utts if u['clean_si']):>9}"
              f" {sum(1 for r in rows if r['summary_en']):>10}"
              f" {sum(len(r['action_items']) for r in rows):>8}")
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
