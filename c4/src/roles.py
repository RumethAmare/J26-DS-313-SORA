"""
Person-role classification (FR7): is a named person a PRIVATE_INDIVIDUAL (the
customer, a relative) or an ORGANISATION_REP (an agent giving a work name)?

Role never changes WHETHER a person is redacted -- both are -- only how the
placeholder is described, as the data contract requires. It is a required
schema field, not a novelty claim.

A person is classified once, from all of their mentions in a recording, by a
logistic-regression model over context clues:

  * how the name is given      agents give one work name; customers a full name
  * the words around it        "mama X, <bank> eken", "X speaking", "my name is"
  * an organisation nearby     "<ORG> call centre, X here"
  * when they first appear     agents open the call

    python roles.py train            # fit on complete real + synthetic train
    python roles.py evaluate         # entity-level accuracy on held-out recordings
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from functools import lru_cache

from corpus import (C4_ROOT, EVAL_RECORDINGS, complete_recording_ids, load_real,
                    load_synthetic)

ROLES = ("PRIVATE_INDIVIDUAL", "ORGANISATION_REP")
MODEL_PATH = C4_ROOT / "models" / "roles.joblib"
WINDOW = 6

_TOKEN = re.compile(r"[^\s.,;:!?()\[\]\"'’‘]+")
_UTT = re.compile(r"_u(\d+)_")
_HONORIFIC = re.compile(r"\b(mr|mrs|ms|miss|sir|madam|mahattaya|nona)\b|මහත්මයා|මහත්මිය|මිය", re.I)


def _tokens(text: str) -> list[str]:
    return [t.casefold() for t in _TOKEN.findall(text)]


def entity_features(mentions: list[dict]) -> dict:
    """
    mentions: [{"doc_id", "text", "start", "end", "surface"}, ...] for ONE person.
    Returns a sparse feature dict.
    """
    f: dict[str, float] = defaultdict(float)
    utts = []
    for m in mentions:
        text, s, e = m["text"], m["start"], m["end"]
        left = _tokens(text[max(0, s - 80):s])[-WINDOW:]
        right = _tokens(text[e:e + 80])[:WINDOW]
        for t in left:
            f[f"L={t}"] += 1
        for t in right:
            f[f"R={t}"] += 1
        if left:
            f[f"L1={left[-1]}"] += 1
        if right:
            f[f"R1={right[0]}"] += 1
        f[f"ntok={min(len(_tokens(m['surface'])), 3)}"] += 1
        if _HONORIFIC.search(text[max(0, s - 15):e + 15]):
            f["honorific"] += 1
        kind = ("summary" if "_summary_" in m["doc_id"] else
                "si" if m["doc_id"].endswith("_si_v1") else
                "en" if "_c2_" in m["doc_id"] else "transcript")
        f[f"doc={kind}"] += 1
        u = _UTT.search(m["doc_id"])
        if u:
            utts.append(int(u.group(1)))
    n = max(len(mentions), 1)
    out = {k: v / n for k, v in f.items()}     # per-mention rates
    out["log_mentions"] = min(n, 30) / 30
    if utts:
        out["first_utt"] = min(min(utts), 20) / 20
    return out


def gold_entities(recordings) -> tuple[list[dict], list[str]]:
    """Feature dicts and gold roles, one per annotated PERSON entity."""
    X, y = [], []
    for rec in recordings:
        by_ent = defaultdict(list)
        roles = {}
        for r in rec.rows:
            text = rec.texts.get(r["doc_id"])
            if r["label"] != "PERSON" or text is None or text[r["start_char"]:r["end_char"]] != r["surface"]:
                continue
            by_ent[r["entity_id"]].append({"doc_id": r["doc_id"], "text": text, "start": r["start_char"],
                                           "end": r["end_char"], "surface": r["surface"]})
            if r.get("role") in ROLES:
                roles[r["entity_id"]] = r["role"]
        for eid, mentions in by_ent.items():
            if eid in roles:
                X.append(entity_features(mentions))
                y.append(roles[eid])
    return X, y


def training_recordings():
    real = [load_real(r) for r in complete_recording_ids() if r not in EVAL_RECORDINGS]
    return real + load_synthetic("train")


def build_model():
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(DictVectorizer(), LogisticRegression(max_iter=2000, C=1.0,
                                                              class_weight="balanced"))


def train() -> None:
    import joblib
    X, y = gold_entities(training_recordings())
    model = build_model().fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"trained on {len(y)} people ({y.count(ROLES[0])} private, {y.count(ROLES[1])} rep) "
          f"-> {MODEL_PATH.relative_to(C4_ROOT)}")


@lru_cache(maxsize=1)
def _model():
    import joblib
    if not MODEL_PATH.exists():
        raise FileNotFoundError("no role model; run: python roles.py train")
    return joblib.load(MODEL_PATH)


def classify(mentions: list[dict]) -> str:
    """Role of one person from all of their mentions."""
    return _model().predict([entity_features(mentions)])[0]


def evaluate() -> dict:
    """Held-out entity-level accuracy, plus recording-grouped cross-validation on train."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    X_te, y_te = gold_entities([load_real(r) for r in sorted(EVAL_RECORDINGS)])
    pred = list(_model().predict(X_te))
    majority = max(ROLES, key=lambda r: gold_entities(training_recordings())[1].count(r))

    recs = [load_real(r) for r in complete_recording_ids() if r not in EVAL_RECORDINGS]
    X_tr, y_tr, groups = [], [], []
    for rec in recs:
        x, y = gold_entities([rec])
        X_tr += x; y_tr += y; groups += [rec.rid] * len(y)
    cv = cross_val_predict(build_model(), X_tr, y_tr, groups=groups, cv=GroupKFold(n_splits=5))
    return {
        "held_out": {"people": len(y_te), "accuracy": round(accuracy_score(y_te, pred), 4),
                     "macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
                     "majority_baseline_accuracy": round(sum(t == majority for t in y_te) / len(y_te), 4)},
        "cross_validation_real": {"people": len(y_tr), "accuracy": round(accuracy_score(y_tr, cv), 4),
                                  "macro_f1": round(f1_score(y_tr, cv, average="macro"), 4)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="C4 person-role classification (FR7)")
    ap.add_argument("cmd", choices=["train", "evaluate"])
    args = ap.parse_args()
    if args.cmd == "train":
        train()
    else:
        import json
        result = evaluate()
        print(json.dumps(result, indent=2))
        out = C4_ROOT / "eval" / "roles_real-eval.json"
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, indent=2)
        print(f"saved -> {out.relative_to(C4_ROOT)}")


if __name__ == "__main__":
    main()
