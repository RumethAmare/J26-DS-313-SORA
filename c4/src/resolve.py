"""
Cross-script entity resolution (Contribution 2, SO5, FR6).

A person named in Latin script in one document and in Sinhala script in
another -- "Fernando" / "ප්‍රනාන්දු" -- must receive one entity_id, and so one
redaction placeholder, everywhere. The two forms share no characters, so
string comparison cannot associate them; redacting one and leaving the other
labelled differently lets a reader re-link them.

Method (no training data, fully offline):

  1. romanise   Sinhala script -> Latin, character by character, honouring
                the inherent vowel, al-lakuna (්) and conjuncts (්‍ර, ්‍ය)
                    ප්‍රනාන්දු -> pranaandu        නිමල් -> nimal
  2. key        one phonetic normalisation for both sides: f->p, v->w,
                aspirates and doubled letters collapsed, final vowel dropped
                    Fernando -> pernand       pranaandu -> pranand
  3. compare    token-level similarity: the better of the key and the
                consonant skeleton (p-r-n-n-d for both), so vowel spelling
                variation costs little
  4. cluster    within a recording, full names first; every mention joins the
                best-scoring entity at or above the threshold, otherwise it
                starts a new one

A linking error never causes a disclosure -- detection decides what is
redacted -- it only splits or merges placeholders. The threshold is tuned on
the synthetic train split only and evaluated on held-out names.

    python resolve.py --source synthetic-test
    python resolve.py --source real-dev --system translit-exact
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

# ---------------------------------------------------------------------------
# 1. Sinhala -> Latin romanisation
# ---------------------------------------------------------------------------

_CONSONANTS = {
    "ක": "k", "ඛ": "kh", "ග": "g", "ඝ": "gh", "ඞ": "ng", "ඟ": "ng",
    "ච": "ch", "ඡ": "ch", "ජ": "j", "ඣ": "jh", "ඤ": "gn", "ඥ": "gn", "ඦ": "nj",
    "ට": "t", "ඨ": "th", "ඩ": "d", "ඪ": "dh", "ණ": "n", "ඬ": "nd",
    "ත": "th", "ථ": "th", "ද": "d", "ධ": "dh", "න": "n", "ඳ": "nd",
    "ප": "p", "ඵ": "ph", "බ": "b", "භ": "bh", "ම": "m", "ඹ": "mb",
    "ය": "y", "ර": "r", "ල": "l", "ව": "w", "ශ": "sh", "ෂ": "sh", "ස": "s",
    "හ": "h", "ළ": "l", "ෆ": "f",
}
_INDEPENDENT_VOWELS = {
    "අ": "a", "ආ": "aa", "ඇ": "ae", "ඈ": "aee", "ඉ": "i", "ඊ": "ii", "උ": "u",
    "ඌ": "uu", "ඍ": "ru", "ඎ": "ruu", "එ": "e", "ඒ": "ee", "ඓ": "ai",
    "ඔ": "o", "ඕ": "oo", "ඖ": "au",
}
_VOWEL_SIGNS = {
    "ා": "aa", "ැ": "ae", "ෑ": "aee", "ි": "i", "ී": "ii", "ු": "u", "ූ": "uu",
    "ෘ": "ru", "ෲ": "ruu", "ෙ": "e", "ේ": "ee", "ෛ": "ai", "ො": "o", "ෝ": "oo",
    "ෞ": "au", "ෟ": "lu",
}
_AL_LAKUNA = "්"
_ANUSVARA = "ං"          # ං: nasal, "singha" in සිංහ
_VISARGA = "ඃ"
_JOINERS = {"‍", "‌"}   # ZWJ / ZWNJ inside conjuncts carry no sound

_SINHALA = re.compile(r"[඀-෿]")


def romanise(text: str) -> str:
    """Sinhala script to a Latin approximation; other characters pass through."""
    out: list[str] = []
    pending_vowel = False           # a consonant is waiting for its vowel
    for ch in text:
        if ch in _JOINERS:
            continue
        if ch in _VOWEL_SIGNS:
            out.append(_VOWEL_SIGNS[ch])
            pending_vowel = False
            continue
        if ch == _AL_LAKUNA:
            pending_vowel = False       # consonant with no vowel
            continue
        if pending_vowel:
            out.append("a")             # the inherent vowel
            pending_vowel = False
        if ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
            pending_vowel = True
        elif ch in _INDEPENDENT_VOWELS:
            out.append(_INDEPENDENT_VOWELS[ch])
        elif ch == _ANUSVARA:
            out.append("ng")
        elif ch == _VISARGA:
            out.append("h")
        else:
            out.append(ch)
    if pending_vowel:
        out.append("a")
    return "".join(out)


# ---------------------------------------------------------------------------
# 2. Phonetic key
# ---------------------------------------------------------------------------

# Honorifics and address terms: part of the annotated span (maximal span
# rule) but not part of the name.
HONORIFICS = {
    "mr", "mrs", "ms", "miss", "dr", "rev", "prof", "sir", "madam", "mahattaya",
    "mahaththaya", "nona", "mea", "maya", "aiya", "akka", "nangi", "malli",
    "මහත්මයා", "මහතා", "මහත්මිය", "මිය", "මෙනවිය", "මිස්", "මිසිස්", "මිස්ටර්", "ඩොක්ටර්",
    "සර්", "මැඩම්", "අයියා", "අක්කා", "නංගි", "මල්ලි",
}

_REWRITES = [
    ("ph", "p"), ("f", "p"), ("v", "w"), ("ck", "k"), ("q", "k"), ("x", "ks"),
    ("z", "s"), ("th", "t"), ("dh", "d"), ("kh", "k"), ("gh", "g"), ("bh", "b"),
    ("jh", "j"), ("sh", "s"), ("ae", "e"),
]


def name_tokens(surface: str) -> list[str]:
    """Name tokens of a mention, honorifics removed, in original script."""
    # Split on space and punctuation only: \w does not match Sinhala vowel
    # signs or al-lakuna, so a \w+ tokeniser cuts every Sinhala name apart.
    tokens = re.findall(r"[^\s.,;:!?()\[\]\"'’‘/-]+", surface.casefold())
    return [t for t in tokens if t not in HONORIFICS and not t.isdigit()]


def phonetic_key(token: str) -> str:
    """One normalisation for Latin and romanised Sinhala alike."""
    latin = romanise(token) if _SINHALA.search(token) else token
    latin = unicodedata.normalize("NFKD", latin.casefold())
    latin = re.sub(r"[^a-z]", "", latin)
    for a, b in _REWRITES:
        latin = latin.replace(a, b)
    latin = re.sub(r"(.)\1+", r"\1", latin)          # doubled letters / long vowels
    if len(latin) > 3:
        # Final a/e/o/u vary freely (-singhe / -singha). Final -i does not: it
        # is how Sinhala forms a woman's name (Nimal / Nimali), so it is kept.
        latin = re.sub(r"y$", "i", latin)        # Selvy = සෙල්වී: final -y is -i
        latin = re.sub(r"[aeou]+$", "", latin)
    return latin


def skeleton(key: str) -> str:
    return re.sub(r"[aeiouy]", "", key)


# ---------------------------------------------------------------------------
# 3. Similarity
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1 - levenshtein(a, b) / max(len(a), len(b))


# Case endings that attach to a Sinhala name in speech: අමාශිට "to Amashi",
# සෙල්වීගෙන් "from Selvy", නිමල්ගේ "Nimal's". The annotated span covers the
# whole word, so the name is compared with and without them.
SINHALA_CASE_ENDINGS = ("ගෙන්ම", "ගෙන්", "ගෙන", "ටත්", "ගේ", "ගෙ", "ට", "ත්", "ව", "ද", "යි", "ලා")


def _variants(token: str) -> set[str]:
    forms = {token}
    if _SINHALA.search(token):
        for ending in SINHALA_CASE_ENDINGS:
            if token.endswith(ending) and len(token) > len(ending) + 1:
                forms.add(token[: -len(ending)])
    return forms


def _key_similarity(ka: str, kb: str) -> float:
    # Nimal / Nimali, Mahesh / Maheshi, Chamara / Chamari: a man and a woman,
    # not two spellings of one name.
    if ka + "i" == kb or kb + "i" == ka:
        return 0.5
    score = _ratio(ka, kb)
    sa, sb = skeleton(ka), skeleton(kb)
    # The consonant skeleton carries the name; it is only trusted when long
    # enough not to match by accident ("kml" vs "kmr").
    if min(len(sa), len(sb)) >= 3:
        score = max(score, _ratio(sa, sb))
    return score


def token_similarity(a: str, b: str) -> float:
    return max(_key_similarity(phonetic_key(x), phonetic_key(y))
               for x in _variants(a) for y in _variants(b))


def mention_similarity(a: str, b: str) -> float:
    """
    Every token of the shorter mention must find a partner in the longer one:
    "Kasun" matches "කසුන් වික්‍රමසිංහ", "Mr. Perera" matches "Nimal Perera".
    """
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return 0.0
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    # The weakest token decides: "Kamal Perera" vs "Nimal Perera" share a
    # surname, and averaging would let it carry the different first name.
    return min(max(token_similarity(s, t) for t in long_) for s in short)


# ---------------------------------------------------------------------------
# 4. Clustering
# ---------------------------------------------------------------------------

# Tuned on the synthetic TRAIN split only (see tune_threshold); frozen here.
DEFAULT_THRESHOLD = 0.775


@dataclass
class Mention:
    mid: str            # unique id within the recording
    surface: str

    @property
    def script(self) -> str:
        return "SINHALA" if _SINHALA.search(self.surface) else "LATIN"


def resolve(mentions: list[Mention], threshold: float = DEFAULT_THRESHOLD,
            system: str = "resolver") -> dict[str, int]:
    """
    Assign a cluster number to every mention of one recording.

    system  "resolver"        transliteration + fuzzy similarity (this work)
            "translit-exact"  transliteration + identical phonetic key
            "exact"           identical surface, case-insensitive (no resolution)
    """
    if system == "exact":
        keys: dict[str, int] = {}
        return {m.mid: keys.setdefault(" ".join(name_tokens(m.surface)), len(keys))
                for m in mentions}
    if system == "translit-exact":
        keys = {}
        return {m.mid: keys.setdefault(
                    " ".join(phonetic_key(t) for t in name_tokens(m.surface)), len(keys))
                for m in mentions}

    # Full names first: they define the entities, short mentions attach to them.
    ordered = sorted(mentions, key=lambda m: -len(name_tokens(m.surface)))
    clusters: list[list[Mention]] = []
    assignment: dict[str, int] = {}
    for m in ordered:
        best, best_score = None, threshold
        for i, members in enumerate(clusters):
            score = max(mention_similarity(m.surface, o.surface) for o in members)
            # Strictly better only: on a tie ("Mr. Perera" vs two Pereras) the
            # entity introduced first -- usually the caller -- keeps it.
            if score > best_score or (best is None and score == best_score):
                best, best_score = i, score
        if best is None:
            clusters.append([m])
            best = len(clusters) - 1
        else:
            clusters[best].append(m)
        assignment[m.mid] = best
    return assignment


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def person_mentions(rec) -> tuple[list[Mention], dict[str, str]]:
    """Gold PERSON mentions of a recording and their gold entity ids."""
    mentions, gold = [], {}
    for row in rec.rows:
        if row["label"] == "PERSON":
            mid = row["ann_id"]
            mentions.append(Mention(mid, row["surface"]))
            gold[mid] = row["entity_id"]
    return mentions, gold


def score_linking(recordings: Iterable, threshold: float = DEFAULT_THRESHOLD,
                  system: str = "resolver") -> dict:
    """
    cross_script_accuracy   of the gold entities named in BOTH scripts, the
                            share whose mentions all land in one predicted
                            cluster containing no other entity -- one
                            entity, one placeholder, nothing merged with it
    pairwise                precision / recall / F1 of "same entity" decisions
                            over Latin-Sinhala mention pairs
    """
    ok = total = 0
    tp = fp = fn = 0
    failures = []
    for rec in recordings:
        mentions, gold = person_mentions(rec)
        if not mentions:
            continue
        pred = resolve(mentions, threshold, system)
        by_script = {m.mid: m.script for m in mentions}

        members: dict[str, set] = defaultdict(set)
        for mid, eid in gold.items():
            members[eid].add(mid)
        for eid, mids in members.items():
            if {by_script[m] for m in mids} != {"LATIN", "SINHALA"}:
                continue
            total += 1
            clusters = {pred[m] for m in mids}
            intruders = [m for m in pred if pred[m] in clusters and gold[m] != eid]
            if len(clusters) == 1 and not intruders:
                ok += 1
            else:
                failures.append({"recording": rec.rid, "entity": eid,
                                 "clusters": len(clusters), "merged_with": len(intruders)})

        for a, b in combinations(mentions, 2):
            if a.script == b.script:
                continue
            same_gold = gold[a.mid] == gold[b.mid]
            same_pred = pred[a.mid] == pred[b.mid]
            tp += same_gold and same_pred
            fp += same_pred and not same_gold
            fn += same_gold and not same_pred

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "system": system,
        "threshold": threshold,
        "cross_script_accuracy": round(ok / total, 4) if total else 0.0,
        "entities": total,
        "pairwise": {"precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0},
        "failures": failures,
    }


def tune_threshold(recordings: list, grid: Iterable[float] | None = None) -> float:
    """Pick the threshold with the best cross-script accuracy on `recordings`."""
    grid = grid or [round(0.50 + 0.025 * i, 3) for i in range(19)]
    # Ties go to the LOWEST threshold: equally accurate here, and the most
    # tolerant of the spelling variation real transcripts add.
    return max(grid, key=lambda t: (score_linking(recordings, t)["cross_script_accuracy"], -t))


def main() -> None:
    from corpus import EVAL_RECORDINGS, load_real, load_synthetic, real_recording_ids

    ap = argparse.ArgumentParser(description="C4 cross-script entity resolution")
    ap.add_argument("--source", default="synthetic-test",
                    choices=["synthetic-train", "synthetic-test", "real-dev", "real-eval"])
    ap.add_argument("--system", default="all",
                    choices=["all", "resolver", "translit-exact", "exact"])
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--save", action="store_true", help="write results to eval/")
    ap.add_argument("--tune", action="store_true",
                    help="report the best threshold on the synthetic train split")
    args = ap.parse_args()

    if args.tune:
        best = tune_threshold(load_synthetic("train"))
        print(f"best threshold on synthetic-train: {best}")
        return

    if args.source.startswith("synthetic-"):
        recs = load_synthetic(args.source.removeprefix("synthetic-"))
    else:
        ids = [r for r in real_recording_ids()
               if (r in EVAL_RECORDINGS) == (args.source == "real-eval")]
        recs = [load_real(r) for r in ids]

    systems = ["exact", "translit-exact", "resolver"] if args.system == "all" else [args.system]
    print(f"cross-script linking  source={args.source}  threshold={args.threshold}\n")
    print(f"  {'system':<16} {'accuracy':>9} {'entities':>9} {'pair P':>7} {'pair R':>7} {'pair F1':>8}")
    results = []
    for system in systems:
        s = score_linking(recs, args.threshold, system)
        results.append(s)
        pw = s["pairwise"]
        print(f"  {system:<16} {s['cross_script_accuracy']:>9.3f} {s['entities']:>9} "
              f"{pw['precision']:>7.3f} {pw['recall']:>7.3f} {pw['f1']:>8.3f}")
    if args.save:
        import json
        from corpus import C4_ROOT
        out = C4_ROOT / "eval" / f"linking_{args.source}.json"
        out.parent.mkdir(exist_ok=True)
        # Failures carry recording and entity ids only -- no surfaces.
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"source": args.source, "recordings": [r.rid for r in recs],
                       "results": results}, f, indent=2)
        print(f"\nsaved -> {out.relative_to(C4_ROOT)}")


if __name__ == "__main__":
    main()
