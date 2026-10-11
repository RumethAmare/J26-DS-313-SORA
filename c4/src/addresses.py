"""
Cross-script address linking (extends Contribution 2 from names to addresses).

"24, ගාලු පාර, දෙහිවල" and "24, Galle Road, Dehiwala" are one address. Left
unlinked they receive two placeholders, and a reader comparing the Sinhala
and English documents can tell the same house appears in both.

Addresses mix transliteration (ගාලු = Galle, දෙහිවල = Dehiwala) with
translation (පාර = Road), so they are compared by parts:

  1. house numbers must agree ("24" = "24", "24/3" = "24/3"); an address with
     a number never matches one without
  2. generic address words are dropped in both languages (road / පාර,
     street / වීදිය, lane / පටුමග, mawatha / මාවත, number / අංක ...)
  3. the remaining place-name words are compared phonetically, with the
     resolver's keys; every place word of the shorter address must match one
     in the longer address

    python addresses.py            # linking accuracy on the tuning recordings
"""
from __future__ import annotations

import re

from resolve import _key_similarity, phonetic_key

GENERIC = {
    # English and romanised
    "no", "number", "num", "road", "rd", "street", "st", "lane", "ln", "avenue", "ave",
    "mawatha", "mw", "mawathe", "place", "pl", "junction", "jn", "cross", "terrace",
    "gardens", "garden", "estate", "watta", "para", "pare", "district", "province", "sri",
    "lanka", "postal", "code", "house", "flat", "apartment", "floor", "unit", "the", "of",
    "slash", "dr", "doctor", "rev", "sir", "de", "da",
    # Sinhala
    "පාර", "පාරේ", "වීදිය", "පටුමග", "මාවත", "මාවතේ", "අංක", "අංකය", "නොම්බර", "නම්බර්",
    "පෙදෙස", "හන්දිය", "ගම", "වත්ත", "නිවස", "ගෙදර", "ලංකාව", "ශ්‍රී",
    # English address words written in Sinhala letters
    "ලේන්", "ලේන්එක", "රෝඩ්", "රෝඩ්එක", "ස්ට්‍රීට්", "ඇවනියු", "ප්ලේස්", "ජන්ෂන්", "නම්බර",
    "ස්ලෑෂ්", "ආචාර්ය", "වෛද්‍ය", "ද",
}
SIMILARITY = 0.8

# Translated, not transliterated: general Sinhala/English address vocabulary
# and the Sinhala names of Sri Lankan cities. General geography, not taken
# from any recording. Each Sinhala form maps to its English equivalent.
TRANSLATIONS = {
    "පන්සල": "temple", "පන්සල්": "temple", "විහාර": "vihara", "පල්ලිය": "church",
    "කෝවිල": "kovil", "පල්ලි": "mosque", "දුම්රියපොළ": "station", "ස්ටේෂන්": "station",
    "වැව": "lake", "වැව්": "lake", "රෝහල": "hospital", "රෝහල්": "hospital",
    "පාසල": "school", "පාසල්": "school", "වෙළඳපොළ": "market", "ප්‍රධාන": "main",
    "නව": "new", "පැරණි": "old", "මල්": "flower", "කඳු": "hill", "මුහුද": "sea",
    "මහනුවර": "kandy", "ගාල්ල": "galle", "යාපනය": "jaffna", "මීගමුව": "negombo",
    "හලාවත": "chilaw", "ත්‍රිකුණාමලය": "trincomalee", "මඩකලපුව": "batticaloa",
    "කොළඹ": "colombo", "ශ්‍රී ජයවර්ධනපුර": "kotte", "කෝට්ටේ": "kotte",
    "මහනුවරට": "kandy", "ගාල්ලේ": "galle",
    "ශාලාව": "hall", "ශාලා": "hall", "විශ්වවිද්‍යාලය": "university", "කර්මාන්තපුරය": "industrial",
    "කර්මාන්ත": "industrial", "කාර්යාලය": "office",
    # the same Sinhala words written in Latin letters
    "pansala": "temple", "palliya": "church", "kovila": "kovil", "wewa": "lake",
}
_TOKEN = re.compile(r"[^\s,.;:()\[\]/'\"]+")


def _address_key(text: str) -> str:
    # C before a vowel sounds k in place names (Colombo = කොළඹ); applied to
    # both sides, and only here, so the frozen name resolver is unchanged.
    return phonetic_key(re.sub(r"c(?=[aou])", "k", text.casefold()))


def _numbers(surface: str) -> tuple[frozenset, set]:
    """
    Number groups of an address, written any way: "14th" = "14", "05" = "5",
    "82/1" = "82න් 1" = {82, 1}, "අසූ අට" (spoken) = 88. Also returns the
    spoken-number words, so they are not taken for place names.
    """
    from numerals import find_number_word_runs
    text = surface.casefold()
    groups = set()
    spoken_words = set()
    for s, e, digits, _ in find_number_word_runs(text, min_digits=1):
        if " " not in text[s:e].strip() and text[s:e].isascii():
            continue   # one Latin word ("De" in "De Mel") is a name, not a number
        groups.add(str(int(digits)))
        spoken_words.update(text[s:e].split())
    for m in re.finditer(r"\d+(?:st|nd|rd|th|[a-z](?![a-z]))?", text):
        tok = re.sub(r"(st|nd|rd|th)$", "", m.group())
        digits = re.match(r"\d+", tok).group()
        groups.add(str(int(digits)) + tok[len(digits):])
    return frozenset(groups), spoken_words


def signature(surface: str) -> tuple[frozenset, tuple]:
    """(house-number groups, phonetic keys of place-name words)."""
    numbers, spoken = _numbers(surface)
    places = []
    for tok in _TOKEN.findall(surface.casefold()):
        if tok in GENERIC or tok in spoken or any(ch.isdigit() for ch in tok):
            continue
        key = _address_key(TRANSLATIONS.get(tok, tok))
        if len(key) >= 3:
            places.append(key)
    return numbers, tuple(places)


def same_address(a: str, b: str) -> bool:
    na, pa = signature(a)
    nb, pb = signature(b)
    if na != nb:
        return False
    if not pa or not pb:
        # "නොම්බර 23" said back for "No. 23, Temple Road": the house number
        # alone, which must then be a real number, not empty on both sides.
        return bool(na) and bool(pa or pb)
    short, long_ = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
    matched = [max(_key_similarity(s, l) for l in long_) >= SIMILARITY for s in short]
    # With the same house number, one shared place word is enough: two
    # different addresses in one call rarely share a number and a place.
    # Without a number, every place word must agree.
    return any(matched) if na else all(matched)


def part_of(a: str, b: str) -> bool:
    """
    Whether `a` is a shortened mention of `b`: "Nugegoda" or "Colombo 7" said
    again after "No. 82/1, Wijerama Road, ..., Nugegoda". Every number and
    every place word of `a` appears in `b`, and `a` names at least one place.
    """
    na, pa = signature(a)
    nb, pb = signature(b)
    if not pa or not na <= nb or len(pa) > len(pb):
        return False
    return all(max(_key_similarity(s, l) for l in pb) >= SIMILARITY for s in pa)


def cluster(surfaces: list[str]) -> list[int]:
    """
    Cluster number per surface. The most complete address seeds a cluster; a
    shortened mention joins only when it fits exactly one cluster, so "Colombo
    7" stays apart when two different Colombo 7 addresses were given.
    """
    sigs = [signature(s) for s in surfaces]
    order = sorted(range(len(surfaces)), key=lambda i: -(len(sigs[i][0]) + len(sigs[i][1])))
    seeds: list[list[int]] = []
    out = [0] * len(surfaces)
    for i in order:
        same = [c for c, members in enumerate(seeds)
                if any(same_address(surfaces[i], surfaces[j]) for j in members)]
        if not same:
            parts = [c for c, members in enumerate(seeds)
                     if any(part_of(surfaces[i], surfaces[j]) for j in members)]
            same = parts if len(parts) == 1 else []
        if same:
            seeds[same[0]].append(i)
            out[i] = same[0]
        else:
            seeds.append([i])
            out[i] = len(seeds) - 1
    return out


def evaluate(recordings) -> dict:
    """Linking of gold ADDRESS mentions: entities written in both scripts."""
    sinhala = re.compile(r"[඀-෿]")
    ok = total = tp = fp = fn = 0
    for rec in recordings:
        rows = [r for r in rec.rows if r["label"] == "ADDRESS"]
        if not rows:
            continue
        pred = cluster([r["surface"] for r in rows])
        gold = [r["entity_id"] for r in rows]
        for eid in set(gold):
            idx = [i for i, g in enumerate(gold) if g == eid]
            if len({bool(sinhala.search(rows[i]["surface"])) for i in idx}) < 2:
                continue
            total += 1
            clusters = {pred[i] for i in idx}
            intruders = [i for i, g in enumerate(gold) if pred[i] in clusters and g != eid]
            ok += len(clusters) == 1 and not intruders
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if bool(sinhala.search(rows[i]["surface"])) == bool(sinhala.search(rows[j]["surface"])):
                    continue
                same_g, same_p = gold[i] == gold[j], pred[i] == pred[j]
                tp += same_g and same_p
                fp += same_p and not same_g
                fn += same_g and not same_p
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"cross_script_addresses": total, "accuracy": round(ok / total, 4) if total else 0.0,
            "pair_precision": round(p, 4), "pair_recall": round(r, 4)}


if __name__ == "__main__":
    from corpus import EVAL_RECORDINGS, complete_recording_ids, load_real
    tuning = [load_real(r) for r in complete_recording_ids() if r not in EVAL_RECORDINGS]
    print("tuning recordings:", evaluate(tuning))
