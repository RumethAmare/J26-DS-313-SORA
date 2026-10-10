"""
Redaction and re-identification (FR6, FR8, NFR2).

Turns detections into redacted text. Every mention of one real-world entity
gets the same placeholder in every document of a recording -- transcript,
clean_en, clean_si and summary -- so a reader cannot re-link mentions that the
redaction was meant to separate, and an authorised reader sees consistent
labels ([PERSON_1] is the same person everywhere):

    mage NIC eka 953201456V, number eka 0 7 7 1 2 3 4 5 6 7
    mage NIC eka [NIC_1], number eka [PHONE_1]

Three steps:

  1. detect     run the detector on every document
  2. resolve    group mentions into entities by normalised value, so a spoken
                "0 7 7 1 ..." and a written "0771234567" are one [PHONE_1]
  3. propagate  once an entity is known, redact every other exact occurrence
                of its surface forms in the recording, even where the detector
                missed it -- recall is the primary metric, and a value found
                once should never survive elsewhere

The re-identification map records every replacement exactly, so restore()
reverses redaction byte-for-byte (FR8). It is personal data: write_reid_map
refuses any location git would commit (NFR2).

ORG and LOCATION, and identifiers owned by an organisation (a hotline), are
detected but left visible -- they are not personal data.

    python redact.py --text "mage NIC eka 953201456V, number eka 0771234567"
    python redact.py --synthetic SYN_T0003
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from corpus import C4_ROOT, load_synthetic, preceding_texts
from rules import Detection
from rules import detect as rule_detect

NOT_REDACTED = {"ORG", "LOCATION"}
REID_DIR = C4_ROOT / "reid_map"

# Surfaces shorter than this are not propagated: "12" (an extension) or a
# short name would match inside unrelated text.
MIN_PROPAGATE_LEN = 6

Detector = Callable[[str, str], list]


def should_redact(d) -> bool:
    return d.label not in NOT_REDACTED and getattr(d, "role", "") != "ORGANISATION"


def entity_key(d) -> tuple[str, str]:
    """
    Two mentions with the same key are the same entity.

    A resolver may assign `entity_id` directly (cross-script names). Otherwise
    identifiers are keyed by their normalised value, so every spoken and
    written form of one number collapses to one entity. An NIC is keyed
    without its V/X, which speakers often drop.
    """
    if getattr(d, "entity_id", None):
        return d.label, d.entity_id
    value = getattr(d, "value", "") or d.surface
    if d.label in ("NIC", "PHONE", "ACCOUNT"):
        value = re.sub(r"[\s\-/]", "", value).upper()
        if d.label == "NIC":
            value = value.rstrip("VX")
    else:
        value = " ".join(value.casefold().split())
    return d.label, value


# ---------------------------------------------------------------------------
# Detection over a whole recording
# ---------------------------------------------------------------------------


def _overlaps(start: int, end: int, spans: list) -> bool:
    return any(start < s.end and s.start < end for s in spans)


def _boundary_ok(text: str, start: int, end: int) -> bool:
    """
    An occurrence must not sit inside a longer word or number -- except that a
    Sinhala name may carry a case ending (නිමල්ට "to Nimal"): the name is still
    the name, and is redacted with the ending kept ("[PERSON_1]ට").
    """
    from resolve import SINHALA_CASE_ENDINGS

    before = text[start - 1] if start else " "
    if before.isalnum():
        return False
    rest = text[end:]
    if not rest or not rest[0].isalnum():
        return True
    for ending in SINHALA_CASE_ENDINGS:
        if rest.startswith(ending):
            after = rest[len(ending):len(ending) + 1]
            if not after or not after.isalnum():
                return True
    return False


def detect_recording(texts: dict[str, str], detect: Detector = rule_detect,
                     propagate: bool = True) -> dict[str, list]:
    """doc_id -> detections, with cross-document propagation of known surfaces."""
    previous = preceding_texts(texts)
    found = {doc_id: list(detect(text, previous[doc_id])) for doc_id, text in texts.items()}
    if not propagate:
        return found

    # Every surface form each redactable entity was seen with, anywhere.
    forms: dict[tuple, Detection] = {}
    for dets in found.values():
        for d in dets:
            if should_redact(d) and len(d.surface.strip()) >= MIN_PROPAGATE_LEN:
                forms.setdefault((d.surface, d.label), d)

    for doc_id, text in texts.items():
        spans = list(found[doc_id])
        added = []
        # Longest first, so "0771234567 extension 12" wins over "0771234567".
        for (surface, _), d in sorted(forms.items(), key=lambda kv: -len(kv[0][0])):
            start = text.find(surface)
            while start != -1:
                end = start + len(surface)
                if _boundary_ok(text, start, end):
                    clash = [s for s in spans + added if start < s.end and s.start < end]
                    # A known personal identifier outranks a "keep visible"
                    # label on the same words: the model calling a customer's
                    # name ORG in one sentence must not leave it exposed there.
                    if not clash or not any(should_redact(s) for s in clash):
                        spans = [s for s in spans if s not in clash]
                        added = [s for s in added if s not in clash]
                        added.append(replace(d, start=start, end=end, source="propagated"))
                start = text.find(surface, start + 1)
        found[doc_id] = sorted(spans + added, key=lambda x: x.start)
    return propagate_names(found, texts)


# A name token must be at least this long (phonetic key) to be propagated in
# another form, and match a known name token this closely: "Nimal" -> "නිමල්"
# / "nimal" / "නිමල්ට", but not a short common word that happens to be similar.
NAME_KEY_MIN_LEN = 4
NAME_TOKEN_SIMILARITY = 0.9
_NAME_TOKEN = re.compile(r"[^\s.,;:!?()\[\]\"'’‘/]+")


def propagate_names(found: dict[str, list], texts: dict[str, str]) -> dict[str, list]:
    """
    Cross-script name propagation. A person found once is looked for in every
    document in every form the resolver treats as the same name: another case
    ("nimal"), the other script (නිමල්), or with a Sinhala case ending
    (නිමල්ට). Consecutive matching tokens become one span ("නිමල් පෙරේරා").

    Exact-surface propagation cannot do this: the Sinhala and Latin forms of a
    name share no characters, which is the gap Contribution 2 addresses.
    """
    from resolve import HONORIFICS, NAME_PARTICLES, _variants, phonetic_key

    keys: dict[str, Detection] = {}
    for dets in found.values():
        for d in dets:
            if d.label == "PERSON":
                for tok in _NAME_TOKEN.findall(d.surface.casefold()):
                    if tok not in HONORIFICS and tok not in NAME_PARTICLES:
                        k = phonetic_key(tok)
                        if len(k) >= NAME_KEY_MIN_LEN:
                            keys.setdefault(k, d)
    if not keys:
        return found

    def match(token: str):
        """(known detection, length of the name part of the token) or None."""
        for form in sorted(_variants(token.casefold()), key=len, reverse=True):
            k = phonetic_key(form)
            if len(k) < NAME_KEY_MIN_LEN:
                continue
            for known, d in keys.items():
                if k == known or _close(k, known):
                    # The case ending stays visible: "[PERSON_1]ගේ".
                    return d, len(token) - (len(token.casefold()) - len(form))
        return None

    for doc_id, text in texts.items():
        spans = list(found[doc_id])
        # Proper English capitalises names; Singlish transcripts often do not.
        lowercase_ok = "_transcript_" in doc_id
        hits = []
        for m in _NAME_TOKEN.finditer(text):
            s, e = m.start(), m.end()
            token = m.group()
            if any(s < x.end and x.start < e for x in spans):
                continue
            if not lowercase_ok and token[:1].islower():
                continue
            hit = match(token)
            if hit is None:
                continue
            d, length = hit
            e = s + length
            # Join neighbouring name tokens into one span: "නිමල් පෙරේරා".
            if hits and text[hits[-1][1]:s].strip() == "":
                hits[-1] = (hits[-1][0], e, hits[-1][2])
            else:
                hits.append((s, e, d))
        for s, e, d in hits:
            spans.append(replace(d, start=s, end=e, surface=text[s:e], source="name-propagated"))
        found[doc_id] = sorted(spans, key=lambda x: x.start)
    return found


def _close(key: str, known: str) -> bool:
    """
    Same name in another script or spelling. A consonant-skeleton match alone
    is too loose for a single word (කැරට් "carrot" shares k-r-t with a name),
    so it also needs a long skeleton and a reasonably close full spelling.
    """
    from resolve import _ratio, skeleton

    if _ratio(key, known) >= NAME_TOKEN_SIMILARITY:
        return True
    sk, sn = skeleton(key), skeleton(known)
    return min(len(sk), len(sn)) >= 4 and _ratio(sk, sn) >= 1.0 and _ratio(key, known) >= 0.6


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


@dataclass
class Redaction:
    rid: str
    texts: dict[str, str]                  # doc_id -> redacted text (shareable)
    entities: list[dict]                   # placeholder, label, role, mentions
    reid: dict[str, list[dict]]            # doc_id -> replacements (personal data)
    leaks: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        """What may be shared: no values, no surfaces."""
        return {
            "recording": self.rid,
            "entities": [{k: e[k] for k in ("placeholder", "label", "role")}
                         | {"mentions": len(e["mentions"])} for e in self.entities],
            "leaks": len(self.leaks),
        }


def link_names(detections: dict[str, list]) -> dict[str, list]:
    """
    Cross-script resolution (Contribution 2): give every PERSON mention of one
    individual the same entity_id, so "Nimal Perera" and "නිමල් පෙරේරා" share
    one placeholder.
    """
    from resolve import Mention, resolve

    mentions, where = [], {}
    for doc_id, dets in detections.items():
        for i, d in enumerate(dets):
            if d.label == "PERSON":
                mid = f"{doc_id}#{i}"
                mentions.append(Mention(mid, d.surface))
                where[mid] = (doc_id, i)
    if not mentions:
        return detections
    linked = {doc_id: list(dets) for doc_id, dets in detections.items()}
    for mid, cluster in resolve(mentions).items():
        doc_id, i = where[mid]
        linked[doc_id][i] = replace(linked[doc_id][i], entity_id=f"person{cluster}")
    return linked


def assign_roles(detections: dict[str, list], texts: dict[str, str]) -> dict[str, list]:
    """
    FR7: classify each linked person once, from all of their mentions, and
    give every mention that role. Without a trained role model, roles are
    left as detected.
    """
    from roles import MODEL_PATH, classify
    if not MODEL_PATH.exists():
        return detections
    people: dict[str, list] = defaultdict(list)
    for doc_id, dets in detections.items():
        for i, d in enumerate(dets):
            if d.label == "PERSON":
                people[d.entity_id or f"{doc_id}#{i}"].append((doc_id, i, d))
    out = {doc_id: list(dets) for doc_id, dets in detections.items()}
    for mentions in people.values():
        role = classify([{"doc_id": doc_id, "text": texts[doc_id], "start": d.start,
                          "end": d.end, "surface": d.surface} for doc_id, _, d in mentions])
        for doc_id, i, d in mentions:
            out[doc_id][i] = replace(d, role=role)
    return out


def redact_recording(texts: dict[str, str], rid: str = "", detect: Detector = rule_detect,
                     propagate: bool = True, resolve_names: bool = True,
                     classify_roles: bool = True) -> Redaction:
    detections = detect_recording(texts, detect, propagate)
    if resolve_names:
        detections = link_names(detections)
    if classify_roles:
        detections = assign_roles(detections, texts)
    order = {doc_id: i for i, doc_id in enumerate(texts)}

    # Group mentions into entities; number placeholders by first appearance.
    groups: dict[tuple, list] = defaultdict(list)
    for doc_id, dets in detections.items():
        for d in dets:
            if should_redact(d):
                groups[entity_key(d)].append((doc_id, d))
    first_seen = sorted(groups, key=lambda k: min((order[doc], d.start) for doc, d in groups[k]))

    counters: dict[str, int] = defaultdict(int)
    placeholder_of: dict[tuple, str] = {}
    entities = []
    for key in first_seen:
        label = key[0]
        counters[label] += 1
        placeholder = f"[{label}_{counters[label]}]"
        placeholder_of[key] = placeholder
        mentions = groups[key]
        entities.append({
            "placeholder": placeholder,
            "label": label,
            "role": getattr(mentions[0][1], "role", ""),
            "value": getattr(mentions[0][1], "value", "") or mentions[0][1].surface,
            "mentions": [{"doc_id": doc, "start": d.start, "end": d.end,
                          "surface": d.surface, "source": getattr(d, "source", "")}
                         for doc, d in mentions],
        })

    redacted, reid = {}, {}
    for doc_id, text in texts.items():
        out, replacements, pos = [], [], 0
        cursor = 0
        for d in detections[doc_id]:
            if not should_redact(d):
                continue
            placeholder = placeholder_of[entity_key(d)]
            out.append(text[cursor:d.start])
            pos += d.start - cursor
            replacements.append({"placeholder": placeholder, "start": pos,
                                 "end": pos + len(placeholder), "original": d.surface})
            out.append(placeholder)
            pos += len(placeholder)
            cursor = d.end
        out.append(text[cursor:])
        redacted[doc_id] = "".join(out)
        if replacements:
            reid[doc_id] = replacements

    result = Redaction(rid, redacted, entities, reid)
    result.leaks = leak_check(result)
    return result


def restore(redacted: dict[str, str], reid: dict[str, list[dict]]) -> dict[str, str]:
    """Reverse redaction exactly, using the re-identification map (FR8)."""
    restored = {}
    for doc_id, text in redacted.items():
        for r in sorted(reid.get(doc_id, []), key=lambda r: -r["start"]):
            assert text[r["start"]:r["end"]] == r["placeholder"], "map does not fit text"
            text = text[:r["start"]] + r["original"] + text[r["end"]:]
        restored[doc_id] = text
    return restored


def leak_check(result: Redaction) -> list[dict]:
    """
    Every known surface form and normalised value of a redacted entity must be
    absent from every redacted document. Digit values are also searched with
    spacing removed, so an undetected "077 123 4567" is still caught.
    """
    leaks = []
    for e in result.entities:
        needles = {m["surface"] for m in e["mentions"] if len(m["surface"]) >= MIN_PROPAGATE_LEN}
        digits = re.sub(r"\D", "", e["value"])
        for doc_id, text in result.texts.items():
            squeezed = re.sub(r"[\s\-]", "", text)
            if any(n in text for n in needles) or (len(digits) >= 7 and digits in squeezed):
                leaks.append({"placeholder": e["placeholder"], "doc_id": doc_id})
    return leaks


# ---------------------------------------------------------------------------
# Re-identification map storage (NFR2)
# ---------------------------------------------------------------------------


def is_git_ignored(path: Path) -> bool:
    try:
        return subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=C4_ROOT,
                              capture_output=True).returncode == 0
    except OSError:
        return False


def write_reid_map(result: Redaction, directory: Path = REID_DIR) -> Path:
    """
    Store the map where git will never commit it. The map reverses redaction,
    so committing one would publish exactly the data this component removes --
    and git history is permanent. Refuses any path .gitignore does not cover.
    """
    path = directory / f"{result.rid or 'text'}.reid.json"
    if not is_git_ignored(path):
        raise PermissionError(f"refusing to write a re-identification map to {path}: "
                              f"not git-ignored (NFR2)")
    directory.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"recording": result.rid, "entities": result.entities,
                   "replacements": result.reid}, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _print_side_by_side(texts: dict[str, str], result: Redaction, kinds: tuple) -> None:
    for doc_id, text in texts.items():
        if not any(k in doc_id for k in kinds):
            continue
        print(f"{doc_id}")
        print(f"  original : {text}")
        print(f"  redacted : {result.texts[doc_id]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="C4 redaction demo")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="redact one piece of text")
    src.add_argument("--synthetic", metavar="RID", help="a synthetic recording, e.g. SYN_T0003")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--all-docs", action="store_true",
                    help="show clean_en, clean_si and summary too, not just the transcript")
    ap.add_argument("--model", metavar="DATA", choices=["real", "synthetic", "both"],
                    help="also detect names/addresses with the trained model (hybrid)")
    ap.add_argument("--save-map", action="store_true",
                    help="write the re-identification map to reid_map/ (git-ignored)")
    args = ap.parse_args()

    if args.text:
        texts, rid = {"text": args.text}, "text"
    else:
        recs = {r.rid: r for r in load_synthetic(args.split)}
        if args.synthetic not in recs:
            sys.exit(f"no synthetic recording {args.synthetic} in split {args.split}")
        texts, rid = recs[args.synthetic].texts, args.synthetic

    detector = rule_detect
    if args.model:
        from ner import HybridDetector
        detector = HybridDetector(args.model)
    result = redact_recording(texts, rid, detect=detector)
    kinds = ("",) if args.all_docs or args.text else ("_transcript_", "_summary_")
    _print_side_by_side(texts, result, kinds)

    print("\nentities")
    for e in result.entities:
        print(f"  {e['placeholder']:<12} {e['label']:<8} {len(e['mentions'])} mention(s)")
    print(f"\nleak check: {'PASS' if not result.leaks else f'{len(result.leaks)} LEAK(S)'}")
    assert restore(result.texts, result.reid) == texts
    print("restore from map: exact")
    if args.save_map:
        print(f"re-identification map -> {write_reid_map(result).relative_to(C4_ROOT)}")


if __name__ == "__main__":
    main()
