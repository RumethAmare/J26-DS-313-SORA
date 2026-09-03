#!/usr/bin/env python3
"""
Audit the gold language labels and propose the SI / EN / NUM / OTHER schema.

THE PROBLEM
-----------
docs/DATA_CONTRACT.md defines lang as {SI, EN, OTHER}, and OTHER is intended to
mean Tamil — Sri Lanka's other national language — not a residual bucket. The
annotations do not behave that way. Of 92 OTHER tokens, roughly 85 are numerals,
phone numbers, NIC numbers, reference codes and one email address; only a
handful are genuinely Tamil. Meanwhile ~200 numeric tokens are labelled EN. The
same phenomenon is split across two classes on no consistent principle.

The measured consequence is in SORA_Dataset/results/c1_lid_eval.json: OTHER
scores precision 0.00, recall 0.00, F1 0.00. The class is unlearnable as
annotated, and no amount of model work fixes a label definition problem.

THE FIX
-------
Four labels: SI / EN / NUM / OTHER, with OTHER reserved for Tamil as intended.
NUM becomes a deterministic rule rather than a judgement call, which keeps ~85
numeric tokens from poisoning the Tamil class and gives C2/C4 the NIC and
phone-number spans they need anyway.

This script only PROPOSES. It writes a reviewable patch under c1/ and never
edits the shared SORA_Dataset annotations — relabelling the team's corpus is a
team decision.

Usage:
    python -m src.audit_labels
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import sora_paths
from src.script_fold import fold, is_sinhala_script

# A token is NUM if, once surrounding punctuation is stripped, it still carries
# a digit. That catches bare numerals (1990), grouped amounts (15,000),
# decimals (1.5), percentages (70%), times (10:00), ordinals (21st), phone
# numbers (0777998876), NICs (8812304567V) and reference codes
# (CEB/KW/2024/008834) with one rule nobody has to adjudicate.
_STRIP_PUNCT = re.compile(r"^[^\w඀-෿]+|[^\w඀-෿%]+$")
_HAS_DIGIT = re.compile(r"\d")
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)

LABELS = ["SI", "EN", "NUM", "OTHER"]


def core(token):
    return _STRIP_PUNCT.sub("", token)


def is_num(token):
    """Deterministic NUM rule. Email-like tokens are excluded — they carry
    digits but are not numbers, and belong in a review bucket, not a class."""
    c = core(token)
    if not c or "@" in c:
        return False
    return bool(_HAS_DIGIT.search(c))


def classify(token, gold_lang):
    """
    Return (proposed_label, rule). Conservative by design: only the NUM rule is
    confident enough to auto-propose. Everything else keeps its gold label, so
    a human reviews the genuinely ambiguous cases rather than a heuristic
    silently overwriting them.
    """
    if is_num(token):
        rule = "ordinal" if _ORDINAL.match(core(token)) else "contains-digit"
        return "NUM", f"num:{rule}"
    return gold_lang, "keep:gold"


# A recording contradicting the rest of the corpus on more than this share of
# its checkable tokens is systematically mislabelled, not merely ambiguous. The
# corpus splits cleanly: every sound recording sits at or below 0.07.
CONTRADICTION_THRESHOLD = 0.15
MIN_OTHER_EVIDENCE = 2      # occurrences elsewhere before a token can judge
MAJORITY_RATIO = 3          # how lopsided the outside evidence must be


def consistency_check(tokens_by_rec):
    """
    Find recordings whose labels contradict how the same words are labelled
    everywhere else.

    Needs no external lexicon: the corpus is its own reference. A token is
    checkable when it appears in at least one other recording with enough
    occurrences to form a majority. Comparison is on the folded surface, so
    කොහොමද and kohomada count as the same word.

    This catches the failure that matters most for Stage C — a whole recording
    annotated on a different understanding of the label set — which token-level
    spot checks miss and which would silently corrupt CRF training.
    """
    global_labels = defaultdict(Counter)
    recs_with = defaultdict(set)
    for rid, tokens in tokens_by_rec.items():
        for tok in tokens:
            if is_num(tok.get("token", "")):
                continue
            key = fold(tok.get("token", "")).strip()
            if key:
                global_labels[key][tok.get("lang")] += 1
                recs_with[key].add(rid)

    report = {}
    for rid, tokens in sorted(tokens_by_rec.items()):
        checked = 0
        contradictions = []
        for tok in tokens:
            if is_num(tok.get("token", "")):
                continue
            key = fold(tok.get("token", "")).strip()
            if not key or len(recs_with[key]) < 2:
                continue
            gold = tok.get("lang")
            outside = Counter(global_labels[key])
            outside[gold] -= 1  # discount this occurrence
            if sum(outside.values()) < MIN_OTHER_EVIDENCE:
                continue
            checked += 1
            majority, _ = outside.most_common(1)[0]
            if majority != gold and outside[majority] >= MAJORITY_RATIO * max(1, outside[gold]):
                contradictions.append({
                    "token": tok.get("token"), "utt_id": tok.get("utt_id"),
                    "tok_id": tok.get("tok_id"), "labelled": gold,
                    "corpus_majority": majority,
                })
        if checked:
            rate = len(contradictions) / checked
            report[rid] = {
                "n_checked": checked,
                "n_contradictions": len(contradictions),
                "contradiction_rate": round(rate, 4),
                "systematic": rate > CONTRADICTION_THRESHOLD,
                "examples": contradictions[:15],
            }
    return report


def main():
    tokens_by_rec = {}
    for rid in sora_paths.gold_recording_ids():
        tokens_by_rec[rid] = sora_paths.load_tokens(rid)

    before = Counter()
    after = Counter()
    proposals = []
    review = defaultdict(list)
    other_survivors = []
    switch_stats = Counter()

    for rid, tokens in sorted(tokens_by_rec.items()):
        prev_lang_by_utt = {}
        for tok in tokens:
            token, gold = tok.get("token", ""), tok.get("lang")
            before[gold] += 1
            proposed, rule = classify(token, gold)
            after[proposed] += 1

            if proposed != gold:
                proposals.append({
                    "recording": rid,
                    "utt_id": tok.get("utt_id"),
                    "tok_id": tok.get("tok_id"),
                    "token": token,
                    "current_lang": gold,
                    "proposed_lang": proposed,
                    "rule": rule,
                })

            # Not a defect: English loanwords written in Sinhala script
            # (හොටෙල් "hotel", නම්බර් "number"). Collected because they are the
            # hardest case for a script-feature-based LID — the script says
            # Sinhala and the word is English — so the CRF needs them counted,
            # not corrected.
            sinhala = is_sinhala_script(token)
            if proposed == "EN" and sinhala:
                review["en_loanword_in_sinhala_script"].append(
                    {"recording": rid, "token": token})
            if proposed == "OTHER":
                other_survivors.append({
                    "recording": rid, "token": token,
                    "script": "sinhala" if sinhala else "latin",
                })
            if "@" in token:
                review["email_like"].append({"recording": rid, "token": token})

            # Does the annotated `switch` flag match a language change?
            utt = tok.get("utt_id")
            prev = prev_lang_by_utt.get(utt)
            derived = prev is not None and prev != gold
            annotated = bool(tok.get("switch"))
            switch_stats["total"] += 1
            switch_stats["annotated_true"] += int(annotated)
            switch_stats["derived_true"] += int(derived)
            switch_stats["agree"] += int(derived == annotated)
            if annotated and not derived:
                switch_stats["annotated_not_derived"] += 1
            if derived and not annotated:
                switch_stats["derived_not_annotated"] += 1
            prev_lang_by_utt[utt] = gold

    total = sum(before.values())

    print("=" * 74)
    print("LABEL AUDIT — current vs. proposed SI / EN / NUM / OTHER")
    print("=" * 74)
    print(f"\n{'label':8} {'before':>8} {'after':>8}   delta")
    for lab in LABELS:
        b, a = before.get(lab, 0), after.get(lab, 0)
        print(f"{lab:8} {b:>8} {a:>8}   {a - b:+d}")
    print(f"{'TOTAL':8} {total:>8} {sum(after.values()):>8}")

    print(f"\nProposed relabels: {len(proposals)} of {total} tokens "
          f"({len(proposals)/total:.1%})")
    moved = Counter(f"{p['current_lang']} -> {p['proposed_lang']}" for p in proposals)
    for change, n in moved.most_common():
        print(f"  {change:16} {n:>5}")

    print(f"\nOTHER survivors (the Tamil candidates), {len(other_survivors)} tokens:")
    for item in other_survivors:
        print(f"  {item['recording']}  {item['script']:8} {item['token']!r}")

    print("\nFlagged for human review:")
    for kind, items in sorted(review.items()):
        uniq = Counter(i["token"] for i in items)
        print(f"  {kind}: {len(items)} tokens, {len(uniq)} distinct")
        for tokv, n in uniq.most_common(8):
            print(f"      {n:>4}x {tokv!r}")
    print("  Note: en_loanword_in_sinhala_script is a real code-mixing "
          "phenomenon, not an annotation error — script alone cannot decide "
          "these, so the CRF needs lexical features, not just script features.")

    agree_pct = switch_stats["agree"] / switch_stats["total"]
    print(f"\nSWITCH FIELD — annotated flag vs. a language change in the same utterance")
    print(f"  annotated switch=true : {switch_stats['annotated_true']:>5} "
          f"({switch_stats['annotated_true']/switch_stats['total']:.1%})")
    print(f"  derived   switch=true : {switch_stats['derived_true']:>5} "
          f"({switch_stats['derived_true']/switch_stats['total']:.1%})")
    print(f"  agreement             : {agree_pct:.1%}")
    print(f"  annotated but not derived: {switch_stats['annotated_not_derived']:>5}")
    print(f"  derived but not annotated: {switch_stats['derived_not_annotated']:>5}")
    if agree_pct < 0.95:
        print("  -> The switch field does not simply mean 'the language changed'. "
              "Stage D should DERIVE switch markers rather than trust this field, "
              "and switch-point accuracy (§9) needs a stated definition first.")

    # ── cross-recording label consistency ─────────────────────
    consistency = consistency_check(tokens_by_rec)
    suspect = {r: v for r, v in consistency.items() if v["systematic"]}

    print(f"\nCROSS-RECORDING LABEL CONSISTENCY — does this recording label words "
          f"the way the rest of the corpus does?")
    print(f"{'recording':22} {'checked':>8} {'contra':>7} {'rate':>6}")
    for rid, v in sorted(consistency.items(),
                         key=lambda kv: -kv[1]["contradiction_rate"])[:6]:
        mark = "  <-- SYSTEMATIC" if v["systematic"] else ""
        print(f"{rid:22} {v['n_checked']:>8} {v['n_contradictions']:>7} "
              f"{v['contradiction_rate']:>6.2f}{mark}")

    if suspect:
        print(f"\n{len(suspect)} recording(s) are systematically mislabelled:")
        for rid, v in suspect.items():
            ex = ", ".join(f"{e['token']!r}={e['labelled']}(corpus says "
                           f"{e['corpus_majority']})" for e in v["examples"][:4])
            print(f"  {rid}: {v['contradiction_rate']:.0%} of "
                  f"{v['n_checked']} checkable tokens — {ex}")
        print("  -> These must be corrected before any CRF training or LID "
              "evaluation. A recording annotated on a different reading of the "
              "label set poisons training and makes the F1 unfalsifiable.")

    # ── write outputs ─────────────────────────────────────────
    sora_paths.ensure_dir(sora_paths.data_dir())

    audit = {
        "n_recordings": len(tokens_by_rec),
        "n_tokens": total,
        "label_counts_before": dict(before),
        "label_counts_after": dict(after),
        "n_proposed_relabels": len(proposals),
        "relabel_transitions": dict(moved),
        "other_survivors": other_survivors,
        "review": {k: v for k, v in review.items()},
        "switch_field": dict(switch_stats),
        "switch_agreement": round(agree_pct, 4),
        "cross_recording_consistency": consistency,
        "systematically_mislabelled": sorted(suspect),
        "num_rule": "strip surrounding punctuation; NUM if the remainder "
                    "contains a digit and no '@'",
        "note": "Proposals only. Nothing in SORA_Dataset is modified by this "
                "script; applying the patch is a team decision (Phase 1).",
    }
    audit_path = os.path.join(sora_paths.data_dir(), "label_audit.json")
    with open(audit_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)

    patch_path = os.path.join(sora_paths.data_dir(), "label_relabel_patch.jsonl")
    with open(patch_path, "w", encoding="utf-8") as fh:
        for p in proposals:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nWrote {audit_path}")
    print(f"Wrote {patch_path}  ({len(proposals)} proposed changes, NOT applied)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
