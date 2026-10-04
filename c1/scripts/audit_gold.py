#!/usr/bin/env python3
"""
audit_gold.py -- sanity-check every C1 gold tokens file before it is trained on.

validate.py checks that a recording's files are structurally consistent with
each other. It does not check whether the *labels* are plausible, and the
October 2026 batch (R0029-R0065) showed why that matters: six recordings were
committed with every token bulk-tagged one language, which validate.py passes
without comment. This script catches that class of problem, plus the
bookkeeping gaps that break joins downstream.

Per recording it reports:
  * filename / utt_id missing the J26DS313_ prefix
  * unparseable lines and tokens with null start/end
  * label mix, and a BULK-LABEL flag when one language takes >= 98% of a
    recording that has >= 50 tokens and zero switches in the gold `lang`
    sequence -- the R0017 signature
  * Sinhala-script tokens labelled EN (a Unicode-range contradiction)
  * script convention: Sinhala written in Sinhala Unicode vs romanized
  * whether audio exists and whether the manifest lists the recording
  * near-duplicate transcripts (same conversation annotated twice), which
    would leak across cross-validation folds

Outputs results/c1_gold_audit.csv and results/c1_gold_audit.md. Exit code is
always 0 -- this is a report, not a gate.

Usage:
    python audit_gold.py
"""
import csv
import difflib
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corpus
import lid_data
import script_normalize as sn

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(C1_ROOT, "results")

BULK_SHARE = 0.98
BULK_MIN_TOKENS = 50

# Romanized Sinhala function words with no English homograph. Any of these
# tagged EN is a labelling error, so their EN rate measures partial
# mislabelling that the bulk check cannot see. Deliberately excludes words
# that ARE English ('one', 'me', 'api', 'na', 'ne').
SI_FUNCTION_WORDS = {
    "eka", "ekak", "eke", "ekata", "ekka", "ekath", "mama", "mage", "mata",
    "oya", "oyata", "oyage", "nam", "nan", "neda", "thiyenawa", "kiyala",
    "karanna", "hari", "mokakda", "kohomada", "dan", "tika",
}
PARTIAL_MIN_WORDS = 10
PARTIAL_EN_RATE = 0.10
DUP_JACCARD_PREFILTER = 0.5
DUP_SEQ_RATIO = 0.8


def read_manifest_ids():
    if not os.path.exists(corpus.MANIFEST_PATH):
        return set()
    with open(corpus.MANIFEST_PATH, encoding="utf-8") as fh:
        return {corpus.canonical_rid(r["recording_id"]) for r in csv.DictReader(fh)}


def audit_file(rid, path, manifest_ids):
    fname = os.path.basename(path)
    toks, bad_lines = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                toks.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1

    n = len(toks)
    langs = [t.get("lang") for t in toks]
    counts = {lbl: langs.count(lbl) for lbl in lid_data.LABELS}
    unknown_labels = sum(1 for l in langs if l not in lid_data.LABELS)
    scripts = [lid_data.token_script(t.get("token", "")) for t in toks]

    # Switches re-derived from `lang` within utterances (same convention as
    # switch_detection_eval.py), not read from the unreliable `switch` field.
    n_switches, prev_lang, prev_utt = 0, None, None
    for t in toks:
        if t.get("utt_id") != prev_utt:
            prev_lang = None
        if prev_lang is not None and t.get("lang") != prev_lang:
            n_switches += 1
        prev_lang, prev_utt = t.get("lang"), t.get("utt_id")

    top_share = max(counts["SI"], counts["EN"]) / n if n else 0.0
    bulk = n >= BULK_MIN_TOKENS and top_share >= BULK_SHARE and n_switches <= 2

    null_ts = sum(1 for t in toks
                  if not isinstance(t.get("start"), (int, float))
                  or not isinstance(t.get("end"), (int, float)))
    fw_labels = [t.get("lang") for t in toks
                 if t.get("token", "").strip(".,:;!?\"'()").lower() in SI_FUNCTION_WORDS]
    fw_en_rate = fw_labels.count("EN") / len(fw_labels) if fw_labels else 0.0
    partial = len(fw_labels) >= PARTIAL_MIN_WORDS and fw_en_rate >= PARTIAL_EN_RATE

    si_script_as_en = sum(1 for t, s in zip(toks, scripts)
                          if s == "sinhala" and t.get("lang") == "EN")
    si_in_sinhala = sum(1 for t, s in zip(toks, scripts)
                        if s == "sinhala" and t.get("lang") == "SI")
    si_in_latin = sum(1 for t, s in zip(toks, scripts)
                      if s == "latin" and t.get("lang") == "SI")
    si_total = si_in_sinhala + si_in_latin
    if not si_total:
        script_conv = "none"
    elif si_in_sinhala / si_total >= 0.9:
        script_conv = "unicode"
    elif si_in_latin / si_total >= 0.9:
        script_conv = "romanized"
    else:
        script_conv = "mixed"

    prefix_ok = fname.startswith(corpus.RID_PREFIX)
    utt_prefix_ok = all(str(t.get("utt_id", "")).startswith(corpus.RID_PREFIX)
                        for t in toks)

    issues = []
    if rid in corpus.EXCLUDED_RECORDINGS:
        issues.append("EXCLUDED")
    if bulk:
        issues.append("BULK_LABEL")
    elif partial:
        issues.append("PARTIAL_MISLABEL")
    if bad_lines:
        issues.append("BAD_JSON")
    if unknown_labels:
        issues.append("UNKNOWN_LABEL")
    if null_ts:
        issues.append("NULL_TIMESTAMPS")
    if si_script_as_en:
        issues.append("SINHALA_SCRIPT_AS_EN")
    if not (prefix_ok and utt_prefix_ok):
        issues.append("NO_PREFIX")
    if not os.path.exists(os.path.join(corpus.AUDIO_DIR, f"{rid}.wav")):
        issues.append("NO_AUDIO")
    if rid not in manifest_ids:
        issues.append("NOT_IN_MANIFEST")

    row = {
        "recording_id": rid,
        "file": fname,
        "n_tokens": n,
        "SI": counts["SI"], "EN": counts["EN"], "OTHER": counts["OTHER"],
        "unknown_labels": unknown_labels,
        "derived_switches": n_switches,
        "top_language_share": round(top_share, 3),
        "si_function_words": len(fw_labels),
        "si_function_words_tagged_EN": round(fw_en_rate, 3),
        "null_timestamps": null_ts,
        "bad_json_lines": bad_lines,
        "sinhala_script_tagged_EN": si_script_as_en,
        "sinhala_script_convention": script_conv,
        "issues": " ".join(issues),
    }
    skeleton = [x for x in (sn.normalize_scripted(t.get("token", "")) for t in toks) if x]
    return row, skeleton


def find_duplicates(skeletons):
    """Pairs whose romanized token sequences match closely, across scripts."""
    dups = []
    for a, b in itertools.combinations(sorted(skeletons), 2):
        sa, sb = set(skeletons[a]), set(skeletons[b])
        if not sa or not sb:
            continue
        if len(sa & sb) / len(sa | sb) < DUP_JACCARD_PREFILTER:
            continue
        ratio = difflib.SequenceMatcher(
            None, skeletons[a], skeletons[b], autojunk=False).ratio()
        if ratio >= DUP_SEQ_RATIO:
            dups.append((a, b, round(ratio, 3)))
    return dups


def write_report(rows, dups, path):
    flagged = [r for r in rows if r["issues"].replace("NO_AUDIO", "").strip()]
    usable = [r for r in rows if r["recording_id"] not in corpus.EXCLUDED_RECORDINGS]
    L = ["# C1 gold-data audit\n\n",
         "Produced by `scripts/audit_gold.py`. Report only; nothing is modified.\n\n",
         f"- gold token files: **{len(rows)}** "
         f"({sum(r['n_tokens'] for r in rows):,} tokens)\n",
         f"- usable after exclusions: **{len(usable)}** "
         f"({sum(r['n_tokens'] for r in usable):,} tokens)\n",
         f"- with audio: **{sum(1 for r in rows if 'NO_AUDIO' not in r['issues'])}**\n",
         f"- not in manifest: **{sum(1 for r in rows if 'NOT_IN_MANIFEST' in r['issues'])}**\n\n",
         "## Excluded recordings\n\n",
         "| recording | reason |\n|---|---|\n"]
    for rid, why in sorted(corpus.EXCLUDED_RECORDINGS.items()):
        L.append(f"| {rid} | {why} |\n")

    for flag, label in (("BULK_LABEL", "Bulk-labelled"),
                        ("PARTIAL_MISLABEL", "Partially mislabelled")):
        missed = [r for r in rows if flag in r["issues"]
                  and "EXCLUDED" not in r["issues"]]
        if missed:
            L.append(f"\n**{label} recordings NOT yet excluded:** "
                     + ", ".join(r["recording_id"] for r in missed) + "\n")

    L.append("\n## Near-duplicate transcripts\n\n")
    if dups:
        L.append("Same conversation annotated more than once. If both copies are "
                 "used they must sit in the same CV fold.\n\n")
        L.append("| recording A | recording B | sequence similarity |\n|---|---|---|\n")
        for a, b, r in dups:
            L.append(f"| {a} | {b} | {r:.2f} |\n")
    else:
        L.append("None found.\n")

    L.append("\n## Recordings with issues (NO_AUDIO alone omitted)\n\n")
    L.append("`SI fw→EN` is the share of unambiguous romanized-Sinhala function "
             "words (`eka`, `mata`, `oyata`, …) tagged EN, with their count; it "
             "should be 0%.\n\n")
    L.append("| recording | tokens | SI | EN | OTHER | switches | SI fw→EN | null ts "
             "| SI script | issues |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for r in flagged:
        L.append(f"| {r['recording_id']} | {r['n_tokens']} | {r['SI']} | {r['EN']} "
                 f"| {r['OTHER']} | {r['derived_switches']} "
                 f"| {r['si_function_words_tagged_EN']:.0%} ({r['si_function_words']}) "
                 f"| {r['null_timestamps']} "
                 f"| {r['sinhala_script_convention']} | {r['issues']} |\n")

    conv = {}
    for r in usable:
        conv[r["sinhala_script_convention"]] = conv.get(r["sinhala_script_convention"], 0) + 1
    L.append("\n## Sinhala script convention (usable recordings)\n\n")
    for k, v in sorted(conv.items()):
        L.append(f"- {k}: {v}\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(L)


def main():
    manifest_ids = read_manifest_ids()
    rows, skeletons = [], {}
    for rid, path in corpus.gold_token_files():
        row, skel = audit_file(rid, path, manifest_ids)
        rows.append(row)
        skeletons[rid] = skel
    dups = find_duplicates(skeletons)

    print(f"{'recording':16} {'toks':>5} {'SI':>4} {'EN':>4} {'OTH':>4} "
          f"{'sw':>4} {'fw>EN':>6} {'nullts':>6}  issues")
    for r in rows:
        print(f"{r['recording_id']:16} {r['n_tokens']:>5} {r['SI']:>4} {r['EN']:>4} "
              f"{r['OTHER']:>4} {r['derived_switches']:>4} "
              f"{r['si_function_words_tagged_EN']:>6.0%} {r['null_timestamps']:>6}  "
              f"{r['issues']}")
    print("\nNear-duplicates:", dups or "none")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "c1_gold_audit.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md_path = os.path.join(RESULTS_DIR, "c1_gold_audit.md")
    write_report(rows, dups, md_path)
    print(f"\nWrote {csv_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
