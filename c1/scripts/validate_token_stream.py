#!/usr/bin/env python3
"""
validate_token_stream.py -- enforce the extended token schema (Task 2).

Checks every `<rid>.tokens.jsonl` in a directory against docs/TOKEN_SCHEMA.md.
Runs against predicted streams by default, and against the gold annotations
with `--gold` (gold legitimately lacks the three new fields, so those checks
are skipped there).

Why this exists: Task 8 merges this stream with Tasks 3/4/5 output into the
released dataset, and a malformed token layer would propagate silently into
every downstream consumer. Catching it here is cheap; catching it after C2/C3
have built on top of it is not.

Checks:
  [1] every line is valid JSON with all required fields
  [2] lang is one of SI / EN / OTHER
  [3] start <= end, both non-negative
  [4] tok_id is contiguous from 0 within each utterance
  [5] switch matches its derived definition (lang[i] != lang[i-1], first=False)
  [6] confidences are in [0, 1]
  [7] lang_method is a known value and consistent with the label
      (unicode_tamil must yield OTHER, unicode_sinhala must yield SI, ...)
  [8] timestamps are non-decreasing within an utterance

Exit code is non-zero if any ERROR is found. Warnings do not fail the run.
"""
import argparse
import collections
import json
import os
import sys

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = "/mnt/F/SLIIT/Research/SORA_Dataset"
PRED_DIR = os.path.join(C1_ROOT, "predictions", "token_stream")
GOLD_DIR = os.path.join(DATASET_ROOT, "annotations", "c1")

BASE_FIELDS = ["utt_id", "tok_id", "token", "lang", "start", "end", "switch"]
EXTENDED_FIELDS = ["asr_confidence", "lang_confidence", "lang_method"]
VALID_LANGS = {"SI", "EN", "OTHER"}

METHOD_IMPLIES = {
    "unicode_sinhala": "SI",
    "unicode_tamil": "OTHER",
    "numeric": "EN",
    "wordfreq_en": "EN",
    "fallback_si": "SI",
}


def validate_file(path, require_extended):
    errors, warnings = [], []
    tokens = []
    for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            tok = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON ({exc})")
            continue

        required = BASE_FIELDS + (EXTENDED_FIELDS if require_extended else [])
        for field in required:
            if field not in tok:
                errors.append(f"line {lineno}: missing field {field!r}")   # [1]

        if tok.get("lang") not in VALID_LANGS:
            errors.append(f"line {lineno}: bad lang {tok.get('lang')!r}")   # [2]

        start, end = tok.get("start"), tok.get("end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            if start < 0 or end < 0:
                errors.append(f"line {lineno}: negative timestamp")          # [3]
            elif end < start:
                errors.append(f"line {lineno}: end {end} < start {start}")

        if require_extended:
            for field in ("asr_confidence", "lang_confidence"):             # [6]
                val = tok.get(field)
                if isinstance(val, (int, float)) and not (0.0 <= val <= 1.0):
                    errors.append(f"line {lineno}: {field}={val} outside [0,1]")
            method = tok.get("lang_method")
            if method not in METHOD_IMPLIES:
                errors.append(f"line {lineno}: unknown lang_method {method!r}")  # [7]
            elif METHOD_IMPLIES[method] != tok.get("lang"):
                errors.append(f"line {lineno}: lang_method {method!r} implies "
                              f"{METHOD_IMPLIES[method]} but lang is {tok.get('lang')!r}")
        tokens.append(tok)

    by_utt = collections.OrderedDict()
    for tok in tokens:
        by_utt.setdefault(tok.get("utt_id"), []).append(tok)

    switch_mismatches = 0
    for utt_id, toks in by_utt.items():
        for i, tok in enumerate(toks):
            if tok.get("tok_id") != i:                                       # [4]
                errors.append(f"{utt_id}: tok_id {tok.get('tok_id')} at position {i}")
                break
        prev_lang = None
        prev_end = None
        for tok in toks:
            expected = prev_lang is not None and tok.get("lang") != prev_lang   # [5]
            if bool(tok.get("switch")) != expected:
                switch_mismatches += 1
            prev_lang = tok.get("lang")
            start = tok.get("start")
            if (prev_end is not None and isinstance(start, (int, float))
                    and start < prev_end - 1e-6):                            # [8]
                warnings.append(f"{utt_id}: timestamps go backwards at tok "
                                f"{tok.get('tok_id')} ({start} < {prev_end})")
            if isinstance(tok.get("end"), (int, float)):
                prev_end = tok["end"]

    return errors, warnings, len(tokens), len(by_utt), switch_mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--gold", action="store_true",
                    help="validate the gold annotations instead (skips the "
                         "three prediction-only fields)")
    a = ap.parse_args()

    target = a.dir or (GOLD_DIR if a.gold else PRED_DIR)
    require_extended = not a.gold

    if not os.path.isdir(target):
        raise SystemExit(f"No such directory: {target}")
    files = sorted(f for f in os.listdir(target) if f.endswith(".tokens.jsonl"))
    if not files:
        raise SystemExit(f"No .tokens.jsonl files in {target}")

    print(f"Validating {len(files)} file(s) in {target}")
    print(f"Extended fields {'required' if require_extended else 'skipped (gold)'}\n")

    total_err = total_warn = total_tokens = total_switch_mismatch = 0
    for fname in files:
        errors, warnings, n_tok, n_utt, sw_mismatch = validate_file(
            os.path.join(target, fname), require_extended)
        total_err += len(errors)
        total_warn += len(warnings)
        total_tokens += n_tok
        total_switch_mismatch += sw_mismatch

        status = "FAIL" if errors else ("warn" if warnings or sw_mismatch else "ok")
        line = (f"  {status:4} {fname[:-len('.tokens.jsonl')]:20} "
                f"{n_tok:5} tokens {n_utt:4} utts")
        if sw_mismatch:
            line += f"  switch-mismatch={sw_mismatch}"
        print(line)
        for e in errors[:5]:
            print(f"         ERROR {e}")
        if len(errors) > 5:
            print(f"         ... and {len(errors) - 5} more errors")
        for w in warnings[:3]:
            print(f"         warn  {w}")
        if len(warnings) > 3:
            print(f"         ... and {len(warnings) - 3} more warnings")

    print(f"\n{total_tokens} tokens across {len(files)} files")
    print(f"errors={total_err}  warnings={total_warn}  "
          f"switch-mismatches={total_switch_mismatch}")

    if total_switch_mismatch:
        pct = 100 * total_switch_mismatch / total_tokens if total_tokens else 0
        print(f"\nNote: switch disagrees with its derived definition on "
              f"{total_switch_mismatch} tokens ({pct:.1f}%). In gold this is an "
              f"annotation issue to audit before Task 4 quotes switch-point F1; "
              f"in predictions it would be a bug in build_token_stream.py.")

    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())
