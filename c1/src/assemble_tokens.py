#!/usr/bin/env python3
"""
Stage D -- merge ASR words + per-token LID labels into the output contract.

At inference time (Phase 2+) this takes real Whisper word output (word, start,
end, probability) and a trained CRF's real label predictions and produces the
token JSON docs/TOKEN_SCHEMA.md defines for C2/C3. Neither model exists yet,
but the merge logic and the switch-derivation rule can be built and verified
against gold data today -- that's what --selftest does.

WHY switch IS DERIVED, NOT COPIED
-----------------------------------
docs/TOKEN_SCHEMA.md measured the annotated switch field against "language
differs from the previous token in the same utterance" across all 26
recordings: 97.6% agreement, not 100%. Close enough that copying the field
would look right almost all the time, and far enough that it isn't the same
definition -- 136 tokens are flagged as switches with no language change, 1 the
reverse. Copying an annotation field that means something slightly different
from what it's used for is exactly the kind of bug that survives every review
because it looks correct. Deriving it is one line, has a stated definition,
and reproduces the 97.6% as a property of the DATA rather than an assumption
baked into the code.

Usage:
    python -m src.assemble_tokens --selftest
"""
import sys
import os

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import sora_paths

REQUIRED_FIELDS = {"utt_id", "tok_id", "token", "lang", "start", "end", "conf", "switch"}


def assemble(utt_id, words, labels):
    """
    Merge one utterance's ASR words with per-word language labels.

    words:  list of {"word", "start", "end", "probability"} in order, e.g.
            faster-whisper's word_timestamps output (src/decode_compare.py
            already captures exactly this shape).
    labels: list of language strings, same length and order as words --
            e.g. a trained CRF's predict() output for this sequence.

    Returns a list of dicts matching docs/TOKEN_SCHEMA.md's output contract,
    with `switch` derived rather than sourced from either input.
    """
    if len(words) != len(labels):
        raise ValueError(
            f"{utt_id}: {len(words)} words but {len(labels)} labels -- "
            "assembly requires one label per word, already aligned."
        )

    out = []
    prev_lang = None
    for i, (w, lang) in enumerate(zip(words, labels)):
        out.append({
            "utt_id": utt_id,
            "tok_id": i,
            "token": w["word"].strip(),
            "lang": lang,
            "start": round(w["start"], 3),
            "end": round(w["end"], 3),
            "conf": round(w.get("probability", 0.0), 4),
            "switch": prev_lang is not None and lang != prev_lang,
        })
        prev_lang = lang
    return out


def validate_schema(tokens):
    """Raise if any assembled token is missing a required field or has the
    wrong type where it matters. Cheap, deliberate check before this output
    ever reaches C2/C3 -- a silently malformed token is worse than a crash."""
    for tok in tokens:
        missing = REQUIRED_FIELDS - tok.keys()
        if missing:
            raise ValueError(f"assembled token missing fields {missing}: {tok}")
        if not isinstance(tok["switch"], bool):
            raise ValueError(f"switch must be bool, got {type(tok['switch'])}: {tok}")
        if not (0.0 <= tok["conf"] <= 1.0):
            raise ValueError(f"conf out of [0,1] range: {tok}")
    return True


def _words_and_labels_from_gold(rid):
    """
    Reshape gold tokens.jsonl into assemble()'s input shape, grouped by
    utterance -- this is what lets --selftest run the real merge+derive logic
    against real data with no model in the loop.

    assemble() expects numeric start/end, matching real ASR word-timestamp
    output (Whisper never emits a non-numeric timestamp). R0008's tokens.jsonl
    literally has the string "PROVISIONAL" in place of start/end for every
    token -- a genuine data-quality issue distinct from R0008's already-known
    incomplete-reference flag -- so those utterances are skipped here rather
    than silently coerced, and reported by the caller.
    """
    tokens = sora_paths.load_tokens(rid)
    by_utt = {}
    for tok in tokens:
        by_utt.setdefault(tok["utt_id"], []).append(tok)

    for utt_id, toks in by_utt.items():
        if any(not isinstance(t.get("start"), (int, float)) for t in toks):
            yield utt_id, None, None, None
            continue
        words = [{"word": t["token"], "start": t["start"], "end": t["end"],
                  "probability": 1.0} for t in toks]
        labels = [t["lang"] for t in toks]
        annotated_switch = [bool(t.get("switch")) for t in toks]
        yield utt_id, words, labels, annotated_switch


def _selftest():
    """
    Feed every recording's real gold tokens through assemble() and compare the
    derived switch field against the annotated one. Should land close to the
    97.6% TOKEN_SCHEMA.md already measured independently via audit_labels.py --
    a large deviation means this implementation disagrees with the documented
    definition and has a bug, not just an expected annotation gap.
    """
    total = agree = 0
    schema_failures = 0
    skipped_non_numeric = 0
    mismatch_examples = []

    for rid in sora_paths.gold_recording_ids():
        for utt_id, words, labels, annotated_switch in _words_and_labels_from_gold(rid):
            if words is None:
                skipped_non_numeric += 1
                continue
            assembled = assemble(utt_id, words, labels)
            try:
                validate_schema(assembled)
            except ValueError as exc:
                print(f"  SCHEMA FAIL {utt_id}: {exc}")
                schema_failures += 1
                continue

            for tok, gold_switch in zip(assembled, annotated_switch):
                total += 1
                if tok["switch"] == gold_switch:
                    agree += 1
                elif len(mismatch_examples) < 8:
                    mismatch_examples.append(
                        (rid, tok["tok_id"], tok["token"], tok["switch"], gold_switch)
                    )

    rate = agree / total if total else 0.0
    print(f"Assembled and validated tokens across {len(sora_paths.gold_recording_ids())} "
          f"recordings: {total} tokens, {schema_failures} schema failures")
    if skipped_non_numeric:
        print(f"Skipped {skipped_non_numeric} utterance(s) with non-numeric "
              f"timestamps (e.g. R0008's literal \"PROVISIONAL\" start/end -- "
              f"a data-quality issue, not an assembly bug)")
    print(f"Derived-switch vs. annotated-switch agreement: {rate:.4f} "
          f"({agree}/{total})")
    print("Reference (audit_labels.py, measured independently): 0.9760")

    if mismatch_examples:
        print("\nSample disagreements (derived != annotated):")
        for rid, tid, tok, derived, annotated in mismatch_examples:
            print(f"  {rid} tok#{tid} {tok!r:15} derived={derived} annotated={annotated}")

    ok = schema_failures == 0 and abs(rate - 0.976) < 0.01
    if not ok:
        print("\nFAIL: derived-switch rate drifted from the documented 97.6% "
              "by more than 0.01, or a schema check failed -- check the logic "
              "above before trusting this against real model output.")
        return 1
    print("\nMatches the documented rate. Assembly logic is consistent with "
          "the schema's stated definition.")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
