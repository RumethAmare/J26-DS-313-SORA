#!/usr/bin/env python3
"""
The eval-split leakage guard.

C1_FINAL_IMPLEMENTATION.md §7 makes this a hard invariant enforced in code, not
convention, and §12 lists training on eval-split data as "a hard stop, not a
warning". So this raises; it never warns and never filters silently. A run that
would have leaked must die loudly enough that nobody reports its numbers.

Every script that consumes recordings for TRAINING calls assert_no_eval_leakage
before it touches a model — finetune_whisper.py and train_lid_crf.py in Phase 2,
and anything added later.

    from src.split_guard import assert_no_eval_leakage, train_ids
    assert_no_eval_leakage(my_recording_ids, context="whisper LoRA fine-tune")
"""
import csv
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import sora_paths

MANIFEST = os.path.join(sora_paths.data_dir(), "manifest.csv")


class EvalLeakageError(RuntimeError):
    """Raised when eval-split recordings reach a training code path."""


def _require_manifest():
    if not os.path.exists(MANIFEST):
        raise FileNotFoundError(
            f"{MANIFEST} not found. Run `python -m src.build_manifest` first — "
            "no training may start before the split is locked."
        )


def load_split():
    """recording_id -> 'train' | 'eval'."""
    _require_manifest()
    with open(MANIFEST, encoding="utf-8") as fh:
        return {r["recording_id"]: r["split"] for r in csv.DictReader(fh)}


def load_manifest_rows():
    _require_manifest()
    with open(MANIFEST, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def eval_ids():
    return sorted(r for r, s in load_split().items() if s == "eval")


def train_ids():
    return sorted(r for r, s in load_split().items() if s == "train")


def script_groups():
    """recording_id -> script_group, for breaking results down by script."""
    return {r["recording_id"]: r["script_group"] for r in load_manifest_rows()}


def assert_no_eval_leakage(recording_ids, context="training"):
    """
    Raise if any eval-split recording appears in a training set.

    Unknown recording ids are also fatal: an id absent from the locked manifest
    has no split, so it cannot be shown to be safe to train on.
    """
    split = load_split()
    ids = list(recording_ids)

    leaked = sorted({r for r in ids if split.get(r) == "eval"})
    unknown = sorted({r for r in ids if r not in split})

    if leaked:
        raise EvalLeakageError(
            f"[{context}] {len(leaked)} eval-split recording(s) in the training "
            f"list: {leaked}. C1 §12 makes this a hard stop — any model trained "
            f"this way produces meaningless eval numbers. Fix the file list."
        )
    if unknown:
        raise EvalLeakageError(
            f"[{context}] {len(unknown)} recording(s) are not in "
            f"{MANIFEST}: {unknown}. Every training recording must carry an "
            f"explicit split. Re-run `python -m src.build_manifest --force` to "
            f"admit new recordings (they join the train pool)."
        )
    print(f"[split_guard] {context}: {len(ids)} recordings, no eval leakage.")
    return ids


def _selftest():
    ev, tr = eval_ids(), train_ids()
    print(f"  train: {len(tr)} recordings")
    print(f"  eval : {len(ev)} recordings -> {ev}")

    ok = True

    assert_no_eval_leakage(tr, context="selftest/clean-train-list")

    if ev:
        try:
            assert_no_eval_leakage(tr + ev[:1], context="selftest/seeded-leak")
            print(f"  FAIL: leaking {ev[0]} did not raise")
            ok = False
        except EvalLeakageError as exc:
            print(f"  [ok ] seeded leak raised: {str(exc)[:78]}...")

    try:
        assert_no_eval_leakage(["J26DS313_R9999"], context="selftest/unknown-id")
        print("  FAIL: unknown recording id did not raise")
        ok = False
    except EvalLeakageError as exc:
        print(f"  [ok ] unknown id raised: {str(exc)[:78]}...")

    print("\nGuard is live." if ok else "\nGuard is NOT enforcing. Do not train.")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(f"eval : {eval_ids()}")
    print(f"train: {train_ids()}")
