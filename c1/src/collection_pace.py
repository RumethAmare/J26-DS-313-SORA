#!/usr/bin/env python3
"""
Turns "track hours/week from day one" (C1_FINAL_IMPLEMENTATION.md §8) and
"escalate if pace is short by end of Week 2" (§12.1) into an actual number,
runnable anytime, rather than something the team has to remember to eyeball.

Reads the LOCKED train-split duration from data/manifest.csv -- not the raw
upstream manifest -- so this always reflects what Phase 1 scripts (preprocess.py
included) actually treat as the training pool, not the whole corpus.

Usage:
    python -m src.collection_pace
    python -m src.collection_pace --floor-hours 15 --stretch-hours 20
"""
import argparse
import sys
from datetime import date, timedelta

if __package__ in (None, ""):
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.split_guard import load_manifest_rows

# §11's timeline: Phase 1 is Weeks 1-3, Sep 6 - Sep 26 2026.
PHASE1_START = date(2026, 9, 6)
PHASE1_END = date(2026, 9, 26)
WEEK2_CHECKPOINT = PHASE1_START + timedelta(days=13)  # end of Week 2

FLOOR_HOURS = 15.0    # §6.2 minimum
STRETCH_HOURS = 20.0  # §6.2 preferred
GATE_HOURS = 5.0      # §12.2 -- do not start Phase 2 fine-tuning below this


def current_train_hours():
    rows = load_manifest_rows()
    seconds = sum(float(r["duration_s"]) for r in rows if r["split"] == "train")
    return seconds / 3600, sum(1 for r in rows if r["split"] == "train")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floor-hours", type=float, default=FLOOR_HOURS)
    ap.add_argument("--stretch-hours", type=float, default=STRETCH_HOURS)
    ap.add_argument("--today", help="override today's date, YYYY-MM-DD (testing)")
    args = ap.parse_args()

    today = date.fromisoformat(args.today) if args.today else date.today()
    hours, n_recordings = current_train_hours()

    print(f"Training pool today ({today.isoformat()}): "
          f"{hours:.2f}h across {n_recordings} recordings")
    print(f"  Phase 2 gate (§12.2)   : {GATE_HOURS:.0f}h -- "
          f"{'CLEARED' if hours >= GATE_HOURS else f'{GATE_HOURS - hours:.2f}h short'}")
    print(f"  Floor target (§6.2)    : {args.floor_hours:.0f}h -- "
          f"{'MET' if hours >= args.floor_hours else f'{args.floor_hours - hours:.2f}h short'}")
    print(f"  Stretch target (§6.2)  : {args.stretch_hours:.0f}h -- "
          f"{'MET' if hours >= args.stretch_hours else f'{args.stretch_hours - hours:.2f}h short'}")

    days_left = (PHASE1_END - today).days
    print(f"\nPhase 1 window: {PHASE1_START.isoformat()} -> {PHASE1_END.isoformat()} "
          f"({days_left} day(s) remaining)")

    if today > PHASE1_END:
        print("Phase 1 window has closed.")
    elif hours >= args.floor_hours:
        print(f"Floor already met -- Phase 2 can proceed once the team is ready "
              f"(subject to the label/annotation fixes still pending on "
              f"SORA_Dataset, tracked separately).")
    else:
        needed = args.floor_hours - hours
        weeks_left = max(days_left / 7, 1 / 7)
        rate = needed / weeks_left
        print(f"Need {needed:.2f}h more to clear the floor -> "
              f"{rate:.2f}h/week required from today to {PHASE1_END.isoformat()}.")

    if today <= WEEK2_CHECKPOINT < PHASE1_END or today == WEEK2_CHECKPOINT:
        remaining_to_checkpoint = (WEEK2_CHECKPOINT - today).days
        print(f"\n§12.1 escalation checkpoint: {WEEK2_CHECKPOINT.isoformat()} "
              f"({'today' if remaining_to_checkpoint == 0 else f'{remaining_to_checkpoint} day(s) away'}).")
    elif today > WEEK2_CHECKPOINT:
        status = "PASSED while still under the floor -- escalate per §12.1" \
            if hours < args.floor_hours else "passed, floor already met"
        print(f"\n§12.1 escalation checkpoint ({WEEK2_CHECKPOINT.isoformat()}): {status}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
