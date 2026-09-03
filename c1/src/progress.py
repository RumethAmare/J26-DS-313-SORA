#!/usr/bin/env python3
"""
Live progress for the Phase 0 decode grid.

The full 3x3 grid is several hours of GPU time, and `decode_compare` writes one
JSON per recording as it goes — so progress is readable straight off the disk,
with no coupling to the running process. That means this works whether the grid
is running in this shell, another terminal, or a background task, and it keeps
working after a crash.

ETA is measured, not guessed: each prediction file records its own
`decode_seconds` and `audio_duration_s`, so the real-time factor of a config in
flight comes from the recordings it has already finished. Configs not yet
started are projected from a finished config of the same model size, falling
back to rough size multipliers only when there is nothing measured to use.

Usage:
    python -m src.progress              # one-shot summary
    python -m src.progress --watch      # refresh until the grid is done
    python -m src.progress --oneline    # single line, for a status bar
"""
import argparse
import csv
import json
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import sora_paths
from src.decode_compare import ALL_MODES, ALL_SIZES, config_dir
from src.split_guard import load_manifest_rows

# Used only when no config of that size has finished yet. Derived from the
# measured small/auto RTF of 0.248 and medium/auto's early 0.85.
SIZE_RTF_FALLBACK = {"small": 0.25, "medium": 0.85, "large-v3": 1.70}
# Forced-language decoding avoids the fallback/repetition loops the Sinhala path
# triggers, so it runs far faster: small/en was 0.056 against small/auto's 0.248.
MODE_RTF_SCALE = {"auto": 1.0, "si": 1.0, "en": 0.23}


def total_audio_seconds():
    path = os.path.join(sora_paths.data_dir(), "manifest.csv")
    with open(path, encoding="utf-8") as fh:
        return sum(float(r["duration_s"] or 0) for r in csv.DictReader(fh))


def scan_config(size, mode):
    """Read one config's state off disk. No contact with the running process."""
    path = config_dir(size, mode)
    state = {
        "size": size, "mode": mode, "label": f"{size}/{mode}", "path": path,
        "n_done": 0, "done_audio_s": 0.0, "decode_s": 0.0, "rtf": None,
        "complete": False, "started": os.path.isdir(path),
    }
    if not state["started"]:
        return state

    for name in os.listdir(path):
        if not name.endswith(".transcript.json"):
            continue
        try:
            with open(os.path.join(path, name), encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue  # being written right now; it will be counted next refresh
        state["n_done"] += 1
        state["done_audio_s"] += payload.get("audio_duration_s") or 0.0
        state["decode_s"] += payload.get("decode_seconds") or 0.0

    state["complete"] = os.path.exists(os.path.join(path, "_config_meta.json"))
    state["rtf"] = (state["decode_s"] / state["done_audio_s"]
                    if state["done_audio_s"] else None)
    return state


def estimate_rtf(state, all_states):
    """Measured RTF if this config has produced anything; else projected."""
    if state["rtf"]:
        return state["rtf"], "measured"

    # A finished config of the same size is the best available evidence.
    same_size = [s for s in all_states
                 if s["size"] == state["size"] and s["rtf"] and s["n_done"] >= 3]
    if same_size:
        ref = same_size[0]
        scale = (MODE_RTF_SCALE.get(state["mode"], 1.0)
                 / MODE_RTF_SCALE.get(ref["mode"], 1.0))
        return ref["rtf"] * scale, f"from {ref['label']}"

    return (SIZE_RTF_FALLBACK.get(state["size"], 1.0)
            * MODE_RTF_SCALE.get(state["mode"], 1.0)), "estimated"


def bar(fraction, width=22):
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def human(seconds):
    if seconds is None:
        return "--"
    seconds = int(max(0, seconds))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    return f"{seconds / 3600:.1f}h"


def collect(sizes, modes):
    total_audio = total_audio_seconds()
    n_recordings = len(load_manifest_rows())
    states = [scan_config(s, m) for s in sizes for m in modes]

    remaining_s = 0.0
    for st in states:
        rtf, source = estimate_rtf(st, states)
        st["rtf_estimate"], st["rtf_source"] = rtf, source
        left_audio = max(0.0, total_audio - st["done_audio_s"])
        st["eta_s"] = 0.0 if st["complete"] else left_audio * rtf
        remaining_s += st["eta_s"]

    return {
        "states": states,
        "total_audio_s": total_audio,
        "n_recordings": n_recordings,
        "n_configs": len(states),
        "n_complete": sum(1 for s in states if s["complete"]),
        "done_audio_s": sum(s["done_audio_s"] for s in states),
        "grand_audio_s": total_audio * len(states),
        "remaining_s": remaining_s,
    }


def render(snap, oneline=False):
    frac = (snap["done_audio_s"] / snap["grand_audio_s"]
            if snap["grand_audio_s"] else 0.0)

    if oneline:
        running = next((s["label"] for s in snap["states"]
                        if s["started"] and not s["complete"]), None)
        return (f"C1 Phase 0 {bar(frac, 12)} {frac:5.1%} "
                f"| {snap['n_complete']}/{snap['n_configs']} configs"
                + (f" | now {running}" if running else "")
                + f" | ~{human(snap['remaining_s'])} left")

    lines = [
        "Phase 0 decode grid — " + time.strftime("%H:%M:%S"),
        "",
        f"  {'config':16} {'progress':24} {'done':>7} {'RTF':>6}  {'ETA':>6}",
        "  " + "-" * 66,
    ]
    n_rec = snap["n_recordings"] or 26
    for st in snap["states"]:
        f = st["done_audio_s"] / snap["total_audio_s"] if snap["total_audio_s"] else 0
        if st["complete"]:
            status, eta = "done", "-"
        elif st["started"]:
            status, eta = f"{st['n_done']}/{n_rec}", human(st["eta_s"])
        else:
            status, eta = "queued", human(st["eta_s"])
        rtf = f"{st['rtf_estimate']:.3f}"
        if st["rtf_source"] != "measured":
            rtf = f"~{rtf}"
        lines.append(f"  {st['label']:16} {bar(f)} {status:>7} {rtf:>6}  {eta:>6}")

    lines += [
        "  " + "-" * 66,
        f"  {'TOTAL':16} {bar(frac)} {frac:6.1%}",
        "",
        f"  {snap['n_complete']}/{snap['n_configs']} configs complete · "
        f"{human(snap['remaining_s'])} remaining · "
        f"done ~{time.strftime('%H:%M', time.localtime(time.time() + snap['remaining_s']))}",
        "  RTF marked ~ is projected from a finished config, not yet measured.",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", action="store_true", help="refresh until complete")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--oneline", action="store_true",
                    help="one compact line, for a status bar")
    ap.add_argument("--sizes", default=",".join(ALL_SIZES))
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    if not args.watch:
        print(render(collect(sizes, modes), oneline=args.oneline))
        return 0

    try:
        while True:
            snap = collect(sizes, modes)
            if not args.oneline:
                print("\033[2J\033[H", end="")  # clear, home
            print(render(snap, oneline=args.oneline), flush=True)
            if snap["n_complete"] == snap["n_configs"]:
                print("\nGrid complete. Next: python -m src.evaluate_asr")
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n(watch stopped; the decode grid is unaffected)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
