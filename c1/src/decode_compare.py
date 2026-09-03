#!/usr/bin/env python3
"""
Phase 0: the decoding-mode x model-size grid.

C1_FINAL_IMPLEMENTATION.md §2 and §5 both refuse to assume these two choices —
they are decided on measured WER/CER against the real corpus. This script runs
{auto, forced-en, forced-si} x {small, medium, large-v3} and writes predictions
for evaluate_asr.py to score.

  auto  Whisper detects the language per segment. What the prior baseline did.
  en    Forced English — the "script collapse" / romanized path, which matches
        how 8 of the 26 recordings are actually annotated.
  si    Forced Sinhala — native Unicode output, matching the other 17.

Word-level timestamps and per-word probabilities are captured from the start:
Stage B owes C3 word timings and Stage D owes a confidence per token, and
asking for them now costs one flag rather than a re-run later.

Structure follows SORA_Dataset/scripts/c1/transcribe_baseline.py so the
`small`/`auto` cell reproduces the prior baseline as a harness check.

Usage:
    python -m src.decode_compare --sizes small --modes auto      # smoke test
    python -m src.decode_compare                                 # full 3x3 grid
    python -m src.decode_compare --sizes small,medium            # skip large-v3
"""
import argparse
import json
import os
import shutil
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import env_bootstrap  # noqa: F401  — sets HF_HOME; must precede HF imports
from src import sora_paths
from src.cuda_bootstrap import ensure_cuda_libs
from src.split_guard import load_manifest_rows

ALL_SIZES = ["small", "medium", "large-v3"]
ALL_MODES = ["auto", "en", "si"]

# Approximate CTranslate2 float16 download sizes, for the disk precheck.
SIZE_GB = {"small": 0.5, "medium": 1.5, "large-v3": 3.1}


def config_dir(size, mode):
    return os.path.join(sora_paths.predictions_dir(), f"{size.replace('-', '')}_{mode}")


def load_model(size):
    """
    Load a Whisper model, preferring the local HF cache.

    faster-whisper contacts huggingface.co on every load to revalidate the
    snapshot. On this network that call stalls for minutes with no output and
    no error — it looks exactly like a slow decode. Trying local_files_only
    first turns a cached load into ~1s and only reaches the network when the
    model genuinely is not present yet.

    This is a startup fix, not the §3 offline-mode requirement, which applies to
    the deployed inference entry point rather than to research scripts.
    """
    from faster_whisper import WhisperModel

    from src.fetch_models import direct_complete

    # A directly-downloaded directory wins: it only exists when
    # huggingface_hub failed for this model, so re-consulting the Hub would
    # just hang again.
    local = direct_complete(size)
    if local:
        return WhisperModel(local, device="cuda", compute_type="float16"), "direct"

    try:
        model = WhisperModel(size, device="cuda", compute_type="float16",
                             local_files_only=True)
        return model, "cache"
    except Exception:
        print(f"  '{size}' not in the local HF cache — downloading "
              f"(~{SIZE_GB.get(size, 1.0):.1f} GB, this can be slow)...")
        model = WhisperModel(size, device="cuda", compute_type="float16")
        return model, "download"


def transcribe_one(model, audio_path, mode):
    """One recording. Returns (utterances, info, elapsed_seconds)."""
    started = time.time()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        language=None if mode == "auto" else mode,
    )

    utterances = []
    for idx, seg in enumerate(segments, 1):  # generator: work happens here
        words = [
            {
                "word": w.word.strip(),
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "probability": round(w.probability, 4),
            }
            for w in (seg.words or [])
        ]
        utterances.append({
            "utt_id": f"w{idx:03d}",
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 4),
            "no_speech_prob": round(seg.no_speech_prob, 4),
            "words": words,
        })
    return utterances, info, time.time() - started


def run_config(size, mode, recordings, force=False):
    ensure_cuda_libs()  # must precede the first CTranslate2 CUDA allocation

    out_dir = config_dir(size, mode)
    sora_paths.ensure_dir(out_dir)
    meta_path = os.path.join(out_dir, "_config_meta.json")

    if os.path.exists(meta_path) and not force:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("n_recordings") == len(recordings):
            print(f"[{size}/{mode}] complete ({len(recordings)} recordings) — skipping. "
                  f"Use --force to redo.")
            return meta

    print(f"\n[{size}/{mode}] loading model (cuda, float16)...", flush=True)
    load_started = time.time()
    model, source = load_model(size)
    load_s = time.time() - load_started
    print(f"[{size}/{mode}] model ready in {load_s:.1f}s (from {source})", flush=True)

    results = []
    total_audio = total_decode = 0.0

    for i, rid in enumerate(recordings, 1):
        audio_path = sora_paths.audio_path(rid)
        utterances, info, elapsed = transcribe_one(model, audio_path, mode)

        payload = {
            "recording": rid,
            "config": {"model_size": size, "decode_mode": mode},
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 4),
            "audio_duration_s": round(info.duration, 2),
            "decode_seconds": round(elapsed, 2),
            "note": f"Phase 0 decode grid (size={size}, mode={mode}). "
                    f"Whisper's own segmentation — utt_ids do NOT align with gold.",
            "utterances": utterances,
        }
        with open(os.path.join(out_dir, f"{rid}.transcript.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        n_words = sum(len(u["words"]) for u in utterances)
        total_audio += info.duration
        total_decode += elapsed
        results.append({
            "recording": rid, "detected_language": info.language,
            "audio_duration_s": round(info.duration, 2),
            "decode_seconds": round(elapsed, 2),
            "n_segments": len(utterances), "n_words": n_words,
        })
        print(f"  [{i:2}/{len(recordings)}] {rid}  {info.duration:6.1f}s audio  "
              f"{elapsed:5.1f}s decode  lang={info.language}  "
              f"{len(utterances):3} seg  {n_words:4} words", flush=True)

    del model  # release VRAM before the next config loads

    meta = {
        "model_size": size,
        "decode_mode": mode,
        "n_recordings": len(recordings),
        "model_load_seconds": round(load_s, 2),
        "total_audio_s": round(total_audio, 2),
        "total_decode_s": round(total_decode, 2),
        # RTF < 1 means faster than real time — the §9 deployability column.
        "real_time_factor": round(total_decode / total_audio, 4) if total_audio else None,
        "per_recording": results,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[{size}/{mode}] done: {total_decode:.1f}s decode for "
          f"{total_audio/60:.1f} min audio (RTF {meta['real_time_factor']})")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default=",".join(ALL_SIZES))
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--force", action="store_true", help="redo already-complete configs")
    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for s in sizes:
        if s not in ALL_SIZES:
            ap.error(f"unknown size {s!r}; choose from {ALL_SIZES}")
    for m in modes:
        if m not in ALL_MODES:
            ap.error(f"unknown mode {m!r}; choose from {ALL_MODES}")

    # Every scorable recording. Phase 0 is zero-shot — nothing is trained, so
    # decoding the train split leaks nothing. evaluate_asr.py reports the eval
    # split separately, which is the number later phases compare against.
    recordings = sorted(r["recording_id"] for r in load_manifest_rows())

    free_gb = shutil.disk_usage(sora_paths.C1_ROOT).free / 1e9
    needed = sum(SIZE_GB.get(s, 1.0) for s in sizes)
    print(f"Phase 0 decode grid: {len(sizes)}x{len(modes)} = {len(sizes)*len(modes)} "
          f"configs over {len(recordings)} recordings")
    print(f"sizes={sizes}  modes={modes}")
    print(f"disk: {free_gb:.1f} GB free, up to ~{needed:.1f} GB of models may download")
    if free_gb < needed + 2:
        print(f"WARNING: only {free_gb:.1f} GB free. Drop large-v3 with "
              f"--sizes small,medium if the download fails.")

    summaries = []
    for size in sizes:
        for mode in modes:
            try:
                summaries.append(run_config(size, mode, recordings, force=args.force))
            except Exception as exc:  # one bad cell must not lose the other eight
                print(f"\n[{size}/{mode}] FAILED: {type(exc).__name__}: {exc}")
                summaries.append({"model_size": size, "decode_mode": mode,
                                  "error": f"{type(exc).__name__}: {exc}"})

    print(f"\n{'config':22} {'RTF':>7} {'decode_s':>9}  status")
    for m in summaries:
        label = f"{m['model_size']}/{m['decode_mode']}"
        if "error" in m:
            print(f"{label:22} {'-':>7} {'-':>9}  {m['error'][:40]}")
        else:
            print(f"{label:22} {m['real_time_factor']:>7} {m['total_decode_s']:>9.1f}  ok")

    print("\nNext: python -m src.evaluate_asr")
    return 1 if any("error" in m for m in summaries) else 0


if __name__ == "__main__":
    sys.exit(main())
