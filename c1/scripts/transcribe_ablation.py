#!/usr/bin/env python3
"""
transcribe_ablation.py -- Task 1 ablation: model size x language forcing.

Runs faster-whisper over every recording once per configuration and saves
predictions under c1/predictions/<config_name>/<rid>.transcript.json, in the
same shape as the gold annotations so evaluate_normalized.py can score them.

The two axes:
  * model size    -- 'small' vs 'medium' (medium only if locally available;
                     this machine is offline-constrained, see --sizes)
  * language      -- auto-detect vs forced 'si' vs forced 'en'

Language forcing matters for code-mixed audio specifically: with auto-detect
Whisper picks ONE language for the whole recording and decodes everything in
that script, which is the wrong shape for Sinhala-English code-mixing. The
ablation shows how much that single choice costs.

Usage:
    python3 transcribe_ablation.py                    # all available configs
    python3 transcribe_ablation.py --sizes small      # skip medium
    python3 transcribe_ablation.py --configs small_auto small_si
"""
import argparse
import json
import os
import time

# This host has no working outbound HTTPS route (IPv6 connects hang in
# SYN-SENT). faster-whisper otherwise contacts HuggingFace to revalidate even
# an already-cached model, so a run with the weights sitting in ~/.cache
# hangs indefinitely rather than failing fast. Pin to the local cache before
# faster_whisper is imported anywhere. This also matches C1's offline
# deployment constraint, so it is the right default regardless.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cuda_env  # noqa: E402  (must run before faster_whisper touches CUDA)

cuda_env.ensure_cuda_libs()

DATASET_ROOT = "/mnt/F/SLIIT/Research/SORA_Dataset"
C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(DATASET_ROOT, "processed", "audio")
GOLD_DIR = os.path.join(DATASET_ROOT, "annotations", "c1")
PRED_ROOT = os.path.join(C1_ROOT, "predictions")

LANG_SETTINGS = [("auto", None), ("si", "si"), ("en", "en")]


def gold_recording_ids():
    ids = set()
    for fname in os.listdir(GOLD_DIR):
        if fname.endswith(".transcript.json"):
            ids.add(fname[: -len(".transcript.json")])
    return sorted(ids)


def model_is_local(size):
    """True if the CTranslate2 weights are already in the HF cache."""
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    return os.path.isdir(os.path.join(cache, f"models--Systran--faster-whisper-{size}"))


def run_config(size, lang_name, lang_code, rids, compute_type):
    from faster_whisper import WhisperModel

    config_name = f"{size}_{lang_name}"
    out_dir = os.path.join(PRED_ROOT, config_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 62}")
    print(f"config: {config_name}  (size={size}, language={lang_code or 'auto-detect'})")
    print(f"{'=' * 62}")

    load_start = time.time()
    model = WhisperModel(size, device="cuda", compute_type=compute_type)
    print(f"model loaded in {time.time() - load_start:.1f}s")

    total_audio = 0.0
    total_wall = 0.0

    for i, rid in enumerate(rids, 1):
        audio_path = os.path.join(AUDIO_DIR, f"{rid}.wav")
        t0 = time.time()
        segments, info = model.transcribe(
            audio_path, beam_size=5, vad_filter=True, language=lang_code
        )
        utterances = []
        for idx, seg in enumerate(segments, 1):
            utterances.append({
                "utt_id": f"{rid}_w{idx:03d}",
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
        elapsed = time.time() - t0
        total_wall += elapsed
        total_audio += info.duration

        out = {
            "recording": rid,
            "config": config_name,
            "model_size": size,
            "language_setting": lang_code or "auto",
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 3),
            "audio_duration_s": round(info.duration, 2),
            "transcribe_wall_s": round(elapsed, 2),
            "utterances": utterances,
        }
        with open(os.path.join(out_dir, f"{rid}.transcript.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)

        print(f"  [{i:2}/{len(rids)}] {rid}  {len(utterances):3} segs  "
              f"detected={info.language}  {elapsed:5.1f}s")

    rtf = total_wall / total_audio if total_audio else float("nan")
    print(f"  -> {len(rids)} recordings, {total_audio:.0f}s audio, "
          f"{total_wall:.0f}s wall, RTF={rtf:.3f}")

    meta = {
        "config": config_name,
        "model_size": size,
        "language_setting": lang_code or "auto",
        "n_recordings": len(rids),
        "total_audio_s": round(total_audio, 2),
        "total_wall_s": round(total_wall, 2),
        "real_time_factor": round(rtf, 4),
        "compute_type": compute_type,
    }
    with open(os.path.join(out_dir, "_config_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    del model
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["small", "medium"])
    ap.add_argument("--configs", nargs="+", default=None,
                    help="explicit config names to run, e.g. small_auto small_si")
    ap.add_argument("--compute-type", default="float16")
    a = ap.parse_args()

    all_ids = gold_recording_ids()
    rids = [r for r in all_ids if os.path.exists(os.path.join(AUDIO_DIR, f"{r}.wav"))]
    missing = [r for r in all_ids if r not in rids]
    print(f"Gold recordings: {len(all_ids)}; with audio: {len(rids)}; "
          f"skipped (no audio): {missing}")

    sizes = [s for s in a.sizes if model_is_local(s)]
    unavailable = [s for s in a.sizes if s not in sizes]
    if unavailable:
        print(f"NOTE: model(s) {unavailable} not in local HF cache and this host "
              f"appears offline -- skipping that arm of the ablation.")
    if not sizes:
        raise SystemExit("No requested model size is available locally.")

    metas = []
    for size in sizes:
        for lang_name, lang_code in LANG_SETTINGS:
            name = f"{size}_{lang_name}"
            if a.configs and name not in a.configs:
                continue
            metas.append(run_config(size, lang_name, lang_code, rids, a.compute_type))

    os.makedirs(os.path.join(C1_ROOT, "results"), exist_ok=True)
    out = os.path.join(C1_ROOT, "results", "c1_ablation_runs.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"configs": metas}, fh, indent=2)
    print(f"\nRan {len(metas)} config(s). Wrote {out}")


if __name__ == "__main__":
    main()
