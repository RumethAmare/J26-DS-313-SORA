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

import corpus  # noqa: E402
DATASET_ROOT = corpus.DATASET_ROOT
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


def _decode_worker(size, compute_type, lang_code, in_q, out_q):
    """Child process: load the model once, then decode recordings from in_q.

    Decoding lives in a child so the parent can kill it. faster-whisper's
    temperature fallback can loop on a single 30s window for over an hour
    (medium_auto on R0014: 5,662s for 91s of audio), and that happens inside
    one CTranslate2 call that nothing in-process can interrupt.
    """
    from faster_whisper import WhisperModel
    model = WhisperModel(size, device="cuda", compute_type=compute_type)
    out_q.put(("ready", None))
    while True:
        rid = in_q.get()
        if rid is None:
            return
        t0 = time.time()
        segments, info = model.transcribe(
            os.path.join(AUDIO_DIR, f"{rid}.wav"),
            beam_size=5, vad_filter=True, language=lang_code)
        utterances = [{"utt_id": f"{rid}_w{idx:03d}",
                       "start": round(seg.start, 2), "end": round(seg.end, 2),
                       "text": seg.text.strip()}
                      for idx, seg in enumerate(segments, 1)]
        out_q.put(("done", {
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 3),
            "audio_duration_s": round(info.duration, 2),
            "transcribe_wall_s": round(time.time() - t0, 2),
            "utterances": utterances,
        }))


class _Worker:
    def __init__(self, size, compute_type, lang_code):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.in_q, self.out_q = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_decode_worker, daemon=True,
                                args=(size, compute_type, lang_code,
                                      self.in_q, self.out_q))
        import queue
        t0 = time.time()
        self.proc.start()
        while True:                                 # wait for "ready"
            try:
                self.out_q.get(timeout=2)
                break
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError("decode worker died while loading the model")
                if time.time() - t0 > 600:
                    self.kill()
                    raise RuntimeError("decode worker did not load the model in 600s")
        print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    def kill(self):
        self.proc.kill()
        self.proc.join()

    def stop(self):
        self.in_q.put(None)
        self.proc.join(timeout=30)
        if self.proc.is_alive():
            self.kill()


def run_config(size, lang_name, lang_code, rids, compute_type, resume=False,
               timeout_factor=5.0, timeout_min_s=600.0):
    import queue

    import soundfile as sf

    config_name = f"{size}_{lang_name}"
    out_dir = os.path.join(PRED_ROOT, config_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 62}")
    print(f"config: {config_name}  (size={size}, language={lang_code or 'auto-detect'})")
    print(f"{'=' * 62}", flush=True)

    # --resume skips recordings already transcribed, so a multi-hour run that
    # is interrupted picks up where it stopped instead of starting over.
    todo = [r for r in rids if not (resume and os.path.exists(
        os.path.join(out_dir, f"{r}.transcript.json")))]
    if len(todo) < len(rids):
        print(f"resuming: {len(rids) - len(todo)} already done, {len(todo)} to go")

    worker = _Worker(size, compute_type, lang_code) if todo else None

    for i, rid in enumerate(todo, 1):
        duration = sf.info(os.path.join(AUDIO_DIR, f"{rid}.wav")).duration
        # Time cap per recording. Normal decodes run at 0.1-1.2x real time,
        # so 5x (at least 10 min) only ever catches a runaway fallback loop.
        cap = max(timeout_min_s, timeout_factor * duration) if timeout_factor else None
        worker.in_q.put(rid)
        try:
            _, res = worker.out_q.get(timeout=cap)
            timed_out = False
        except queue.Empty:
            worker.kill()
            res = {"detected_language": None, "language_probability": 0.0,
                   "audio_duration_s": round(duration, 2),
                   "transcribe_wall_s": round(cap, 2), "utterances": []}
            timed_out = True
            if i < len(todo):
                worker = _Worker(size, compute_type, lang_code)

        out = {
            "recording": rid,
            "config": config_name,
            "model_size": size,
            "language_setting": lang_code or "auto",
            **{k: res[k] for k in ("detected_language", "language_probability",
                                   "audio_duration_s", "transcribe_wall_s")},
            "timed_out": timed_out,
            "utterances": res["utterances"],
        }
        with open(os.path.join(out_dir, f"{rid}.transcript.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)

        status = f"TIMED OUT after {cap:.0f}s" if timed_out else f"{res['transcribe_wall_s']:5.1f}s"
        print(f"  [{i:2}/{len(todo)}] {rid}  {len(res['utterances']):3} segs  "
              f"detected={res['detected_language']}  {status}", flush=True)

    if worker is not None:
        worker.stop()

    # Totals come from the saved files, so they cover every recording in the
    # config even when part of it was transcribed by an earlier, resumed run.
    total_audio = total_wall = 0.0
    timed_out = []
    for rid in rids:
        with open(os.path.join(out_dir, f"{rid}.transcript.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        total_audio += saved["audio_duration_s"]
        total_wall += saved["transcribe_wall_s"]
        if saved.get("timed_out"):
            timed_out.append(rid)
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
        "timeout_rule": (f"max({timeout_min_s:.0f}s, {timeout_factor}x audio)"
                         if timeout_factor else None),
        "timed_out_recordings": timed_out,
    }
    with open(os.path.join(out_dir, "_config_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["small", "medium"])
    ap.add_argument("--configs", nargs="+", default=None,
                    help="explicit config names to run, e.g. small_auto small_si")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--resume", action="store_true",
                    help="skip recordings whose prediction file already exists")
    ap.add_argument("--timeout-factor", type=float, default=5.0,
                    help="kill a recording's decode after max(600s, N x audio "
                         "length) and save it as timed out; 0 disables")
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
            metas.append(run_config(size, lang_name, lang_code, rids, a.compute_type,
                                    resume=a.resume, timeout_factor=a.timeout_factor))

    os.makedirs(os.path.join(C1_ROOT, "results"), exist_ok=True)
    out = os.path.join(C1_ROOT, "results", "c1_ablation_runs.json")
    # Merge by config name so configs run in separate invocations accumulate.
    merged = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            merged = {m["config"]: m for m in json.load(fh).get("configs", [])}
    merged.update({m["config"]: m for m in metas})
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"configs": [merged[k] for k in sorted(merged)]}, fh, indent=2)
    print(f"\nRan {len(metas)} config(s). Wrote {out}")


if __name__ == "__main__":
    main()
