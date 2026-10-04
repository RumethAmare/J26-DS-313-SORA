#!/usr/bin/env python3
"""
benchmark_latency.py -- Task 7: quantization and latency.

Two parts:

1. ASR. Sweeps faster-whisper model size x compute_type on the GPU, plus int8
   on the CPU as the no-GPU deployment baseline. Per configuration it records
   model load time, real-time factor (wall time / audio time, lower is
   faster), peak GPU memory, peak process RAM and on-disk model size.

   Decoding uses the token-stream pipeline's settings (beam 5, VAD, word
   timestamps), so the latency is what C1 actually pays, not a stripped-down
   best case.

   WHY NOT WER: clean-audio WER is ~0.99 (c1_report.md), so it cannot
   register the small damage quantization does. Each quantized transcript is
   instead scored against the SAME model's float16 transcript ("drift"):
   0 means quantization changed nothing.

2. fastText LID. Before/after `quantize()` on the Task 3 model, with
   cross-validated accuracy so the size saving is set against its cost.

Outputs results/c1_latency.json and results/c1_latency_report.md.

Usage:
    python benchmark_latency.py                 # full sweep
    python benchmark_latency.py --skip-asr      # fastText part only
"""
import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cuda_env  # noqa: E402

cuda_env.ensure_cuda_libs()

import corpus  # noqa: E402

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(C1_ROOT, "results")

# Five of Task 6's stratified noise-subset recordings (LATIN / MIXED / SINHALA
# gold-script buckets), ~8.5 minutes of audio: long enough that per-call
# overhead does not dominate RTF, short enough to run the CPU arm.
BENCH_RECORDINGS = ["J26DS313_R0002", "J26DS313_R0009", "J26DS313_R0014",
                    "J26DS313_R0019", "J26DS313_R0023"]

GPU_CONFIGS = [(size, ct) for size in ("small", "medium")
               for ct in ("float16", "int8_float16", "int8")]
CPU_CONFIGS = [("small", "int8"), ("medium", "int8")]


# --- resource sampling -------------------------------------------------------

class PeakSampler:
    """Poll GPU memory and process RSS in the background, keep the maxima.

    GPU memory is read as device-wide `memory.used` via nvidia-smi, because
    per-process accounting reports N/A under the Windows WDDM driver. A
    baseline taken before the model loads is subtracted, so the figure is
    this model's footprint as long as nothing else starts using the GPU
    mid-run.
    """

    def __init__(self, interval=0.2, gpu=True):
        import psutil
        self.proc = psutil.Process()
        self.interval = interval
        self.gpu = gpu
        self.peak_rss = 0
        self.peak_gpu = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def gpu_used_mib():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip().splitlines()[0])
        except Exception:                                     # noqa: BLE001
            return 0

    def _run(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
            if self.gpu:
                self.peak_gpu = max(self.peak_gpu, self.gpu_used_mib())
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def model_dir_size_mb(size):
    from huggingface_hub import snapshot_download
    path = snapshot_download(f"Systran/faster-whisper-{size}")
    total = sum(os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(path) for f in files)
    return round(total / 2**20, 1)


# --- ASR sweep ---------------------------------------------------------------

def transcribe_text(model, audio):
    segments, info = model.transcribe(
        audio, beam_size=5, vad_filter=True, language=None, word_timestamps=True)
    return " ".join(s.text.strip() for s in segments), info


def run_config(size, compute_type, device, audio_by_rid):
    from faster_whisper import WhisperModel

    gc.collect()
    time.sleep(1.0)
    gpu_base = PeakSampler.gpu_used_mib() if device == "cuda" else 0

    with PeakSampler(gpu=(device == "cuda")) as sampler:
        t0 = time.perf_counter()
        model = WhisperModel(size, device=device, compute_type=compute_type,
                             cpu_threads=os.cpu_count() if device == "cpu" else 0)
        load_s = time.perf_counter() - t0

        # Warm-up on 10s of audio: first-call costs (cuDNN init, kernel JIT
        # for GPUs newer than the CUDA build) are paid once per process and
        # are not decoding speed.
        first = audio_by_rid[BENCH_RECORDINGS[0]]
        transcribe_text(model, first[: 16000 * 10])

        per_rec, texts = [], {}
        for rid in BENCH_RECORDINGS:
            audio = audio_by_rid[rid]
            dur = len(audio) / 16000
            t0 = time.perf_counter()
            text, info = transcribe_text(model, audio)
            wall = time.perf_counter() - t0
            texts[rid] = text
            per_rec.append({"recording": rid, "audio_s": round(dur, 2),
                            "wall_s": round(wall, 2), "rtf": round(wall / dur, 4),
                            "language": info.language,
                            "n_words": len(text.split())})

    del model
    gc.collect()

    audio_total = sum(r["audio_s"] for r in per_rec)
    wall_total = sum(r["wall_s"] for r in per_rec)
    return {
        "model": size, "compute_type": compute_type, "device": device,
        "load_s": round(load_s, 2),
        "audio_s": round(audio_total, 1),
        "wall_s": round(wall_total, 1),
        "rtf": round(wall_total / audio_total, 4),
        "rtf_per_recording_median": round(statistics.median(r["rtf"] for r in per_rec), 4),
        "peak_gpu_mib": (sampler.peak_gpu - gpu_base) if device == "cuda" else None,
        "peak_rss_mib": round(sampler.peak_rss / 2**20),
        "per_recording": per_rec,
    }, texts


def drift(ref_texts, hyp_texts):
    """WER of a transcript against the same model's float16 transcript."""
    import jiwer

    import script_normalize as sn
    refs, hyps = [], []
    for rid in BENCH_RECORDINGS:
        r = sn.normalize_scripted(ref_texts[rid])
        h = sn.normalize_scripted(hyp_texts[rid])
        if r.strip():
            refs.append(r)
            hyps.append(h)
    return round(jiwer.wer(refs, hyps), 4) if refs else None


def asr_sweep():
    from faster_whisper import decode_audio

    audio_by_rid = {rid: decode_audio(os.path.join(corpus.AUDIO_DIR, f"{rid}.wav"))
                    for rid in BENCH_RECORDINGS}
    configs = [(s, c, "cuda") for s, c in GPU_CONFIGS] + \
              [(s, c, "cpu") for s, c in CPU_CONFIGS]

    results, texts = [], {}
    for size, ct, dev in configs:
        print(f"\n== {size} / {ct} / {dev}", flush=True)
        res, txt = run_config(size, ct, dev, audio_by_rid)
        res["disk_mb"] = model_dir_size_mb(size)
        results.append(res)
        texts[(size, ct, dev)] = txt
        print(f"   load {res['load_s']}s  RTF {res['rtf']}  "
              f"GPU {res['peak_gpu_mib']} MiB  RSS {res['peak_rss_mib']} MiB", flush=True)

    for res in results:
        ref = texts[(res["model"], "float16", "cuda")]
        hyp = texts[(res["model"], res["compute_type"], res["device"])]
        res["drift_vs_float16"] = drift(ref, hyp)
    return results


# --- fastText ----------------------------------------------------------------

def fasttext_quantization():
    """Cross-validated Latin-script accuracy before and after quantize()."""
    import lid_data
    import train_lid_fasttext as trainer

    tokens = lid_data.load_gold_tokens()
    recordings = sorted({t["recording"] for t in tokens})
    by_rec = {r: [t for t in tokens if t["recording"] == r] for r in recordings}

    def en_ratio(rid):
        latin = [t for t in by_rec[rid] if t["script"] == "latin"]
        return sum(1 for t in latin if t["lang"] == "EN") / len(latin) if latin else 0.0

    folds = lid_data.make_folds(recordings, k=5, key=en_ratio)
    params = {"epoch": 50, "lr": 0.5, "dim": 50, "minn": 2, "maxn": 5,
              "wordNgrams": 1, "minCount": 1, "loss": "softmax",
              "seed": 13, "thread": 1, "bucket": 50000}
    workdir = tempfile.mkdtemp(prefix="c1_ftq_")

    def acc(model, toks):
        return sum(1 for t in toks
                   if trainer.predict_ft(model, t["token"])[0] == t["lang"]) / len(toks)

    def timed_predict_us(model, toks, reps=3):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            for t in toks:
                trainer.predict_ft(model, t["token"])
            best = min(best, time.perf_counter() - t0)
        return best / len(toks) * 1e6

    full, quant, sizes_full, sizes_q, lat_full, lat_q = [], [], [], [], [], []
    for i, test_recs in enumerate(folds):
        train_latin = [t for r in recordings if r not in set(test_recs)
                       for t in by_rec[r] if t["script"] == "latin"]
        test_latin = [t for r in test_recs for t in by_rec[r] if t["script"] == "latin"]
        model = trainer.train_fold(train_latin, params, workdir)
        path_full = os.path.join(workdir, f"fold{i}.bin")
        model.save_model(path_full)
        full.append(acc(model, test_latin))
        lat_full.append(timed_predict_us(model, test_latin))
        model.quantize(input=os.path.join(workdir, "train.txt"), retrain=False)
        path_q = os.path.join(workdir, f"fold{i}.ftz")
        model.save_model(path_q)
        quant.append(acc(model, test_latin))
        lat_q.append(timed_predict_us(model, test_latin))
        sizes_full.append(os.path.getsize(path_full))
        sizes_q.append(os.path.getsize(path_q))
        print(f"  fold {i}: acc {full[-1]:.4f} -> {quant[-1]:.4f}  "
              f"size {sizes_full[-1] / 2**20:.1f} -> {sizes_q[-1] / 2**20:.2f} MB", flush=True)

    def ms(v):
        return round(statistics.mean(v), 4), round(statistics.stdev(v), 4)

    return {
        "n_recordings": len(recordings),
        "folds": 5,
        "params": params,
        "accuracy_full": ms(full),
        "accuracy_quantized": ms(quant),
        "accuracy_delta_per_fold": [round(q - f, 4) for f, q in zip(full, quant)],
        "size_full_mb": round(statistics.mean(sizes_full) / 2**20, 2),
        "size_quantized_mb": round(statistics.mean(sizes_q) / 2**20, 2),
        "predict_us_per_token_full": round(statistics.mean(lat_full), 2),
        "predict_us_per_token_quantized": round(statistics.mean(lat_q), 2),
    }


# --- report ------------------------------------------------------------------

def write_report(asr, ft, env, path):
    L = ["# Task 7 — Quantization and latency\n\n",
         "Produced by `scripts/benchmark_latency.py`; raw numbers in "
         "`c1_latency.json`.\n\n",
         f"Hardware: {env['gpu']}, {env['cpu']} ({env['cpu_threads']} threads), "
         f"{env['os']}. faster-whisper {env['faster_whisper']}, "
         f"CTranslate2 {env['ctranslate2']}.\n\n"]

    if asr:
        audio_min = asr[0]["audio_s"] / 60
        L += ["## ASR (faster-whisper / CTranslate2)\n\n",
              f"{len(BENCH_RECORDINGS)} recordings, {audio_min:.1f} minutes of audio, "
              "decoded with the token-stream settings (beam 5, VAD, word timestamps, "
              "auto language). One untimed 10-second warm-up per configuration.\n\n",
              "| model | compute type | device | load (s) | RTF | × real-time "
              "| peak GPU (MiB) | peak RAM (MiB) | disk (MB) | drift vs fp16 |\n",
              "|---|---|---|---|---|---|---|---|---|---|\n"]
        for r in asr:
            gpu = r["peak_gpu_mib"] if r["peak_gpu_mib"] is not None else "—"
            d = r["drift_vs_float16"]
            d = "ref" if (r["compute_type"] == "float16" and r["device"] == "cuda") else (
                f"{d:.3f}" if d is not None else "—")
            L.append(f"| {r['model']} | {r['compute_type']} | {r['device']} "
                     f"| {r['load_s']:.1f} | {r['rtf']:.3f} | {1 / r['rtf']:.1f}× "
                     f"| {gpu} | {r['peak_rss_mib']} | {r['disk_mb']:.0f} | {d} |\n")
        L += ["\n- **RTF** = wall time ÷ audio duration; below 1 is faster than "
              "real time.\n",
              "- **drift** = WER of the transcript against the same model's GPU "
              "float16 transcript, after script normalisation. It isolates what "
              "quantization changes; clean-audio WER against gold (~0.99) is too "
              "saturated to show it.\n",
              "- **disk** is the downloaded checkpoint (float16 weights). "
              "CTranslate2 converts to the requested compute type at load time, so "
              "int8 saves memory, not download size; an int8-at-rest copy would "
              "need re-conversion with `ct2-transformers-converter --quantization "
              "int8`.\n",
              "- GPU memory is device-wide usage above a pre-load baseline (per-process "
              "accounting is unavailable under Windows WDDM).\n\n"]

    if ft:
        L += ["## fastText LID (Task 3 model)\n\n",
              f"5-fold cross-validation folded by recording, {ft['n_recordings']} "
              "recordings, Latin-script tokens.\n\n",
              "| | accuracy | size | predict time / token |\n|---|---|---|---|\n",
              f"| full | {ft['accuracy_full'][0]:.4f} ± {ft['accuracy_full'][1]:.4f} "
              f"| {ft['size_full_mb']:.2f} MB | {ft['predict_us_per_token_full']:.1f} µs |\n",
              f"| quantized | {ft['accuracy_quantized'][0]:.4f} ± "
              f"{ft['accuracy_quantized'][1]:.4f} | {ft['size_quantized_mb']:.2f} MB "
              f"| {ft['predict_us_per_token_quantized']:.1f} µs |\n\n",
              f"Per-fold accuracy change from quantizing: "
              f"{', '.join(f'{d:+.4f}' for d in ft['accuracy_delta_per_fold'])}.\n"]

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(L)


def environment():
    import ctranslate2
    import faster_whisper
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except Exception:                                         # noqa: BLE001
        gpu = "no GPU"
    return {"gpu": gpu, "cpu": platform.processor() or platform.machine(),
            "cpu_threads": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version(),
            "faster_whisper": faster_whisper.__version__,
            "ctranslate2": ctranslate2.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--skip-fasttext", action="store_true")
    a = ap.parse_args()

    env = environment()
    out_json = os.path.join(RESULTS_DIR, "c1_latency.json")
    prev = {}
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as fh:
            prev = json.load(fh)

    asr = prev.get("asr")
    if not a.skip_asr:
        asr = asr_sweep()
    ft = prev.get("fasttext")
    if not a.skip_fasttext:
        print("\n== fastText quantization", flush=True)
        ft = fasttext_quantization()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"environment": env, "recordings": BENCH_RECORDINGS,
                   "asr": asr, "fasttext": ft}, fh, indent=2, ensure_ascii=False)
    write_report(asr, ft, env, os.path.join(RESULTS_DIR, "c1_latency_report.md"))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
