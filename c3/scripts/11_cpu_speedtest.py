"""
E017 - CPU speed test for the PP1 demo laptop (no GPU).

Question: can community-1 run the demo on this laptop's CPU, fully offline?
  1. Batch mode: time 3 benchmark clips (0.6 / 4.4 / 6.6 min) and check the DER
     matches the earlier GPU numbers (same model -> should be ~the same).
  2. Live-mode feasibility: time the pipeline on short 5 s and 10 s chunks.
     Live mode is only realistic if a chunk finishes well inside its own length.

Run (from the "C3 Pilot" folder, with the venv active):
    python scripts\\11_cpu_speedtest.py                 # first run: downloads the model once
    python scripts\\11_cpu_speedtest.py --offline       # proves it runs with NO internet

Token: read from the HF_TOKEN environment variable. Never written in this file.
Results: results\\E017_cpu_speed\\<computer>_<online|offline>.md
"""
import argparse, os, sys, time, platform, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--offline", action="store_true", help="block all internet access to Hugging Face")
ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
args = ap.parse_args()
if args.offline:
    os.environ["HF_HUB_OFFLINE"] = "1"          # must be set before pyannote/huggingface import

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

ROOT = Path(__file__).resolve().parent.parent           # ...\C3 Pilot
CLIPS = ["IRD_Harshana_Silva", "IRD_Recording_2_IIT", "R0046"]
GPU_REF = {"IRD_Harshana_Silva": 27.99, "IRD_Recording_2_IIT": 4.87, "R0046": 21.95}  # 1 Oct demo, automatic, c0.0
MODEL = "pyannote/speaker-diarization-community-1"

if args.threads:
    torch.set_num_threads(args.threads)


def read_rttm(path, uri):
    ann = Annotation(uri=uri)
    for i, line in enumerate(open(path, encoding="utf-8")):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER" and float(p[4]) > 0:
            ann[Segment(float(p[3]), float(p[3]) + float(p[4])), f"t{i}"] = p[7]
    return ann


def read_uem(path, uri):
    tl = Timeline(uri=uri)
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 4:
            tl.add(Segment(float(p[2]), float(p[3])))
    return tl


def load_wave(path):
    # Load in Python and pass the samples in memory: avoids Windows FFmpeg/torchcodec issues.
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)                              # mono
    return torch.from_numpy(data).unsqueeze(0), sr


def run(pipe, wav, sr):
    out = pipe({"waveform": wav, "sample_rate": sr})
    return getattr(out, "speaker_diarization", out)


lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(s)

host = platform.node() or "pc"
log(f"# E017 CPU speed test - {host} - {'OFFLINE' if args.offline else 'online'}")
log("")
log(f"- CPU: {platform.processor() or platform.machine()} | logical cores: {os.cpu_count()} | torch threads: {torch.get_num_threads()}")
log(f"- GPU available: {torch.cuda.is_available()} (this test forces CPU)")
log(f"- Python {platform.python_version()} | torch {torch.__version__} | {platform.platform()}")
import pyannote.audio; log(f"- pyannote.audio {pyannote.audio.__version__}")
log("")

tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not tok and not args.offline:
    sys.exit('No HF_TOKEN. In this terminal run:  set HF_TOKEN=hf_xxxx   (do not paste it into any file)')

t0 = time.time()
try:
    pipe = Pipeline.from_pretrained(MODEL, token=tok) if tok else Pipeline.from_pretrained(MODEL)
except Exception as e:
    sys.exit(f"Model load failed: {type(e).__name__}: {e}\n"
             "If --offline: run once WITHOUT --offline first so the model is cached.")
if pipe is None:
    sys.exit("Model came back empty: accept the community-1 terms on huggingface.co with this token's account.")
pipe.to(torch.device("cpu"))
log(f"Model load: {time.time() - t0:.1f} s")

# warm-up (first call is always slower)
w, sr = load_wave(ROOT / "processed" / f"{CLIPS[0]}.wav")
t0 = time.time(); run(pipe, w[:, : 5 * sr], sr); log(f"Warm-up (5 s audio): {time.time() - t0:.1f} s")
log("")

# ---------- 1. batch mode ----------
log("## 1. Batch mode (record/upload, then diarize)")
log("")
log("| clip | audio (s) | processing (s) | x real time | speakers found | DER c0.0 (CPU) | DER GPU 1 Oct |")
log("|---|---|---|---|---|---|---|")
results = {}
for c in CLIPS:
    w, sr = load_wave(ROOT / "processed" / f"{c}.wav")
    dur = w.shape[1] / sr
    t0 = time.time(); ann = run(pipe, w, sr); el = time.time() - t0
    ann.uri = c
    der = ""
    tr, ue = ROOT / "rttm_truth" / f"{c}.rttm", ROOT / "rttm_truth" / f"{c}.uem"
    if tr.exists() and ue.exists():
        der = 100 * DiarizationErrorRate(collar=0.0, skip_overlap=False)(read_rttm(tr, c), ann, uem=read_uem(ue, c))
    results[c] = dict(audio_s=round(dur, 1), proc_s=round(el, 1), rt=round(dur / el, 2),
                      speakers=len(ann.labels()), der=round(der, 2) if der != "" else None)
    log(f"| {c} | {dur:.1f} | {el:.1f} | {dur / el:.2f}x | {len(ann.labels())} | "
        f"{der:.2f} % | {GPU_REF.get(c, '')} % |" if der != "" else
        f"| {c} | {dur:.1f} | {el:.1f} | {dur / el:.2f}x | {len(ann.labels())} | n/a | |")
log("")
log("x real time > 1 means faster than the audio plays. A 60 s recording at 2x real time waits 30 s.")
log("")

# ---------- 2. live-mode feasibility ----------
log("## 2. Live-mode feasibility (short chunks)")
log("")
w, sr = load_wave(ROOT / "processed" / "IRD_Recording_2_IIT.wav")
log("| chunk length | chunks timed | mean processing (s) | worst (s) | keeps up? |")
log("|---|---|---|---|---|")
live = {}
for L in (5, 10):
    times = []
    for k in range(6):
        seg = w[:, k * L * sr: (k + 1) * L * sr]
        t0 = time.time(); run(pipe, seg, sr); times.append(time.time() - t0)
    m, mx = float(np.mean(times)), float(np.max(times))
    verdict = "yes, comfortably" if mx < 0.5 * L else ("borderline" if mx < L else "NO, falls behind")
    live[L] = dict(mean=round(m, 2), worst=round(mx, 2), verdict=verdict)
    log(f"| {L} s | {len(times)} | {m:.2f} | {mx:.2f} | {verdict} |")
log("")
log("'Keeps up' = each chunk must finish before the next one has been recorded; comfortably = under half its length.")

out_dir = ROOT / "results" / "E017_cpu_speed"
out_dir.mkdir(parents=True, exist_ok=True)
tag = f"{host}_{'offline' if args.offline else 'online'}"
(out_dir / f"{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(out_dir / f"{tag}.json").write_text(json.dumps(dict(batch=results, live=live), indent=2), encoding="utf-8")
print(f"\nSaved: {out_dir / (tag + '.md')}")
