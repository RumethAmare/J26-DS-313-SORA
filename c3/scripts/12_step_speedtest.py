"""
E018 - Can community-1 run faster on the laptop CPU by sliding its 10 s window in bigger steps?

Default step = 0.1 x 10 s = 1 s (each second is analysed ~10 times).
Tested: 0.1 (default, reference), 0.25, 0.5. Same 3 clips as E017.
Measures processing time AND DER, so the speed-vs-accuracy trade-off is visible.
Runs OFFLINE (model already cached by E017). No token needed.
"""
import os, time, json, platform
os.environ["HF_HUB_OFFLINE"] = "1"
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import soundfile as sf, torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ["IRD_Harshana_Silva", "IRD_Recording_2_IIT", "R0046"]
STEPS = [0.1, 0.25, 0.5]

def read_rttm(p, uri):
    a = Annotation(uri=uri)
    for i, l in enumerate(open(p, encoding="utf-8")):
        x = l.split()
        if len(x) >= 8 and x[0] == "SPEAKER" and float(x[4]) > 0:
            a[Segment(float(x[3]), float(x[3]) + float(x[4])), f"t{i}"] = x[7]
    return a

def read_uem(p, uri):
    t = Timeline(uri=uri)
    for l in open(p, encoding="utf-8"):
        x = l.split()
        if len(x) >= 4: t.add(Segment(float(x[2]), float(x[3])))
    return t

def load(p):
    d, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return torch.from_numpy(d.mean(axis=1)).unsqueeze(0), sr

lines = []
def log(s=""):
    print(s, flush=True); lines.append(s)

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
pipe.to(torch.device("cpu"))
seg = pipe._segmentation
log(f"# E018 step test - {platform.node()} - OFFLINE")
log(f"\n- window {seg.duration} s, default step {seg.step} s, torch threads {torch.get_num_threads()}")
w, sr = load(ROOT / "processed" / f"{CLIPS[0]}.wav"); pipe({"waveform": w[:, :5*sr], "sample_rate": sr})  # warm-up

data = {c: load(ROOT / "processed" / f"{c}.wav") for c in CLIPS}
tot_audio = sum(w.shape[1] / sr for w, sr in data.values())
log("\n| step (x window) | step (s) | clip | audio (s) | processing (s) | x real time | speakers | DER c0.0 |")
log("|---|---|---|---|---|---|---|---|")
summary = {}
for frac in STEPS:
    seg.step = frac * seg.duration
    tot_t, errs = 0.0, []
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    for c in CLIPS:
        w, sr = data[c]; dur = w.shape[1] / sr
        t0 = time.time(); out = pipe({"waveform": w, "sample_rate": sr}); el = time.time() - t0
        ann = getattr(out, "speaker_diarization", out); ann.uri = c
        der = 100 * metric(read_rttm(ROOT/"rttm_truth"/f"{c}.rttm", c), ann, uem=read_uem(ROOT/"rttm_truth"/f"{c}.uem", c))
        tot_t += el
        log(f"| {frac} | {seg.step:.1f} | {c} | {dur:.1f} | {el:.1f} | {dur/el:.2f}x | {len(ann.labels())} | {der:.2f} % |")
    pooled = 100 * abs(metric)
    summary[frac] = dict(total_proc_s=round(tot_t, 1), x_rt=round(tot_audio / tot_t, 2), pooled_der=round(pooled, 2))
    log(f"| **{frac} total** | | 3 clips | {tot_audio:.0f} | **{tot_t:.0f}** | **{tot_audio/tot_t:.2f}x** | | **pooled {pooled:.2f} %** |")

log("\nPooled DER = all 3 clips together. Only 3 clips (11.7 min): a speed result, not an accuracy verdict.")
out = ROOT / "results" / "E018_step_speed"; out.mkdir(parents=True, exist_ok=True)
(out / "E018.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(out / "E018.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
log(f"\nSaved: {out / 'E018.md'}\n=== Finished. ===")
