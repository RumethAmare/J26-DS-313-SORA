"""
E020 - How good is LIVE (chunked) mode compared with the normal full-recording mode? Runs offline on the laptop CPU.

The live algorithm (live.py) is replayed on benchmark files exactly as if the audio arrived from a microphone:
every STEP s the model sees only the last WINDOW s. Window outputs do not depend on the matching threshold tau, so
they are computed once per clip and the matching is replayed for each tau (cheap).
tau is chosen on scripted TUNE clips, then scored ONCE on the 8 real panel clips (never used for choosing).
"""
import os, sys, time, json, platform
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, soundfile as sf, torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "c3_app"))
from live import LiveDiarizer

ROOT = Path(__file__).resolve().parent.parent
TUNE = ["R0023", "R0024", "R0025", "R0026", "J26DS313_R0011", "J26DS313_R0012"]
PANEL = {"TEST": ["IRD_Harshana_Silva", "IRD_Recording_2_IIT"], "TEST2": ["Yt01", "YT02", "YT03"],
         "TEST3": ["YT04", "YT05", "YT07"]}
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7]
WINDOW, STEP = 10.0, 5.0


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


pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1"); pipe.to(torch.device("cpu"))
lines = []
def log(s=""):
    print(s, flush=True); lines.append(s)

log(f"# E020 live-mode simulation - {platform.node()} - window {WINDOW}s, step {STEP}s, CPU, offline")
cache, cpu = {}, []
clips = TUNE + [c for cs in PANEL.values() for c in cs]
t_all = time.time()
for c in clips:
    data, sr = sf.read(str(ROOT / "processed" / f"{c}.wav"), dtype="float32", always_2d=True)
    d = LiveDiarizer(pipe, sr=sr, window=WINDOW, step=STEP)
    d.buf = data.mean(axis=1)
    ends = list(np.arange(STEP, d.now + 1e-9, STEP))
    if d.now - (ends[-1] if ends else 0) > 0.5:
        ends.append(d.now)
    outs = []
    for t_end in ends:
        w, ann, embs, secs = d.window_output(float(t_end))
        outs.append((float(t_end), w, ann, None if embs is None else np.asarray(embs)))
        cpu.append(secs)
    cache[c] = (sr, outs)
    print(f"  {c}: {len(outs)} windows, {d.now:.0f}s audio [{time.time() - t_all:.0f}s]", flush=True)

log(f"\nCPU per {WINDOW:.0f}-s window: mean {np.mean(cpu):.2f}s, 95th pct {np.percentile(cpu, 95):.2f}s, "
    f"worst {np.max(cpu):.2f}s (must stay under the {STEP:.0f}-s step to keep up)")


def run(c, tau):
    sr, outs = cache[c]
    d = LiveDiarizer(None, sr=sr, window=WINDOW, step=STEP, tau=tau)
    for t_end, w, ann, embs in outs:
        d.commit(t_end, w, ann, embs)
    hyp = d.result(); hyp.uri = c
    ref, uem = read_rttm(ROOT / "rttm_truth" / f"{c}.rttm", c), read_uem(ROOT / "rttm_truth" / f"{c}.uem", c)
    return DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref, hyp, uem=uem, detailed=True), len(hyp.labels())


def offline(cl):
    """Full-recording community-1 output (stored drafts in rttm_draft/), same scoring."""
    m = f = k = t = 0.0
    for c in cl:
        ref, uem = read_rttm(ROOT / "rttm_truth" / f"{c}.rttm", c), read_uem(ROOT / "rttm_truth" / f"{c}.uem", c)
        dd = DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref, read_rttm(ROOT / "rttm_draft" / f"{c}.rttm", c),
                                                                 uem=uem, detailed=True)
        m += dd["missed detection"]; f += dd["false alarm"]; k += dd["confusion"]; t += dd["total"]
    return 100 * (m + f + k) / t


def pooled(cl, tau):
    m = f = k = t = 0.0; n = {}
    for c in cl:
        dd, ns = run(c, tau); n[c] = ns
        m += dd["missed detection"]; f += dd["false alarm"]; k += dd["confusion"]; t += dd["total"]
    return 100 * (m + f + k) / t, 100 * m / t, 100 * f / t, 100 * k / t, n


log("\n## Choosing tau on scripted TUNE clips (" + ", ".join(TUNE) + ")\n")
log("| tau | TUNE DER % | miss | fa | wrong person |"); log("|---|---|---|---|---|")
best = None
for tau in TAUS:
    d, m, f, k, _ = pooled(TUNE, tau)
    log(f"| {tau} | {d:.2f} | {m:.2f} | {f:.2f} | {k:.2f} |")
    if best is None or d < best[1]:
        best = (tau, d)
tau = best[0]
log(f"\nChosen tau = {tau} (lowest TUNE DER). Panel clips scored ONCE below.\n")
log("| set | live DER % | miss | fa | wrong person | full-recording DER % (same model) | speakers found per clip |")
log("|---|---|---|---|---|---|---|")
summary = {}
for s, cl in PANEL.items():
    d, m, f, k, n = pooled(cl, tau)
    summary[s] = d
    log(f"| {s} | {d:.2f} | {m:.2f} | {f:.2f} | {k:.2f} | {offline(cl):.2f} | {n} |")
allp = pooled([c for cs in PANEL.values() for c in cs], tau)
log(f"| ALL 8 | {allp[0]:.2f} | {allp[1]:.2f} | {allp[2]:.2f} | {allp[3]:.2f} | {offline([c for cs in PANEL.values() for c in cs]):.2f} | |")
out = ROOT / "results" / "E020_live_sim"; out.mkdir(parents=True, exist_ok=True)
(out / "E020.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
log(f"\nSaved: {out / 'E020.md'}\n=== Finished. ===")
