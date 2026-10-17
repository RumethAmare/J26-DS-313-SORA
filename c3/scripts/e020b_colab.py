"""
E020b - diagnose and repair live mode (exploratory follow-up to E020; pre-registered in C3_LOGBOOK.md).
Runs in the SAME Colab kernel after e020_colab.py: re-uses its cached window outputs (`cache`), FULL, PANEL, TUNE.
Only the way window outputs are stitched together changes; the model calls are identical.
Variants:
  V0   = E020 as run (consistency check: must reproduce 40.65 on ALL 8 at tau 0.6)
  DET  = detection only (every local speech segment committed, one label) -> how much miss is in the windows
  V1   = no-drop: local speakers without a usable fingerprint go to the nearest mapped speaker in the same window
  V2   = 2.5 s look-ahead: commit the 5 s that END 2.5 s before the newest audio (adds 2.5 s latency)
  V3   = V1 + V2
tau in {0.4..0.7} chosen on TUNE per variant; panels scored once per variant. Adopted = lowest TUNE DER (V1-V3).
"""
import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pathlib import Path

_R = Path("/content")
def _ref(c):
    return read_rttm(_R / "rttm_truth" / f"{c}.rttm", c), read_uem(_R / "rttm_truth" / f"{c}.uem", c)

def replay(c, tau, nodrop=False, L=0.0, det=False, min_speech=0.3):
    sr, outs = cache[c]
    cents, counts = [], []
    def match(e):
        e = e / (np.linalg.norm(e) + 1e-9)
        if cents:
            C = np.stack([x / (np.linalg.norm(x) + 1e-9) for x in cents]); s = C @ e; j = int(np.argmax(s))
            if s[j] >= tau:
                cents[j] = (cents[j] * counts[j] + e) / (counts[j] + 1); counts[j] += 1; return j
        cents.append(e.copy()); counts.append(1); return len(cents) - 1
    hyp = Annotation(uri=c); k = 0; lo = 0.0; last = None
    for i, (t_end, w, ann, embs) in enumerate(outs):
        hi = t_end if i == len(outs) - 1 else max(lo, t_end - L)
        region = Segment(lo, hi)
        labels = ann.labels(); mp = {}
        for j, lab in enumerate(labels):
            ok = (embs is not None and j < len(embs) and np.all(np.isfinite(embs[j])) and np.linalg.norm(embs[j]) > 1e-6
                  and ann.label_duration(lab) >= min_speech)
            if ok:
                mp[lab] = match(np.asarray(embs[j], dtype=np.float32))
        if nodrop:
            for lab in labels:
                if lab in mp:
                    continue
                tl = ann.label_timeline(lab)
                best, bd = None, 1e9
                for l2 in mp:
                    for s2 in ann.label_timeline(l2):
                        for s1 in tl:
                            d = max(0.0, max(s1.start, s2.start) - min(s1.end, s2.end))
                            if d < bd:
                                bd, best = d, mp[l2]
                if best is None:
                    best = last if last is not None else -1
                mp[lab] = best
        for seg, _, lab in ann.itertracks(yield_label=True):
            if not det and lab not in mp:
                continue
            g = Segment(seg.start + w, seg.end + w) & region
            if g and g.duration > 0:
                k += 1
                name = "S" if det else f"G{mp[lab]}"
                hyp[g, k] = name
                if not det:
                    last = mp[lab]
        lo = hi
    hyp = hyp.support(collar=0.25)
    ref, uem = _ref(c)
    return DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref, hyp, uem=uem, detailed=True), len(hyp.labels())

def pool(cl, **kw):
    m = f = q = t = 0.0; n = {}
    for c in cl:
        d, ns = replay(c, **kw); n[c] = ns
        m += d["missed detection"]; f += d["false alarm"]; q += d["confusion"]; t += d["total"]
    return 100 * (m + f + q) / t, 100 * m / t, 100 * f / t, 100 * q / t, n

ALL8 = [c for cs in PANEL.values() for c in cs]
L2 = ["# E020b - live-mode diagnosis (exploratory), Colab, same cached windows as E020", ""]
v0 = pool(ALL8, tau=0.6)
L2.append(f"V0 consistency check (E020 settings, tau 0.6): ALL 8 DER {v0[0]:.2f} (E020 logged 40.65)")
det = pool(ALL8, tau=0.6, det=True)
L2.append(f"DET detection only (no speaker labels): miss {det[1]:.2f} fa {det[2]:.2f}  "
          f"(full-recording miss 3.83 fa 5.57)")
L2 += ["", "| variant | tau (TUNE) | TUNE DER | ALL 8 DER | miss | fa | wrong person | TEST | TEST2 | TEST3 | speakers found |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
best = None
for name, kw in (("V1 no-drop", dict(nodrop=True)), ("V2 look-ahead 2.5 s", dict(L=2.5)),
                 ("V3 no-drop + look-ahead", dict(nodrop=True, L=2.5))):
    tun = {tau: pool(TUNE, tau=tau, **kw)[0] for tau in (0.4, 0.5, 0.6, 0.7)}
    tau = min(tun, key=tun.get)
    a = pool(ALL8, tau=tau, **kw)
    per = {s: pool(cl, tau=tau, **kw)[0] for s, cl in PANEL.items()}
    L2.append(f"| {name} | {tau} | {tun[tau]:.2f} | {a[0]:.2f} | {a[1]:.2f} | {a[2]:.2f} | {a[3]:.2f} | "
              f"{per['TEST']:.2f} | {per['TEST2']:.2f} | {per['TEST3']:.2f} | {a[4]} |")
    if best is None or tun[tau] < best[1]:
        best = (name, tun[tau], a[0])
L2.append(f"\nAdopted by rule (lowest TUNE DER): {best[0]} -> ALL 8 live DER {best[2]:.2f} vs full-recording 21.27")
txt = "\n".join(L2)
print(txt)
out = Path("/content/drive/MyDrive/C3_E009/e020"); out.mkdir(parents=True, exist_ok=True)
(out / "E020b.md").write_text(txt + "\n", encoding="utf-8")
print("=== E020b finished ===")
