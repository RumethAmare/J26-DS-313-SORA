"""
C3 CONSOLE - one window, one button.

Does step 4 and step 5 for you:
    corrected_labels\*.txt   ->   rttm_truth\*.rttm .uem .events.jsonl
    then scores them against rttm_draft\ and shows the table.

Run it by double-clicking RUN_C3.bat (or: python C3_Console.py).
Needs nothing installed - no pip, no ffmpeg. Python's own libraries only.

Scoring uses pyannote.metrics if you have it. If you don't, it falls back to a
built-in copy of the same metric, verified to 0.00000000 percentage points
against pyannote on all 8 previously-scored clips (DER and all three
components, both collars). The window tells you which one it used.

05_score.py stays the tool you cite in the thesis. This is the fast loop.
"""

import io
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

BASE = Path(__file__).resolve().parent
SCRIPTS = BASE / "scripts"
COLLARS = [0.0, 0.25]
MIN_SPK = 2.0          # a hypothesis label shorter than this is a fragment,
                       # not a speaker  (the fix flagged in the logbook)


# ----------------------------------------------------------------- reading

def read_rttm(path):
    segs = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            s, d = float(p[3]), float(p[4])
            if d > 0:
                segs.append((s, s + d, p[7]))
    return sorted(segs)


def read_uem(path):
    out = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 4:
            out.append((float(p[2]), float(p[3])))
    return out


# ------------------------------------------------- metric (verified copy)

def _support(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _subtract(regs, holes):
    res = []
    for rs, re_ in regs:
        pieces = [(rs, re_)]
        for hs, he in holes:
            nxt = []
            for ps, pe in pieces:
                if he <= ps or hs >= pe:
                    nxt.append((ps, pe))
                    continue
                if hs > ps:
                    nxt.append((ps, hs))
                if he < pe:
                    nxt.append((he, pe))
            pieces = nxt
        res.extend(p for p in pieces if p[1] - p[0] > 1e-9)
    return res


def _clip(segs, regs):
    out = []
    for s, e, l in segs:
        for rs, re_ in regs:
            lo, hi = max(s, rs), min(e, re_)
            if hi - lo > 1e-9:
                out.append((lo, hi, l))
    return out


def _assign(matrix):
    """Hungarian, maximising. matrix[i][j] = co-occurrence seconds."""
    n, m = len(matrix), len(matrix[0])
    flip = n > m
    if flip:
        matrix = [[matrix[i][j] for i in range(n)] for j in range(m)]
        n, m = m, n
    cost = [[-matrix[i][j] for j in range(m)] for i in range(n)]
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = []
    for j in range(1, m + 1):
        if p[j] > 0:
            a, b = p[j] - 1, j - 1
            pairs.append((b, a) if flip else (a, b))
    return pairs


def score_builtin(ref, hyp, uem, collar):
    scored = list(uem)
    if collar > 0:
        half = collar / 2.0
        holes = []
        for s, e, _ in ref:
            holes.append((s - half, s + half))
            holes.append((e - half, e + half))
        scored = _subtract(scored, _support(holes))

    R, H = _clip(ref, scored), _clip(hyp, scored)
    pts = sorted({round(v, 9)
                  for a, b, _ in R + H for v in (a, b)}
                 | {round(v, 9) for a, b in scored for v in (a, b)})
    rl = sorted({l for _, _, l in R})
    hl = sorted({l for _, _, l in H})
    ri = {l: i for i, l in enumerate(rl)}
    hi = {l: i for i, l in enumerate(hl)}

    spans, cooc = [], {}
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        d = b - a
        if d <= 1e-9:
            continue
        m = (a + b) / 2
        if not any(s <= m <= e for s, e in scored):
            continue
        ra = {l for s, e, l in R if s <= m <= e}
        ha = {l for s, e, l in H if s <= m <= e}
        spans.append((d, ra, ha))
        for x in ra:
            for y in ha:
                k = (ri[x], hi[y])
                cooc[k] = cooc.get(k, 0.0) + d

    mapping = {}
    if rl and hl:
        M = [[cooc.get((a, b), 0.0) for b in range(len(hl))]
             for a in range(len(rl))]
        for a, b in _assign(M):
            if M[a][b] > 0:
                mapping[hl[b]] = rl[a]

    total = miss = fa = conf = 0.0
    for d, ra, ha in spans:
        nr, nh = len(ra), len(ha)
        nc = sum(1 for y in ha if mapping.get(y) in ra)
        total += d * nr
        miss += d * max(0, nr - nh)
        fa += d * max(0, nh - nr)
        conf += d * (min(nr, nh) - nc)
    return total, miss, fa, conf


def score_pyannote(ref_p, hyp_p, uem_p, collar):
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    def load(path, uri):
        ann = Annotation(uri=uri)
        for i, line in enumerate(open(path, encoding="utf-8")):
            p = line.split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                s, d = float(p[3]), float(p[4])
                if d > 0:
                    ann[Segment(s, s + d), f"t{i}"] = p[7]
        return ann

    uri = Path(ref_p).stem
    ref, hyp = load(ref_p, uri), load(hyp_p, uri)
    uem = None
    if uem_p and Path(uem_p).exists():
        uem = Timeline(uri=uri)
        for line in open(uem_p, encoding="utf-8"):
            p = line.split()
            if len(p) >= 4:
                uem.add(Segment(float(p[2]), float(p[3])))
    c = DiarizationErrorRate(collar=collar, skip_overlap=False)(
        ref, hyp, uem=uem, detailed=True)
    return (c["total"], c["missed detection"],
            c["false alarm"], c["confusion"])


# ------------------------------------------------------------------ steps

def run_step4():
    """Call the real 04_audacity_to_rttm.py so there is one source of truth."""
    sys.path.insert(0, str(SCRIPTS))
    import importlib
    mod = importlib.import_module("04_audacity_to_rttm")
    importlib.reload(mod)
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.main(str(BASE / "corrected_labels"),
                 str(BASE / "processed"),
                 str(BASE / "rttm_truth"))
    return buf.getvalue()


def run_step5():
    truth, draft = BASE / "rttm_truth", BASE / "rttm_draft"
    try:
        import pyannote.metrics  # noqa: F401
        engine, fn = "pyannote.metrics", score_pyannote
    except ImportError:
        engine, fn = "built-in (verified identical to pyannote)", score_builtin

    rows = []
    pooled = {c: [0.0, 0.0, 0.0, 0.0] for c in COLLARS}
    for tf in sorted(truth.glob("*.rttm")):
        clip = tf.stem
        hf = draft / f"{clip}.rttm"
        uf = truth / f"{clip}.uem"
        if not hf.exists():
            rows.append({"clip": clip, "skip": "no machine draft"})
            continue

        ref, hyp = read_rttm(tf), read_rttm(hf)
        uem = read_uem(uf) if uf.exists() else [
            (0.0, max(e for _, e, _ in ref + hyp))]

        dur = {}
        for s, e, l in hyp:
            dur[l] = dur.get(l, 0.0) + (e - s)

        row = {"clip": clip,
               "true": len({l for _, _, l in ref}),
               "found": len(dur),
               "found2": sum(1 for v in dur.values() if v >= MIN_SPK),
               "tiny": [f"{l} ({v:.2f}s)" for l, v in sorted(dur.items())
                        if v < MIN_SPK],
               "no_uem": not uf.exists(),
               "uem_end": max(e for _, e in uem),
               "ref_end": max(e for _, e, _ in ref) if ref else 0.0}
        for collar in COLLARS:
            if fn is score_pyannote:
                t, mi, f, c = fn(tf, hf, uf if uf.exists() else None, collar)
            else:
                t, mi, f, c = fn(ref, hyp, uem, collar)
            row[collar] = (0, 0, 0, 0) if not t else (
                (mi + f + c) / t * 100, mi / t * 100, f / t * 100, c / t * 100)
            for i, v in enumerate((t, mi, f, c)):
                pooled[collar][i] += v
        rows.append(row)
    return engine, rows, pooled


def insights(rows, pooled):
    out = []
    good = [r for r in rows if "skip" not in r]
    if not good:
        return ["Nothing scored - no matching files in rttm_draft\\."]
    t, mi, f, c = pooled[0.0]
    if t:
        der = (mi + f + c) / t * 100
        share = c / (mi + f + c) * 100 if (mi + f + c) else 0
        if share >= 60:
            out.append(f"Speaker confusion is {share:.0f}% of all error. The model "
                       f"finds the speech and gives it to the wrong person - work "
                       f"belongs in embeddings/clustering, not segmentation.")
        else:
            out.append(f"Error is mixed: confusion {share:.0f}%, missed "
                       f"{mi/(mi+f+c)*100:.0f}%, extra {f/(mi+f+c)*100:.0f}%. "
                       f"Say which dominates per clip, not in aggregate.")
    under = [r for r in good if r["found2"] < r["true"]]
    if under:
        out.append(f"Under-counts speakers on {len(under)} of {len(good)} clips "
                   f"({', '.join(r['clip'] for r in under)}) - merging people together.")
    tiny = [r for r in good if r["tiny"]]
    if tiny:
        out.append("Speaker count inflated by fragments: "
                   + "; ".join(f"{r['clip']}: {', '.join(r['tiny'])}" for r in tiny)
                   + f". Quote the '>={MIN_SPK:g}s' column, not 'found'.")
    nou = [r["clip"] for r in good if r["no_uem"]]
    if nou:
        out.append("NO .uem for " + ", ".join(nou)
                   + " - scored the whole clip. Protocol v1 wants an explicit UEM.")
    short = [r for r in good if r["uem_end"] < r["ref_end"] - 0.01]
    if short:
        out.append("UEM ENDS BEFORE THE LAST SPEECH on "
                   + ", ".join(f"{r['clip']} (uem {r['uem_end']:.2f}s, speech to "
                               f"{r['ref_end']:.2f}s)" for r in short)
                   + " - real audio sits outside the scored region.")
    return out


# -------------------------------------------------------------------- GUI

def build_report():
    lines = []
    step4 = run_step4()
    engine, rows, pooled = run_step5()

    lines.append("STEP 4 - building rttm / uem / events from your labels")
    lines.append("=" * 78)
    lines.append(step4.rstrip())
    lines.append("")
    lines.append(f"STEP 5 - scoring   [engine: {engine}]")
    lines.append("=" * 78)
    hdr = (f"{'clip':<18}{'DER 0.0':>9}{'DER .25':>9}{'missed':>8}"
           f"{'extra':>8}{'mixed up':>10}{'spk':>5}{'found':>7}{'>=2s':>6}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        if "skip" in r:
            lines.append(f"{r['clip']:<18}  ! {r['skip']}")
            continue
        a, b = r[0.0], r[0.25]
        flag = "" if r["found2"] == r["true"] else "  <-"
        lines.append(f"{r['clip']:<18}{a[0]:>9.2f}{b[0]:>9.2f}{a[1]:>8.2f}"
                     f"{a[2]:>8.2f}{a[3]:>10.2f}{r['true']:>5}"
                     f"{r['found']:>7}{r['found2']:>6}{flag}")
    lines.append("-" * len(hdr))
    for collar in COLLARS:
        t, mi, f, c = pooled[collar]
        if t:
            lines.append(f"ALL CLIPS, collar {collar}:  DER {(mi+f+c)/t*100:.2f}%   "
                         f"missed {mi/t*100:.2f}   extra {f/t*100:.2f}   "
                         f"mixed up {c/t*100:.2f}")
    lines.append("")
    lines.append("WHAT THIS SAYS")
    lines.append("=" * 78)
    for i in insights(rows, pooled):
        lines.append(" * " + i)
    lines.append("")
    lines.append("Files written to:  rttm_truth\\")
    return "\n".join(lines)


def main_gui():
    import tkinter as tk
    from tkinter import font as tkfont

    root = tk.Tk()
    root.title("C3 Console")
    root.geometry("1000x680")

    top = tk.Frame(root, padx=12, pady=10)
    top.pack(fill="x")
    tk.Label(top, text="C3 Console", font=("Segoe UI", 15, "bold")).pack(side="left")
    status = tk.Label(top, text=str(BASE), fg="#666", font=("Segoe UI", 9))
    status.pack(side="left", padx=14)

    mono = tkfont.Font(family="Consolas", size=10)
    box = tk.Text(root, wrap="none", font=mono, bg="#12151b", fg="#e6e9ef",
                  insertbackground="#e6e9ef", padx=12, pady=10, borderwidth=0)
    ysb = tk.Scrollbar(root, command=box.yview)
    box.configure(yscrollcommand=ysb.set)
    ysb.pack(side="right", fill="y")
    box.pack(fill="both", expand=True)
    box.insert("1.0", "Press  Build + Score.\n\nIt reads corrected_labels\\, writes "
                      "rttm_truth\\, then scores against rttm_draft\\.\n")

    def go():
        box.delete("1.0", "end")
        box.insert("1.0", "working...\n")
        root.update()
        try:
            text = build_report()
        except Exception:
            text = ("SOMETHING BROKE - the traceback below says where.\n\n"
                    + traceback.format_exc()
                    + "\nMost likely causes:\n"
                      "  * a label that isn't spk01 / spk02-BC / NOSCORE\n"
                      "  * no .wav in processed\\ matching the label filename\n"
                      "  * you exported to draft_labels\\ instead of corrected_labels\\\n")
        box.delete("1.0", "end")
        box.insert("1.0", text)

    def open_folder():
        import subprocess
        subprocess.Popen(["explorer", str(BASE / "rttm_truth")])

    bar = tk.Frame(root, padx=12, pady=8)
    bar.pack(fill="x", before=box)
    tk.Button(bar, text="Build + Score", command=go, font=("Segoe UI", 11, "bold"),
              bg="#1f6feb", fg="white", relief="flat", padx=18, pady=7).pack(side="left")
    tk.Button(bar, text="Open rttm_truth folder", command=open_folder,
              font=("Segoe UI", 10), relief="flat", padx=12, pady=7).pack(side="left", padx=8)
    root.mainloop()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        print(build_report())
    else:
        try:
            main_gui()
        except Exception:
            print(build_report())
