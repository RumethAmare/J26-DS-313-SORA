"""
Kathā · C3 — "Who spoke when" demo (PP1). Runs fully OFFLINE (laptop CPU or a GPU PC).

F1  Who spoke when : record with the microphone or upload a file -> timeline, talk time, overlap, turns,
                     downloadable RTTM + JSON (the C3 output contract the other components join on).
F2  Benchmark      : any clip of the C3 Sinhala-English benchmark -> ground truth vs model, error rate split into
                     missed / false alarm / wrong person, an error strip over time, per-speaker breakdown,
                     and the whole-benchmark table.

Start:  run_app.bat   (opens http://127.0.0.1:7860 in the browser)
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never contact the internet; model comes from the local cache
import warnings; warnings.filterwarnings("ignore")
import json, glob, time, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import gradio as gr
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # ...\C3 Pilot
TRUTH, DRAFT, AUDIO = ROOT / "rttm_truth", ROOT / "rttm_draft", ROOT / "processed"
MODEL_ID = "pyannote/speaker-diarization-community-1"
SR = 16000
LIVE_TAU = 0.6          # speaker-matching threshold for live mode (chosen on TUNE clips in E020)
PALETTE = ["#2E86AB", "#E4572E", "#3BA55C", "#F3A712", "#8E6C8A", "#17BEBB", "#C94277", "#5C6B73"]
ERR_COL = {"correct": "#3BA55C", "missed": "#F3A712", "false alarm": "#8E6C8A", "wrong person": "#E4572E"}
SETS = {"TEST (IRD panels)": ["IRD_Harshana_Silva", "IRD_Recording_2_IIT"],
        "TEST2 (YouTube)": ["Yt01", "YT02", "YT03"],
        "TEST3 (YouTube)": ["YT04", "YT05", "YT07"]}

# ---------------------------------------------------------------- model (loaded once, lazily)
_PIPE = {}
def pipe():
    if "p" not in _PIPE:
        from pyannote.audio import Pipeline
        p = Pipeline.from_pretrained(MODEL_ID)
        if p is None:
            raise gr.Error("Model not in the local cache. Close the app and run run_app.bat again (it downloads it once).")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        p.to(torch.device(dev))
        _PIPE["p"] = p
        _PIPE["dev"] = (torch.cuda.get_device_name(0) + " GPU") if dev == "cuda" else "CPU"
    return _PIPE["p"]


# ---------------------------------------------------------------- io
def load_audio(path):
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        raise gr.Error("Could not read this audio file. Please use WAV, FLAC or OGG (or record with the microphone).")
    wav = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    if sr != SR:
        import torchaudio.functional as AF
        wav = AF.resample(wav, sr, SR)
    return wav, SR


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


def diarize_wave(wav, sr, num_speakers=None):
    t0 = time.time()
    kw = {"num_speakers": int(num_speakers)} if num_speakers else {}
    out = pipe()({"waveform": wav, "sample_rate": sr}, **kw)
    ann = getattr(out, "speaker_diarization", out)
    return ann, time.time() - t0


# ---------------------------------------------------------------- helpers
def order(ann):
    seen = []
    for s, _, lab in sorted(ann.itertracks(yield_label=True), key=lambda t: t[0].start):
        if lab not in seen:
            seen.append(lab)
    return seen


def friendly(ann, ref=None, uem=None):
    """Speaker A, B, ... With a reference, model speakers take the name of the true speaker they best match."""
    if ref is None:
        return ann.rename_labels(mapping={l: f"Speaker {chr(65 + i)}" for i, l in enumerate(order(ann))})
    rn = {l: f"Speaker {chr(65 + i)}" for i, l in enumerate(order(ref))}
    m = DiarizationErrorRate().optimal_mapping(ref, ann.crop(uem) if uem is not None else ann)
    names, k = {}, len(rn)
    for h, r in m.items():
        if r in rn:
            names[h] = rn[r]
    for l in order(ann):
        if l not in names:
            names[l] = f"Speaker {chr(65 + k)} (extra)"; k += 1
    return ann.rename_labels(mapping=names)


def overlap_tl(ann):
    tr = list(ann.itertracks(yield_label=True))
    ov = Timeline()
    for i in range(len(tr)):
        for j in range(i + 1, len(tr)):
            (a, _, la), (b, _, lb) = tr[i], tr[j]
            if la != lb and a.intersects(b):
                ov.add(a & b)
    return ov.support()


def color_of(lab, row):
    if lab.startswith("Speaker ") and len(lab) > 8:
        return PALETTE[(ord(lab[8]) - 65) % len(PALETTE)]
    return PALETTE[row % len(PALETTE)]


def timeline_fig(panels, duration, strip=None):
    """panels: [(title, annotation)], strip: optional (frames_category_array, step) error strip under the panels."""
    heights = [max(1, len(a.labels())) for _, a in panels] + ([0.6] if strip is not None else [])
    fig, axes = plt.subplots(len(heights), 1, figsize=(12, 1.0 + 0.5 * sum(heights) + 0.45 * len(heights)),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    for ax, (title, ann) in zip(axes[:, 0], panels):
        labs = sorted(ann.labels())
        for row, lab in enumerate(labs):
            ax.broken_barh([(s.start, s.duration) for s in ann.label_timeline(lab)], (row - 0.38, 0.76),
                           facecolors=color_of(lab, row), edgecolor="none")
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs); ax.set_ylim(-0.6, len(labs) - 0.4)
        ax.invert_yaxis(); ax.set_xlim(0, duration); ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if strip is not None:
        cats, step = strip
        ax = axes[-1, 0]
        for name, col in ERR_COL.items():
            idx = np.where(cats == name)[0]
            if len(idx):
                runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
                ax.broken_barh([(r[0] * step, len(r) * step) for r in runs], (0, 1), facecolors=col, edgecolor="none")
        ax.set_yticks([]); ax.set_xlim(0, duration); ax.set_title("Where the model is right / wrong", loc="left", fontsize=11)
        ax.legend(handles=[Patch(color=c, label=n) for n, c in ERR_COL.items()], ncol=4, loc="lower right",
                  bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel("time (seconds)")
    fig.tight_layout()
    return fig


def turns_df(ann):
    ov = overlap_tl(ann)
    rows = [(round(s.start, 2), round(s.end, 2), round(s.duration, 2), lab, bool(ov.crop(s).duration() > 0))
            for s, _, lab in ann.itertracks(yield_label=True)]
    return pd.DataFrame(rows, columns=["start (s)", "end (s)", "duration (s)", "speaker", "overlaps someone"]
                        ).sort_values("start (s)").reset_index(drop=True)


def export(ann, uri):
    """RTTM + JSON in the shared contract: {speaker_id, start, end, overlap_flag}."""
    d = Path(tempfile.mkdtemp())
    ann = ann.rename_labels(mapping={l: l.replace(" ", "_") for l in ann.labels()})   # RTTM forbids spaces
    rttm = d / f"{uri}.rttm"
    with open(rttm, "w") as fh:
        ann.write_rttm(fh)
    ov = overlap_tl(ann)
    segs = [dict(speaker_id=lab, start=round(s.start, 3), end=round(s.end, 3),
                 overlap_flag=bool(ov.crop(s).duration() > 0))
            for s, _, lab in sorted(ann.itertracks(yield_label=True), key=lambda t: t[0].start)]
    js = d / f"{uri}.c3.json"
    js.write_text(json.dumps({"uri": uri, "component": "C3 speaker diarization", "model": MODEL_ID,
                              "segments": segs}, indent=1), encoding="utf-8")
    return [str(rttm), str(js)]


def components(ref, hyp, uem, collar=0.0):
    c = DiarizationErrorRate(collar=collar, skip_overlap=False)(ref, hyp, uem=uem, detailed=True)
    t = c["total"] or 1.0
    return dict(DER=100 * (c["confusion"] + c["missed detection"] + c["false alarm"]) / t,
                miss=100 * c["missed detection"] / t, fa=100 * c["false alarm"] / t,
                conf=100 * c["confusion"] / t, speech=t)


def error_frames(ref, hyp_named, uem, duration, step=0.02):
    """Per-frame category, hyp already renamed onto reference names (optimal mapping)."""
    n = int(duration / step) + 1
    def mat(ann):
        labs = ann.labels(); M = np.zeros((len(labs), n), bool)
        for s, _, l in ann.itertracks(yield_label=True):
            M[labs.index(l), int(s.start / step):min(n, int(s.end / step))] = True
        return labs, M
    rl, R = mat(ref); hl, H = mat(hyp_named)
    valid = np.zeros(n, bool)
    for s in uem:
        valid[int(s.start / step):min(n, int(s.end / step))] = True
    nr, nh = R.sum(0), H.sum(0)
    correct = np.zeros(n, int)
    for i, l in enumerate(rl):
        if l in hl:
            correct += (R[i] & H[hl.index(l)])
    cats = np.full(n, "", dtype=object)
    cats[(nr > 0) & valid] = "correct"
    cats[(nh > nr) & valid] = "false alarm"
    cats[(nr > nh) & valid] = "missed"
    cats[(np.minimum(nr, nh) > correct) & valid] = "wrong person"
    return cats, step


def per_speaker(ref, hyp_named, uem):
    rows = []
    for lab in order(ref):
        tl = ref.label_timeline(lab).crop(uem)
        tot = tl.duration()
        if tot <= 0:
            continue
        hyp_c = hyp_named.crop(tl)
        same = hyp_c.label_timeline(lab).crop(tl).duration() if lab in hyp_c.labels() else 0.0
        heard = hyp_c.get_timeline().support().crop(tl).duration()
        others = {l: hyp_c.label_timeline(l).crop(tl).duration() for l in hyp_c.labels() if l != lab}
        top = max(others, key=others.get) if others else "-"
        rows.append([lab, round(tot, 1), f"{100 * same / tot:.0f} %", f"{100 * (tot - heard) / tot:.0f} %",
                     f"{top} ({100 * others[top] / tot:.0f} %)" if others else "-"])
    return pd.DataFrame(rows, columns=["true speaker", "speaks (s)", "correctly labelled",
                                       "missed", "most often mistaken for"])


# ---------------------------------------------------------------- F1
def f1_run(audio_path, n_people):
    if not audio_path:
        raise gr.Error("Record something with the microphone, or upload a file first.")
    wav, sr = load_audio(audio_path)
    dur = wav.shape[1] / sr
    if dur < 2:
        raise gr.Error("That recording is shorter than 2 seconds. Please record a little longer.")
    n = int(n_people or 0)
    ann, secs = diarize_wave(wav, sr, n if n > 0 else None)
    uri = Path(audio_path).stem
    shown = friendly(ann)
    chart = shown.chart(); total = sum(d for _, d in chart) or 1.0
    md = [f"### Found **{len(shown.labels())} speaker(s)** in {dur:.1f} s of audio",
          f"Processed on this computer's {_PIPE.get('dev', 'CPU')} in **{secs:.1f} s**, fully offline. "
          f"Overlapping speech: **{overlap_tl(shown).duration():.1f} s**."
          + (f"  _(Told the model: {n} people.)_" if n > 0 else "  _(Speaker count decided by the model.)_"),
          "**Talk time**  \n" + "  \n".join(f"- {lab}: {d:.0f} s ({100 * d / total:.0f} %)" for lab, d in chart)]
    fig = timeline_fig([("Who spoke when (model)", shown)], dur)
    return fig, "\n\n".join(md), turns_df(shown), export(shown, uri)


# ---------------------------------------------------------------- F2
def bench_clips():
    out = []
    for p in sorted(TRUTH.glob("*.rttm")):
        c = p.stem
        if (TRUTH / f"{c}.uem").exists() and (DRAFT / f"{c}.rttm").exists():
            out.append(c)
    real = [c for s in SETS.values() for c in s if c in out]
    return real + [c for c in out if c not in real]


def f2_run(clip, live):
    if not clip:
        raise gr.Error("Pick a benchmark clip first.")
    ref, uem = read_rttm(TRUTH / f"{clip}.rttm", clip), read_uem(TRUTH / f"{clip}.uem", clip)
    if live:
        wav, sr = load_audio(AUDIO / f"{clip}.wav")
        hyp, secs = diarize_wave(wav, sr)
        how = f"run live on this computer in {secs:.0f} s"
        dur = wav.shape[1] / sr
    else:
        hyp = read_rttm(DRAFT / f"{clip}.rttm", clip)
        how = "stored result (pre-computed, identical model)"
        dur = max(uem.extent().end, ref.get_timeline().extent().end)
    ref_c = ref.crop(uem)
    named = friendly(hyp, ref_c, uem)
    refn = friendly(ref_c)
    # rename ref and hyp consistently: friendly() names hyp after ref's first-appearance order
    c0, c25 = components(ref, hyp, uem, 0.0), components(ref, hyp, uem, 0.25)
    cats, step = error_frames(refn, named, uem, dur)
    md = [f"### {clip}  ·  {dur / 60:.1f} min  ·  {how}",
          f"Ground truth: **{len(ref_c.labels())} speakers** · model found **{len(hyp.labels())}**",
          f"**Diarization error rate: {c0['DER']:.2f} %**  =  missed speech {c0['miss']:.2f}  +  false alarm "
          f"{c0['fa']:.2f}  +  **wrong person {c0['conf']:.2f}**   ·   {c25['DER']:.2f} % with the standard "
          f"0.25 s boundary tolerance"]
    fig = timeline_fig([("Ground truth (human annotation)", refn), ("Model (pyannote community-1)", named)], dur,
                       strip=(cats, step))
    return fig, "\n\n".join(md), per_speaker(refn, named, uem)


def f2_table():
    rows, pooled = [], {}
    setname = {c: s for s, cs in SETS.items() for c in cs}
    for c in bench_clips():
        ref, uem = read_rttm(TRUTH / f"{c}.rttm", c), read_uem(TRUTH / f"{c}.uem", c)
        hyp = read_rttm(DRAFT / f"{c}.rttm", c)
        d = DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref, hyp, uem=uem, detailed=True)
        g = setname.get(c, "Scripted corpus (team recordings)")
        p = pooled.setdefault(g, [0, 0, 0, 0, 0, 0.0])
        p[0] += d["missed detection"]; p[1] += d["false alarm"]; p[2] += d["confusion"]; p[3] += d["total"]
        p[4] += 1; p[5] += uem.duration()
        t = d["total"] or 1
        rows.append([g, c, round(uem.duration() / 60, 1), len(ref.crop(uem).labels()), len(hyp.labels()),
                     round(100 * (d["missed detection"] + d["false alarm"] + d["confusion"]) / t, 2),
                     round(100 * d["missed detection"] / t, 2), round(100 * d["false alarm"] / t, 2),
                     round(100 * d["confusion"] / t, 2)])
    cols = ["set", "clip", "minutes", "true speakers", "found", "DER %", "missed %", "false alarm %", "wrong person %"]
    summ = [[g, p[4], round(p[5] / 60, 1), round(100 * (p[0] + p[1] + p[2]) / p[3], 2), round(100 * p[0] / p[3], 2),
             round(100 * p[1] / p[3], 2), round(100 * p[2] / p[3], 2)] for g, p in pooled.items()]
    return (pd.DataFrame(summ, columns=["set", "clips", "minutes", "DER %", "missed %", "false alarm %", "wrong person %"]),
            pd.DataFrame(rows, columns=cols))


# ---------------------------------------------------------------- F1 live (chunked)
def live_new():
    from live import LiveDiarizer
    return LiveDiarizer(pipe(), sr=SR, window=10.0, step=5.0, tau=LIVE_TAU)


def live_chunk(chunk, state):
    if chunk is None:
        return gr.update(), gr.update(), state
    sr, data = chunk
    data = np.asarray(data)
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        import torchaudio.functional as AF
        data = AF.resample(torch.from_numpy(data).unsqueeze(0), sr, SR).squeeze(0).numpy()
    if state is None:
        state = live_new()
    steps = state.feed(data)
    if not steps:
        return gr.update(), gr.update(), state
    return live_view(state), live_text(state), state


def live_view(d):
    ann = d.result()
    return timeline_fig([("Live: who is speaking (updates every 5 s)", ann if ann.labels() else Annotation())],
                        max(d.now, 5.0))


def live_text(d):
    ann = d.result(); chart = ann.chart(); total = sum(x for _, x in chart) or 1.0
    lag = (np.mean(d.cpu_seconds[-5:]) if d.cpu_seconds else 0.0)
    return (f"**{len(ann.labels())} speaker(s)** so far · {d.now:.0f} s heard · last steps took "
            f"{lag:.1f} s of CPU per 5 s of audio  \n" + "  \n".join(f"- {l}: {x:.0f} s ({100 * x / total:.0f} %)"
                                                                   for l, x in chart))


def live_stop(state):
    if state is None:
        return gr.update(), "Nothing recorded yet.", None, None
    state.flush()
    ann = state.result()
    return live_view(state), live_text(state), export(ann, "live_recording") if ann.labels() else None, state


def live_reset():
    return None, "", None, None


# ---------------------------------------------------------------- UI
INTRO = ("# Kathā · Who spoke when?\n"
         "Component 3 of the Sinhala-English conversation pipeline: **speaker diarization**. "
         "Model: pyannote *community-1*, running **on this computer, fully offline** (no audio leaves the machine).")
with gr.Blocks(title="Kathā · C3 who spoke when") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("1 · Who spoke when"):
        with gr.Row():
            with gr.Column(scale=1):
                a_in = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                label="Record a conversation, or upload a WAV file")
                n_in = gr.Number(value=0, precision=0, label="How many people? (0 = let the model decide)")
                b1 = gr.Button("Who spoke when?", variant="primary")
                gr.Markdown("_Tip: 20–60 seconds works best for a live demo. On a laptop CPU, processing takes "
                            "roughly as long as the audio; on a GPU it takes a few seconds._")
            with gr.Column(scale=2):
                p1 = gr.Plot(label="Timeline")
                m1 = gr.Markdown()
                t1 = gr.Dataframe(label="Speaker turns", wrap=True)
                f1 = gr.File(label="Download: RTTM + JSON (the format C1/C2/C4 join on)", file_count="multiple")
        b1.click(f1_run, [a_in, n_in], [p1, m1, t1, f1])
    with gr.Tab("1b · Live (experimental)"):
        gr.Markdown("**Experimental, not part of the PP1 claim.** Every 5 s the model looks at the last 10 s and "
                    "matches voices to people already heard. Measured on the 8 real panel clips (E020): error rate "
                    "~39 % live vs ~21 % for the full-recording mode in tab 1, mostly because 10-second windows miss "
                    "speech. Use tab 1 for the real result.")
        live_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Microphone (live)")
                reset = gr.Button("Reset")
            with gr.Column(scale=2):
                lp = gr.Plot(label="Live timeline")
                lm = gr.Markdown()
                lf = gr.File(label="When you stop: RTTM + JSON", file_count="multiple")
        mic.stream(live_chunk, [mic, live_state], [lp, lm, live_state], stream_every=1.0, time_limit=600)
        mic.stop_recording(live_stop, [live_state], [lp, lm, lf, live_state])
        reset.click(live_reset, None, [lp, lm, lf, live_state])
    with gr.Tab("2 · Benchmark & error analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                c_in = gr.Dropdown(bench_clips(), label="Clip from the C3 Sinhala-English benchmark")
                live_in = gr.Checkbox(value=False, label="Run the model live (slower) instead of the stored result")
                b2 = gr.Button("Compare with ground truth", variant="primary")
            with gr.Column(scale=2):
                p2 = gr.Plot(label="Ground truth vs model")
                m2 = gr.Markdown()
                t2 = gr.Dataframe(label="Per speaker", wrap=True)
        b2.click(f2_run, [c_in, live_in], [p2, m2, t2])
        gr.Markdown("### The whole benchmark (stored results, collar 0.0, overlap scored)")
        b3 = gr.Button("Load benchmark table")
        s3 = gr.Dataframe(label="Pooled by set", wrap=True)
        d3 = gr.Dataframe(label="Every clip", wrap=True)
        b3.click(f2_table, None, [s3, d3])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
