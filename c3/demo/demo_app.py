# C3 demo — "who spoke when" for Sinhala-English conversations (community-1, pyannote.audio 4.x)
# Needs, from the setup cell: PIPE (loaded community-1 pipeline on GPU), ROOT (folder with processed/ + rttm_truth/)
import os, time, tempfile, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import gradio as gr
from pyannote.core import Annotation, Segment, Timeline
from pyannote.audio import Audio
from pyannote.metrics.diarization import DiarizationErrorRate

PALETTE = ['#2E86AB', '#E4572E', '#3BA55C', '#F3A712', '#8E6C8A', '#17BEBB', '#C94277', '#5C6B73']
MODES = ['Automatic (model decides how many speakers)',
         'I know how many people are in the meeting',
         'Compare both side by side']


# ---------- helpers ----------
def read_rttm(path, uri):
    ann = Annotation(uri=uri)
    for i, line in enumerate(open(path, encoding='utf-8')):
        p = line.split()
        if len(p) >= 8 and p[0] == 'SPEAKER' and float(p[4]) > 0:
            ann[Segment(float(p[3]), float(p[3]) + float(p[4])), f't{i}'] = p[7]
    return ann

def read_uem(path, uri):
    tl = Timeline(uri=uri)
    for line in open(path, encoding='utf-8'):
        p = line.split()
        if len(p) >= 4:
            tl.add(Segment(float(p[2]), float(p[3])))
    return tl

def friendly(ann, ref=None, uem=None):
    """Rename labels to Speaker A, B, ... in order of first appearance.
    With a reference, model speakers take the name of the true speaker they best match (optimal mapping),
    so colours line up between the ground-truth and model rows."""
    def order(a):
        seen = []
        for _, _, lab in sorted(a.itertracks(yield_label=True), key=lambda t: t[0].start):
            if lab not in seen:
                seen.append(lab)
        return seen
    if ref is None:
        return ann.rename_labels(mapping={lab: f'Speaker {chr(65 + i)}' for i, lab in enumerate(order(ann))})
    ref_names = {lab: f'Speaker {chr(65 + i)}' for i, lab in enumerate(order(ref))}
    m = DiarizationErrorRate().optimal_mapping(ref, ann.crop(uem) if uem is not None else ann)
    names, used = {}, set()
    for h, r in m.items():
        if r in ref_names:
            names[h] = ref_names[r]
            used.add(ref_names[r])
    k = len(ref_names)
    for lab in order(ann):
        if lab not in names:
            names[lab] = f'Speaker {chr(65 + k)} (extra)'
            k += 1
    return ann.rename_labels(mapping=names)

def turns_table(ann):
    rows = [(round(s.start, 2), round(s.end, 2), round(s.duration, 2), lab)
            for s, _, lab in ann.itertracks(yield_label=True)]
    df = pd.DataFrame(rows, columns=['start (s)', 'end (s)', 'duration (s)', 'speaker'])
    return df.sort_values('start (s)').reset_index(drop=True)

def overlap_seconds(ann):
    tracks = list(ann.itertracks(yield_label=True))
    ov = Timeline()
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            (a, _, la), (b, _, lb) = tracks[i], tracks[j]
            if la != lb and a.intersects(b):
                ov.add(a & b)
    return ov.support().duration()

def talk_time(ann):
    chart = ann.chart()
    total = sum(d for _, d in chart) or 1.0
    return [(lab, d, 100 * d / total) for lab, d in chart]

def timeline_figure(panels, duration):
    """panels: list of (title, annotation). One row per speaker, one panel per annotation."""
    heights = [max(1, len(a.labels())) for _, a in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 0.9 + 0.55 * sum(heights) + 0.5 * len(panels)),
                             gridspec_kw={'height_ratios': heights}, squeeze=False)
    for ax, (title, ann) in zip(axes[:, 0], panels):
        labs = sorted(ann.labels())
        for row, lab in enumerate(labs):
            color = PALETTE[(ord(lab.split()[1][0]) - 65) % len(PALETTE)] if lab.startswith('Speaker ') else PALETTE[row % len(PALETTE)]
            spans = [(s.start, s.duration) for s in ann.label_timeline(lab)]
            ax.broken_barh(spans, (row - 0.38, 0.76), facecolors=color, edgecolor='none')
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs)
        ax.set_ylim(-0.6, len(labs) - 0.4)
        ax.invert_yaxis()
        ax.set_xlim(0, duration)
        ax.set_title(title, loc='left', fontsize=11)
        ax.grid(axis='x', alpha=0.25)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel('time (seconds)')
    fig.tight_layout()
    return fig

def score_clip(ref, hyp, uem):
    out = {}
    for col in (0.0, 0.25):
        c = DiarizationErrorRate(collar=col, skip_overlap=False)(ref, hyp, uem=uem, detailed=True)
        t = c['total']
        out[col] = dict(DER=100 * (c['confusion'] + c['missed detection'] + c['false alarm']) / t,
                        miss=100 * c['missed detection'] / t, fa=100 * c['false alarm'] / t,
                        conf=100 * c['confusion'] / t)
    return out

def run_model(path, n):
    t0 = time.time()
    out = PIPE(path, num_speakers=n) if n else PIPE(path)
    ann = out.speaker_diarization if hasattr(out, 'speaker_diarization') else out
    return ann, time.time() - t0

def benchmark_clips():
    clips = sorted(os.path.basename(p)[:-5] for p in glob.glob(f'{ROOT}/rttm_truth/*.rttm'))
    real = [c for c in clips if c.startswith('IRD')]            # public YouTube panels first
    return real + [c for c in clips if c not in real]


# ---------- the one handler behind both tabs ----------
def diarize(audio_path, clip, truth_file, mode, n_people):
    if not audio_path and clip:                       # an uploaded file always wins over the dropdown
        audio_path = f'{ROOT}/processed/{clip}.wav'
        truth_path, uem_path, uri = f'{ROOT}/rttm_truth/{clip}.rttm', f'{ROOT}/rttm_truth/{clip}.uem', clip
    else:
        if not audio_path:
            raise gr.Error('Pick a benchmark clip or upload an audio file first.')
        uri = os.path.splitext(os.path.basename(audio_path))[0]
        truth_path = truth_file if truth_file else None
        uem_path = None
    duration = Audio().get_duration(audio_path)
    ref = read_rttm(truth_path, uri) if truth_path else None
    uem = read_uem(uem_path, uri) if uem_path else (Timeline([Segment(0, duration)], uri=uri) if ref else None)
    n = int(n_people or 0)
    if mode != MODES[0] and n < 1:
        raise gr.Error('Enter how many people are in the meeting (1 or more).')

    runs = []
    if mode in (MODES[0], MODES[2]):
        runs.append(('Automatic', *run_model(audio_path, None)))
    if mode in (MODES[1], MODES[2]):
        runs.append((f'Told: {n} people', *run_model(audio_path, n)))

    panels, md = [], [f'### {uri}  ·  {duration / 60:.1f} min of audio']
    ref_c = ref.crop(uem) if ref is not None else None
    if ref is not None:
        panels.append(('Ground truth (human annotation)', friendly(ref_c)))
    for name, ann, secs in runs:
        shown = friendly(ann, ref_c, uem)
        panels.append((f'Model: {name}', shown))
        md.append(f'**{name}:** found **{len(shown.labels())} speakers** · processed in {secs:.1f} s '
                  f'({duration / max(secs, 1e-6):.0f}× faster than real time) · '
                  f'overlapping speech {overlap_seconds(shown):.1f} s')
        md.append('  \n'.join(f'- {lab}: {sec:.0f} s ({pct:.0f} %)' for lab, sec, pct in talk_time(shown)))
        if ref is not None:
            s = score_clip(ref, ann, uem)
            md.append(f'Diarization error rate vs ground truth: **{s[0.0]["DER"]:.2f} %** '
                      f'(missed {s[0.0]["miss"]:.2f} · false alarm {s[0.0]["fa"]:.2f} · speaker mix-up {s[0.0]["conf"]:.2f}) '
                      f'· {s[0.25]["DER"]:.2f} % with the usual 0.25 s boundary forgiveness')
    if ref is not None:
        md.append(f'_Ground truth has {len(ref.crop(uem).labels())} speakers._')

    fig = timeline_figure(panels, duration)
    last_name, last_ann, _ = runs[-1]
    table = turns_table(friendly(last_ann, ref_c, uem))
    rttm_path = os.path.join(tempfile.mkdtemp(), f'{uri}.rttm')
    with open(rttm_path, 'w') as fh:
        last_ann.write_rttm(fh)
    return fig, '\n\n'.join(md), table, rttm_path


# ---------- UI ----------
with gr.Blocks(title='Kathā · C3 speaker diarization') as demo:
    gr.Markdown('# Who spoke when?\nSinhala-English conversation diarization — Component 3 of Kathā. '
                'Model: pyannote community-1, running on this machine\'s GPU.')
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab('Benchmark clip'):
                    clip_in = gr.Dropdown(benchmark_clips(), label='Clip from the C3 benchmark (has ground truth)')
                with gr.Tab('Your own audio'):
                    audio_in = gr.Audio(type='filepath', label='Meeting recording (wav / mp3 / m4a)')
                    truth_in = gr.File(label='Optional: ground-truth RTTM (to get an error rate)', type='filepath')
            mode_in = gr.Radio(MODES, value=MODES[0], label='Speaker count')
            n_in = gr.Number(value=0, precision=0, label='How many people are in the meeting?')
            go = gr.Button('Diarize', variant='primary')
        with gr.Column(scale=2):
            plot_out = gr.Plot(label='Timeline')
            md_out = gr.Markdown()
            table_out = gr.Dataframe(label='Speaker turns', wrap=True)
            rttm_out = gr.File(label='Download RTTM')

    go.click(diarize, inputs=[audio_in, clip_in, truth_in, mode_in, n_in],
             outputs=[plot_out, md_out, table_out, rttm_out])
