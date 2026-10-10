# E021 (X1b) = finish X1: other speaker-embedding models through a custom SpeechBrain adapter.
# Same data, same leave-one-set-out CV, same AHC grid as E019 X1. Pre-registered in C3_LOGBOOK.md (10 Oct ~23:15 SL).
# Results append to e019/e019_X1b_grid.csv; the X1 rows (c1emb/wespeaker) are re-read from e019_X1_grid.csv for the table.
import os, csv, json, time, subprocess, itertools, traceback
import numpy as np
import torch

MODE = 'X1b'
ROOT = '/content'
WORK = '/content/drive/MyDrive/C3_E009'
OUT = f'{WORK}/e019'
os.makedirs(OUT, exist_ok=True)
C1 = 'pyannote/speaker-diarization-community-1'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- data ----------
for b in ('/content/drive/MyDrive/E009_upload.tgz', f'{WORK}/E015_upload.tgz', f'{WORK}/E016_upload.tgz'):
    marker = f'{ROOT}/.unpacked_' + os.path.basename(b)
    if not os.path.exists(marker):
        subprocess.run(['tar', 'xzf', b, '-C', ROOT], check=True)
        open(marker, 'w').close()
SETS = {'TEST': ['IRD_Harshana_Silva', 'IRD_Recording_2_IIT'],
        'TEST2': ['Yt01', 'YT02', 'YT03'],
        'TEST3': ['YT04', 'YT05', 'YT07']}
ALL = [c for s in SETS.values() for c in s]
SET_OF = {c: s for s, cs in SETS.items() for c in cs}
missing = [f for c in ALL for f in (f'{ROOT}/processed/{c}.wav', f'{ROOT}/rttm_truth/{c}.rttm', f'{ROOT}/rttm_truth/{c}.uem')
           if not os.path.exists(f)]
assert not missing, f'missing files: {missing}'

from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization as SDPipeline
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
try:
    from google.colab import userdata
    HF = userdata.get('HF_TOKEN')
except Exception:
    HF = os.environ.get('HF_TOKEN')


def load_rttm(path, uri):
    ann = Annotation(uri=uri)
    for i, line in enumerate(open(path, encoding='utf-8')):
        p = line.split()
        if len(p) >= 8 and p[0] == 'SPEAKER' and float(p[4]) > 0:
            ann[Segment(float(p[3]), float(p[3]) + float(p[4])), f't{i}'] = p[7]
    return ann


def load_uem(path, uri):
    tl = Timeline(uri=uri)
    for line in open(path, encoding='utf-8'):
        p = line.split()
        if len(p) >= 4:
            tl.add(Segment(float(p[2]), float(p[3])))
    return tl


REF = {c: load_rttm(f'{ROOT}/rttm_truth/{c}.rttm', c) for c in ALL}
UEM = {c: load_uem(f'{ROOT}/rttm_truth/{c}.uem', c) for c in ALL}


def components(c, ann):
    d = DiarizationErrorRate(collar=0.0, skip_overlap=False)(REF[c], ann, uem=UEM[c], detailed=True)
    return d['missed detection'], d['false alarm'], d['confusion'], d['total']


RES = f'{OUT}/e019_{MODE}_grid.csv'
FIELDS = ['system', 'params', 'clip', 'set', 'miss', 'fa', 'conf', 'total', 'n_spk']
DONE = set()
if os.path.exists(RES):
    DONE = {(r['system'], r['params'], r['clip']) for r in csv.DictReader(open(RES))}


def record(system, params, c, ann):
    key = json.dumps(params, sort_keys=True)
    m, f, k, t = components(c, ann)
    new = not os.path.exists(RES)
    with open(RES, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(dict(system=system, params=key, clip=c, set=SET_OF[c], miss=m, fa=f, conf=k, total=t,
                        n_spk=len(ann.labels())))
    DONE.add((system, key, c))


def as_ann(out):
    return out.speaker_diarization if hasattr(out, 'speaker_diarization') else out


def sweep(system, pipe, grid):
    """pipe.training=True caches segmentation+embeddings per file dict, so each setting re-runs clustering only."""
    pipe.training = True
    files = {c: {'uri': c, 'audio': f'{ROOT}/processed/{c}.wav'} for c in ALL}
    t0 = time.time()
    for i, params in enumerate(grid):
        key = json.dumps(params, sort_keys=True)
        todo = [c for c in ALL if (system, key, c) not in DONE]
        if not todo:
            continue
        pipe.instantiate(params)
        for c in todo:
            record(system, params, c, as_ann(pipe.apply(files[c])))
        if i % 10 == 0:
            print(f'  {system}: setting {i + 1}/{len(grid)}  [{time.time() - t0:.0f}s]', flush=True)
    print(f'{system}: sweep done in {time.time() - t0:.0f}s', flush=True)


def analyse():
    rows = list(csv.DictReader(open(RES)))
    tab = {}
    for r in rows:
        tab.setdefault((r['system'], r['params']), {})[r['clip']] = tuple(float(r[k]) for k in ('miss', 'fa', 'conf', 'total'))

    def der(comp, clips):
        m = sum(comp[c][0] for c in clips); f = sum(comp[c][1] for c in clips)
        k = sum(comp[c][2] for c in clips); t = sum(comp[c][3] for c in clips)
        return 100 * (m + f + k) / t, 100 * m / t, 100 * f / t, 100 * k / t

    systems = sorted({s for s, _ in tab})
    lines = [f'E019 {MODE} — leave-one-set-out CV, collar 0.0, blind. 8 real panel clips.']
    for s in systems:
        cands = {p: comp for (ss, p), comp in tab.items() if ss == s and all(c in comp for c in ALL)}
        if not cands:
            continue
        cv = {}
        chosen = {}
        for held, hclips in SETS.items():
            train = [c for c in ALL if c not in hclips]
            best = min(cands, key=lambda p: (der(cands[p], train)[0], p))
            chosen[held] = best
            for c in hclips:
                cv[c] = cands[best][c]
        d, m, f, k = der(cv, ALL)
        per_set = {st: round(der(cv, cs)[0], 2) for st, cs in SETS.items()}
        oracle_p = min(cands, key=lambda p: der(cands[p], ALL)[0])
        lines.append(f'\n[{s}] CV pooled DER {d:.2f}  (miss {m:.2f} fa {f:.2f} conf {k:.2f})  per set {per_set}')
        lines.append(f'   chosen per held-out set: {chosen}')
        lines.append(f'   in-sample ORACLE (upper bound, not a result): {der(cands[oracle_p], ALL)[0]:.2f} at {oracle_p}')
        per_clip = {c: round(100 * sum(cv[c][:3]) / cv[c][3], 2) for c in ALL}
        lines.append(f'   CV per clip: {per_clip}')
    txt = '\n'.join(lines)
    open(f'{OUT}/e019_{MODE}_summary.txt', 'w').write(txt + '\n')
    print(txt)



# ---------- custom SpeechBrain adapter (pyannote 4.0.7's built-in one is incompatible with speechbrain 1.x) ----------
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pyannote.audio import Audio


class SBEmbedding:
    """Same maths as pyannote's SpeechBrainPretrainedSpeakerEmbedding.__call__, but built with arguments speechbrain 1.x accepts."""
    def __init__(self, source, device):
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = device
        self.m = EncoderClassifier.from_hparams(source=source, savedir=f'/content/sb/{source.replace("/", "_")}',
                                                run_opts={'device': str(device)})
        self.m.eval()
        self.sample_rate = 16000
        self.metric = 'cosine'
        with torch.inference_mode():
            self.dimension = self.m.encode_batch(torch.rand(1, 16000).to(device)).shape[-1]
            lo, hi = 2, 8000
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                try:
                    self.m.encode_batch(torch.randn(1, mid).to(device)); hi = mid
                except RuntimeError:
                    lo = mid
        self.min_num_samples = hi

    def to(self, device):
        return self

    def __call__(self, waveforms, masks=None):
        b, c, n = waveforms.shape
        waveforms = waveforms.squeeze(1)
        if masks is None:
            signals, lens = waveforms, n * torch.ones(b)
        else:
            im = F.interpolate(masks.unsqueeze(1), size=n, mode='nearest').squeeze(1) > 0.5
            signals = pad_sequence([w[m].contiguous() for w, m in zip(waveforms, im)], batch_first=True)
            lens = im.sum(1).float()
        mx = lens.max()
        if mx < self.min_num_samples:
            return np.nan * np.zeros((b, self.dimension))
        short = lens < self.min_num_samples
        rel = lens / mx; rel[short] = 1.0
        with torch.inference_mode():
            e = self.m.encode_batch(signals.to(self.device), wav_lens=rel.to(self.device)).squeeze(1).cpu().numpy()
        e[short.cpu().numpy()] = np.nan
        return e


def build(emb_obj):
    p = SDPipeline(segmentation={'checkpoint': C1, 'subfolder': 'segmentation'},
                   embedding={'checkpoint': C1, 'subfolder': 'embedding'},
                   clustering='AgglomerativeClustering', embedding_batch_size=32, token=HF)
    p.to(DEVICE)
    if emb_obj is not None:
        p._embedding = emb_obj
        p._audio = Audio(sample_rate=emb_obj.sample_rate, mono='downmix')
    return p


def main():
    grid = [{'segmentation': {'min_duration_off': 0.0},
             'clustering': {'method': 'centroid', 'min_cluster_size': 12, 'threshold': round(t, 2)}}
            for t in np.arange(0.20, 1.501, 0.05)]
    arms = [('X1b_ecapa_ahc', 'speechbrain/spkrec-ecapa-voxceleb'),
            ('X1b_sb_resnet_ahc', 'speechbrain/spkrec-resnet-voxceleb'),
            ('X1b_sb_xvector_ahc', 'speechbrain/spkrec-xvect-voxceleb')]
    for name, src in arms:
        try:
            emb = SBEmbedding(src, DEVICE)
            print(f'{name}: dim {emb.dimension}, min samples {emb.min_num_samples}', flush=True)
            sweep(name, build(emb), grid)
            torch.cuda.empty_cache()
        except Exception as e:
            msg = f'{name}: FAILED {type(e).__name__}: {e}'
            print(msg); traceback.print_exc()
            open(f'{OUT}/e019_X1b_failures.txt', 'a').write(msg + '\n')
    # pyannote/embedding through pyannote's own loader (needs the HF terms accepted)
    try:
        p = SDPipeline(segmentation={'checkpoint': C1, 'subfolder': 'segmentation'}, embedding='pyannote/embedding',
                       clustering='AgglomerativeClustering', embedding_batch_size=32, token=HF)
        p.to(DEVICE)
        sweep('X1b_pyannote_emb_ahc', p, grid)
    except Exception as e:
        msg = f'X1b_pyannote_emb_ahc: FAILED {type(e).__name__}: {str(e)[:300]}'
        print(msg); open(f'{OUT}/e019_X1b_failures.txt', 'a').write(msg + '\n')
    # bring the E019 X1 rows into the same table for one comparison
    if os.path.exists(f'{OUT}/e019_X1_grid.csv'):
        have = {(r['system'], r['params'], r['clip']) for r in csv.DictReader(open(RES))} if os.path.exists(RES) else set()
        with open(RES, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            for r in csv.DictReader(open(f'{OUT}/e019_X1_grid.csv')):
                if r['system'] == 'X1_c1emb_ahc' and (r['system'], r['params'], r['clip']) not in have:
                    w.writerow({k: r[k] for k in FIELDS})
    analyse()


main()
