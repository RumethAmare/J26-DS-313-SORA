# Score pyannoteAI precision-3 output on the two public TEST clips with protocol v1
# (explicit UEM, overlap included, collars 0.0 / 0.25) - identical scoring to every C3 experiment.
# Usage: python score_p3.py <truth_dir with .rttm/.uem> <hypothesis_dir with <clip>.precision3.rttm>
# precision-3 was run on the pyannoteAI playground (speakers = auto, overlap-aware) as an external
# reference only; the C3 component itself runs offline with community-1.
import sys
from pathlib import Path
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

TRUTH, HYP = Path(sys.argv[1]), Path(sys.argv[2])
CLIPS = ['IRD_Harshana_Silva', 'IRD_Recording_2_IIT']


def load_rttm(path, uri):
    a = Annotation(uri=uri)
    for i, l in enumerate(open(path, encoding='utf-8')):
        p = l.split()
        if len(p) >= 8 and p[0] == 'SPEAKER' and float(p[4]) > 0:
            a[Segment(float(p[3]), float(p[3]) + float(p[4])), f't{i}'] = p[7]
    return a


def load_uem(path, uri):
    t = Timeline(uri=uri)
    for l in open(path, encoding='utf-8'):
        p = l.split()
        if len(p) >= 4:
            t.add(Segment(float(p[2]), float(p[3])))
    return t


for collar in (0.0, 0.25):
    pooled = dict.fromkeys(('confusion', 'missed detection', 'false alarm', 'total'), 0.0)
    per = {}
    for c in CLIPS:
        ref, uem = load_rttm(TRUTH / f'{c}.rttm', c), load_uem(TRUTH / f'{c}.uem', c)
        hyp = load_rttm(HYP / f'{c}.precision3.rttm', c)
        comp = DiarizationErrorRate(collar=collar, skip_overlap=False)(ref, hyp, uem=uem, detailed=True)
        per[c] = (100 * sum(comp[k] for k in ('confusion', 'missed detection', 'false alarm')) / comp['total'],
                  len(ref.crop(uem).labels()), len(hyp.crop(uem).labels()))
        for k in pooled:
            pooled[k] += comp[k]
    t = pooled['total']
    der = 100 * sum(pooled[k] for k in ('confusion', 'missed detection', 'false alarm')) / t
    print(f'collar {collar}: pooled DER {der:.2f} (miss {100*pooled["missed detection"]/t:.2f} fa {100*pooled["false alarm"]/t:.2f} '
          f'conf {100*pooled["confusion"]/t:.2f})  per clip: ' + ', '.join(f'{c} {d:.2f} (spk {r}->{h})' for c, (d, r, h) in per.items()))
