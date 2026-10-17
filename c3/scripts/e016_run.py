# E016 - replicate E015 on TEST3 (YT04, YT05, YT07; 13.4 min real). No training.
# Run in notebook 09 after Cells 1-4:  exec(open(f'{WORK}/e016_run.py').read(), globals())
# Needs in WORK: E016_upload.tgz (TEST3 wav+truth), E015_upload.tgz (TEST2 wav+truth), rttm_e015/ (TEST2 outputs).
# Pre-registered in C3_LOGBOOK.md (10 Oct 2026) before any scoring:
#   R1 every B big seed < community-1 on TEST3;  R2 every D synth_real seed < every C synth_real seed on TEST3;
#   R3 every B big seed < community-1 pooled over TEST2+TEST3 (6 clips).
import os, csv, json, glob, time, tarfile, torch
from pyannote.audio import Pipeline, Inference, Model
from pyannote.audio.pipelines import SpeakerDiarization as SDPipeline

TEST3 = ['YT04', 'YT05', 'YT07']
TEST2 = ['Yt01', 'YT02', 'YT03']
C1 = 'pyannote/speaker-diarization-community-1'
EMB = 'pyannote/wespeaker-voxceleb-resnet34-LM'
RES16 = f'{WORK}/e016_results.csv'
F16 = ['part', 'arm', 'seed', 'set', 'collar', 'DER', 'miss', 'fa', 'conf', 'per_clip']


def _e016():
    for up in ('E016_upload.tgz', 'E015_upload.tgz'):
        with tarfile.open(f'{WORK}/{up}') as t:
            t.extractall(ROOT)
    for c in TEST3 + TEST2:
        for f in (f'rttm_truth/{c}.rttm', f'rttm_truth/{c}.uem'):
            assert os.path.exists(f'{ROOT}/{f}'), f'missing {f}'
    for c in TEST3:
        assert os.path.exists(f'{ROOT}/processed/{c}.wav'), c
    assert score.__code__.co_varnames[:3] == ('hyps', 'clips', 'collar'), "re-run Cell 4"

    def fresh_c1():
        p = Pipeline.from_pretrained(C1, token=HF); p.to(DEVICE); return p

    def swap_segmentation(p, model):
        old = p._segmentation
        assert model.specifications.powerset == old.model.specifications.powerset
        d = model.specifications.duration
        p._segmentation = Inference(model, duration=d, step=p.segmentation_step * d,
                                    skip_aggregation=True, batch_size=old.batch_size)
        p.to(DEVICE); return p

    def build_A(seg):
        p = SDPipeline(segmentation=seg, embedding=EMB, token=HF)
        p.instantiate(p.default_parameters()); p.to(DEVICE); return p

    done = set()
    if os.path.exists(RES16):
        done = {(r['part'], r['arm'], r['seed']) for r in csv.DictReader(open(RES16))}

    def write(part, arm, seed, hyps):
        new = not os.path.exists(RES16)
        with open(RES16, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=F16)
            if new: w.writeheader()
            for setname, clips in (('TEST3', TEST3), ('TEST2+3', TEST2 + TEST3)):
                for col in (0.0, 0.25):
                    der, miss, fa, conf, per = score({c: hyps[c] for c in clips}, clips, col)
                    w.writerow(dict(part=part, arm=arm, seed=seed, set=setname, collar=col, DER=f'{der:.4f}',
                                    miss=f'{miss:.4f}', fa=f'{fa:.4f}', conf=f'{conf:.4f}',
                                    per_clip=json.dumps({k: round(v, 4) for k, v in per.items()})))

    def run(part, arm, seed, make, **kw):
        if (part, arm, str(seed)) in done:
            print(f'{part} {arm} s{seed}: already scored, skipped'); return
        prev = f'{WORK}/rttm_e015/{part}_{arm}_s{seed}'
        if not all(os.path.exists(f'{prev}/{c}.rttm') for c in TEST2):
            print(f'{part} {arm} s{seed}: no E015 TEST2 outputs - skipped'); return
        t0 = time.time(); pipe = make()
        out = f'{WORK}/rttm_e016/{part}_{arm}_s{seed}'; os.makedirs(out, exist_ok=True)
        hyps = {c: load_rttm(f'{prev}/{c}.rttm', c) for c in TEST2}
        for c in TEST3:
            o = pipe(f'{ROOT}/processed/{c}.wav', **kw)
            ann = o.speaker_diarization if hasattr(o, 'speaker_diarization') else o
            hyps[c] = ann
            with open(f'{out}/{c}.rttm', 'w') as fh: ann.write_rttm(fh)
        write(part, arm, seed, hyps)
        print(f'{part} {arm:>14} s{seed}: TEST3 {score({c: hyps[c] for c in TEST3}, TEST3, 0.0)[0]:6.2f}  '
              f'TEST2+3 {score(hyps, TEST2 + TEST3, 0.0)[0]:6.2f}  ({time.time() - t0:.0f}s)')
        del pipe; torch.cuda.empty_cache()

    run('B', 'c1_zeroshot', '-', fresh_c1)
    run('A', 'pretrained', '-', lambda: build_A('pyannote/segmentation-3.0'))
    pj = f'{WORK}/e013/e013_params.json'
    if os.path.exists(pj):
        prm = json.load(open(pj))
        def tuned():
            p = fresh_c1(); p.instantiate(prm); return p
        run('F', 'c1_tuned', '-', tuned)
    cks = sorted(c for c in glob.glob(f'{WORK}/ckpt/*_s*/best.ckpt') if 'unfinished' not in c)
    print(f'\n{len(cks)} finished checkpoints')
    for ck in cks:
        tag = os.path.basename(os.path.dirname(ck))
        part, rest = tag.split('_', 1); arm, seed = rest.rsplit('_s', 1)
        if part == 'A':
            run(part, arm, seed, lambda ck=ck: build_A(Model.from_pretrained(ck)))
        else:
            run(part, arm, seed, lambda ck=ck: swap_segmentation(fresh_c1(), Model.from_pretrained(ck)))

    rows = [r for r in csv.DictReader(open(RES16)) if float(r['collar']) == 0.0]
    L = ['===== E016 - TEST3 (YT04, YT05, YT07; 13.4 min) and TEST2+3 pooled (27.2 min), collar 0.0 =====']
    V = {}
    for st in ('TEST3', 'TEST2+3'):
        by = {}
        for r in rows:
            if r['set'] == st: by.setdefault((r['part'], r['arm']), []).append(r)
        L += ['', f'--- {st} ---', f'{"part arm":<22}{"n":>3}{"mean":>8}{"min":>8}{"max":>8}{"fa":>7}{"conf":>7}  per clip (mean)']
        for (p, a), rs in sorted(by.items()):
            d = [float(r['DER']) for r in rs]; V[(st, p, a)] = d
            mu = lambda k: sum(float(r[k]) for r in rs) / len(rs)
            pcs = [json.loads(r['per_clip']) for r in rs]
            pc = {c: round(sum(x[c] for x in pcs) / len(pcs), 1) for c in pcs[0]}
            L.append(f'{p + " " + a:<22}{len(d):>3}{mu("DER"):8.2f}{min(d):8.2f}{max(d):8.2f}{mu("fa"):7.2f}{mu("conf"):7.2f}  {pc}')
    def beats(st):
        c1 = V[(st, 'B', 'c1_zeroshot')][0]; bb = V.get((st, 'B', 'big'), [])
        return bb and max(bb) < c1, c1, bb
    ok, c1, bb = beats('TEST3')
    L += ['', f'R1 (TEST3): ' + ('REPLICATES' if ok else 'DOES NOT REPLICATE') + f' - B big {bb} vs community-1 {c1:.2f}']
    d, c = V.get(('TEST3', 'D', 'synth_real')), V.get(('TEST3', 'C', 'synth_real'))
    if d and c:
        L.append('R2 (TEST3): ' + ('REPLICATES' if max(d) < min(c) else 'REVERSED' if min(d) > max(c) else 'NO DETECTABLE DIFFERENCE'))
    ok, c1, bb = beats('TEST2+3')
    L.append(f'R3 (TEST2+3 pooled): ' + ('HOLDS' if ok else 'DOES NOT HOLD') + f' - B big {bb} vs community-1 {c1:.2f}')
    text = '\n'.join(L)
    open(f'{WORK}/e016_summary.txt', 'w').write(text + '\n')
    subprocess.run(f'cd "{WORK}" && tar czf e016_bundle.tgz e016_results.csv e016_summary.txt rttm_e016', shell=True)
    print('\n' + text)


import subprocess
_e016()
