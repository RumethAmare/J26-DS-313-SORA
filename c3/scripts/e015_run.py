# E015 - re-score EVERYTHING already built on the NEW real clips (TEST2 = Yt01, YT02, YT03). No training.
#
# HOW TO RUN (Colab, T4, same as E009-E014):
#   1. Put E015_upload.tgz in  My Drive/C3_E009/   (next to e009_results.csv).
#   2. Open 09_more_data_finetune.ipynb, run Cells 1-4 only (GPU check, install+restart, config, scoring).
#   3. New cell:   exec(open(f'{WORK}/e015_run.py').read(), globals())
#   Resumable: every finished model is written to WORK/e015_results.csv and skipped on a re-run.
#   Cost: ~30-45 min of T4 (~1 compute unit). Nothing is trained; the 28 checkpoints are only read.
#
# PRE-REGISTERED (written 4 Oct 2026, before ANY fine-tuned model was run on TEST2).
#   Already known before writing this (community-1 drafts made locally, scored 4 Oct):
#     community-1 zero-shot TEST2 pooled DER 24.08 @ c0.0 (Yt01 9.22, YT02 26.09, YT03 44.20).
#   Primary measure: TEST2 pooled DER, collar 0.0. Old TEST (2 IRD) results are NOT re-opened.
#   R1 "ship community-1 unchanged still holds": no fine-tuned arm has EVERY seed below community-1 on TEST2.
#      Expectation: holds (28 of 28 fine-tunes lost on TEST).
#   R2 "the gentle-lr win replicates": every E011 (D synth_real, lr 1e-4) seed below every
#      E010 (C synth_real, lr 1e-3) seed on TEST2 (exact p = 0.05). Expectation: plausible, not certain.
#   R3 "the clustering re-tune (E013) still hurts": F c1_tuned above community-1. Expectation: yes.
#   D1 (diagnostic, labelled ORACLE, never a headline): community-1 told the true speaker count.
#      Expectation: helps YT02 most (4 people heard as 2).
#   NOT scored on TEST2: pyannoteAI precision-3. The TEST2 truth was corrected from pyannoteAI drafts
#      (boundary identity 89-100 %), so scoring that model against it would grade it on its own answers.
import os, csv, json, glob, time, tarfile, torch
from pyannote.audio import Pipeline, Inference, Model
from pyannote.audio.pipelines import SpeakerDiarization as SDPipeline

TEST2 = ['Yt01', 'YT02', 'YT03']
C1 = 'pyannote/speaker-diarization-community-1'
EMB = 'pyannote/wespeaker-voxceleb-resnet34-LM'
RES15 = f'{WORK}/e015_results.csv'
F15 = ['part', 'arm', 'seed', 'collar', 'DER', 'miss', 'fa', 'conf', 'per_clip']
KNOWN_C1 = 24.08


def _e015():
    up = f'{WORK}/E015_upload.tgz'
    if not all(os.path.exists(f'{ROOT}/processed/{c}.wav') for c in TEST2):
        if not os.path.exists(up):
            raise SystemExit(f'Put E015_upload.tgz in {WORK} first.')
        with tarfile.open(up) as t:
            t.extractall(ROOT)
    for c in TEST2:
        for f in (f'processed/{c}.wav', f'rttm_truth/{c}.rttm', f'rttm_truth/{c}.uem'):
            assert os.path.exists(f'{ROOT}/{f}'), f'missing {f}'
    assert score.__code__.co_varnames[:3] == ('hyps', 'clips', 'collar'), "Cell 4's score() was replaced - re-run Cell 4"

    def fresh_c1():
        p = Pipeline.from_pretrained(C1, token=HF); p.to(DEVICE); return p

    def swap_segmentation(p, model):                    # identical to 09 Cell 10
        old = p._segmentation
        assert model.specifications.powerset == old.model.specifications.powerset
        d = model.specifications.duration
        p._segmentation = Inference(model, duration=d, step=p.segmentation_step * d,
                                    skip_aggregation=True, batch_size=old.batch_size)
        p.to(DEVICE); return p

    def build_A(seg):                                   # identical to 09 Cell 7
        p = SDPipeline(segmentation=seg, embedding=EMB, token=HF)
        p.instantiate(p.default_parameters()); p.to(DEVICE); return p

    done = set()
    if os.path.exists(RES15):
        done = {(r['part'], r['arm'], r['seed']) for r in csv.DictReader(open(RES15))}

    def run(part, arm, seed, make, **kw):
        if (part, arm, str(seed)) in done:
            print(f'{part} {arm} s{seed}: already scored, skipped'); return
        t0 = time.time(); pipe = make()
        out = f'{WORK}/rttm_e015/{part}_{arm}_s{seed}'; os.makedirs(out, exist_ok=True)
        hyps = {}
        for c in TEST2:
            o = pipe(f'{ROOT}/processed/{c}.wav', **kw)
            ann = o.speaker_diarization if hasattr(o, 'speaker_diarization') else o
            hyps[c] = ann
            with open(f'{out}/{c}.rttm', 'w') as fh: ann.write_rttm(fh)
        new = not os.path.exists(RES15)
        with open(RES15, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=F15)
            if new: w.writeheader()
            for col in (0.0, 0.25):
                der, miss, fa, conf, per = score(hyps, TEST2, col)
                w.writerow(dict(part=part, arm=arm, seed=seed, collar=col, DER=f'{der:.4f}', miss=f'{miss:.4f}',
                                fa=f'{fa:.4f}', conf=f'{conf:.4f}', per_clip=json.dumps({k: round(v, 4) for k, v in per.items()})))
        der0 = score(hyps, TEST2, 0.0)[0]
        print(f'{part} {arm:>14} s{seed}: TEST2 DER {der0:6.2f}  ({time.time() - t0:.0f}s)')
        del pipe; torch.cuda.empty_cache()

    # controls first
    run('B', 'c1_zeroshot', '-', fresh_c1)
    run('A', 'pretrained', '-', lambda: build_A('pyannote/segmentation-3.0'))
    n_true = {c: len(load_rttm(f'{ROOT}/rttm_truth/{c}.rttm', c).labels()) for c in TEST2}
    if ('B', 'c1_oracle', '-') not in done:          # per-clip speaker count -> one call per clip
        hyps, t0 = {}, time.time(); p = fresh_c1()
        for c in TEST2:
            hyps[c] = p(f'{ROOT}/processed/{c}.wav', num_speakers=n_true[c]).speaker_diarization
        new = not os.path.exists(RES15)
        with open(RES15, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=F15)
            if new: w.writeheader()
            for col in (0.0, 0.25):
                der, miss, fa, conf, per = score(hyps, TEST2, col)
                w.writerow(dict(part='B', arm='c1_oracle', seed='-', collar=col, DER=f'{der:.4f}', miss=f'{miss:.4f}',
                                fa=f'{fa:.4f}', conf=f'{conf:.4f}', per_clip=json.dumps({k: round(v, 4) for k, v in per.items()})))
        print(f'B      c1_oracle s-: TEST2 DER {score(hyps, TEST2, 0.0)[0]:6.2f}  ORACLE ({time.time() - t0:.0f}s)')
        del p; torch.cuda.empty_cache()
    pj = f'{WORK}/e013/e013_params.json'
    if os.path.exists(pj):
        prm = json.load(open(pj))
        def tuned():
            p = fresh_c1(); p.instantiate(prm); return p
        run('F', 'c1_tuned', '-', tuned)
    else:
        print('E013 params not found at', pj, '- F skipped')

    # every fine-tuned checkpoint on Drive
    cks = sorted(glob.glob(f'{WORK}/ckpt/*_s*/best.ckpt'))
    print(f'\n{len(cks)} checkpoints found')
    for ck in cks:
        tag = os.path.basename(os.path.dirname(ck))          # e.g. D_synth_real_s1
        part, rest = tag.split('_', 1); arm, seed = rest.rsplit('_s', 1)
        if part == 'A':
            run(part, arm, seed, lambda ck=ck: build_A(Model.from_pretrained(ck)))
        else:
            run(part, arm, seed, lambda ck=ck: swap_segmentation(fresh_c1(), Model.from_pretrained(ck)))

    # ---------------- summary + pre-registered verdicts ----------------
    rows = [r for r in csv.DictReader(open(RES15)) if float(r['collar']) == 0.0]
    by = {}
    for r in rows: by.setdefault((r['part'], r['arm']), []).append(r)
    c1 = float(by[('B', 'c1_zeroshot')][0]['DER'])
    L = [f'===== E015 - everything re-scored on TEST2 (Yt01, YT02, YT03; 13.8 min real), collar 0.0 =====',
         f'community-1 zero-shot here: {c1:.2f}  (local drafts on 4 Oct: {KNOWN_C1}; a gap = version drift)', '',
         f'{"part arm":<22}{"seeds":>6}{"mean":>8}{"min":>8}{"max":>8}{"miss":>7}{"fa":>7}{"conf":>7}  per clip (mean)']
    V = {}
    for (part, arm), rs in sorted(by.items()):
        d = [float(r['DER']) for r in rs]; V[(part, arm)] = d
        mu = lambda k: sum(float(r[k]) for r in rs) / len(rs)
        pcs = [json.loads(r['per_clip']) for r in rs]
        pc = {c: round(sum(p[c] for p in pcs) / len(pcs), 1) for c in TEST2}
        L.append(f'{part + " " + arm:<22}{len(d):>6}{mu("DER"):8.2f}{min(d):8.2f}{max(d):8.2f}{mu("miss"):7.2f}{mu("fa"):7.2f}{mu("conf"):7.2f}  {pc}')
    ft = {k: v for k, v in V.items() if k[1] not in ('c1_zeroshot', 'c1_oracle', 'c1_tuned', 'pretrained', 'c1_selfswap')}
    winners = [f'{p} {a}' for (p, a), d in ft.items() if max(d) < c1]
    n_beat = sum(x < c1 for d in ft.values() for x in d); n_all = sum(len(d) for d in ft.values())
    L += ['', f'Fine-tuned models below community-1: {n_beat} of {n_all}.',
          'R1 (pre-registered): ' + ('HOLDS - no arm beats community-1 on every seed' if not winners
                                     else 'OVERTURNED - every seed beats community-1 for: ' + ', '.join(winners))]
    d, c = V.get(('D', 'synth_real')), V.get(('C', 'synth_real'))
    if d and c:
        L.append('R2 (pre-registered, E011 vs E010 synth+real): ' +
                 ('REPLICATES - gentler is better (p = 0.05)' if max(d) < min(c) else
                  'REVERSED - gentler is worse' if min(d) > max(c) else 'DOES NOT REPLICATE - no detectable difference'))
    if ('F', 'c1_tuned') in V:
        f = V[('F', 'c1_tuned')][0]
        L.append(f'R3 (pre-registered, E013 clustering re-tune): {f:.2f} vs {c1:.2f} -> ' +
                 ('STILL HURTS' if f > c1 + 0.1 else 'NOW HELPS' if f < c1 - 0.1 else 'NO CHANGE'))
    if ('B', 'c1_oracle') in V:
        L.append(f'D1 (ORACLE diagnostic, never a headline): told the true count -> {V[("B", "c1_oracle")][0]:.2f} vs {c1:.2f}')
    text = '\n'.join(L)
    open(f'{WORK}/e015_summary.txt', 'w').write(text + '\n')
    print('\n' + text + f'\n\nSend back: {RES15} and {WORK}/e015_summary.txt')


_e015()
