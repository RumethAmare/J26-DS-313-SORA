# E013 - tune community-1's CLUSTERING hyper-parameters on our own clips, then score the frozen TEST once.
# Why: every fine-tune of the segmentation model (E008/E009/E010) made TEST worse, and E006 showed the dominant error
# on our data is speaker confusion / under-counting (22 of 51 clips get too few speakers). Clustering decides how
# many voices there are, needs no training (no forgetting), and is the adaptation route pyannote itself recommends
# ("tune the pipeline hyper-parameters on your own data"). No model weight changes.
# Needs: Cells 3-5 + 10 of the 09 notebook (score, run_and_score, fresh_c1, TEST, DEV, TRAIN_SMALL, NEW, WORK, ROOT).
# Results: WORK/e013/ (grid CSV, chosen parameters, summary) + part 'F' rows in e009_results.csv.
#
# PRE-REGISTERED (written 1 Oct 2026 ~02:30, before running):
#   Tuning set = the 49 non-TEST clips (DEV + the 45 training clips). TEST (2 IRD clips) is never looked at during
#   the search. Objective: pooled DER, collar 0.0, blind (no speaker count), protocol v1.
#   Search (coordinate descent, fixed in advance): 0) shipped parameters;  1) threshold {0.35 .. 0.70} x Fb
#   {0.2 .. 3.2} at shipped Fa;  2) Fa {0.03, 0.07, 0.15, 0.3} at the best pair;  3) threshold +/- 0.025;
#   4) segmentation.min_duration_off {0, 0.1, 0.2, 0.4}. Winner = lowest tuning DER (ties -> the earlier config).
#   TEST is then scored ONCE with the winner (fresh pipeline, the same run_and_score as every other experiment).
#   Rule (deterministic pipeline, so no seeds; TEST is 2 clips):
#     HELPED  = TEST pooled DER below 7.50 AND neither TEST clip worse by more than 0.5 pp
#     HURT    = TEST pooled DER above 7.50 + 0.5
#     otherwise NO CLEAR CHANGE (incl. "one clip better, the other worse")
#   Expectation: tuning-set DER drops 1-3 pp (known-count gave 18.07 -> 16.08 in E006, a rough ceiling for fixing
#   the count). On TEST: Harshana may gain (its 3rd speaker is missed), the 6-person IIT panel may be over-split.
#   Net: NO CLEAR CHANGE is my best guess.
import os, csv, json, time, copy
import torch

# everything runs inside one function, so no helper name can leak into (or clobber) the notebook namespace
# (lesson from the E010 crash: demo_app.py's score() replaced Cell 4's score()).
def _run_e013():

    E13 = f'{WORK}/e013'
    os.makedirs(E13, exist_ok=True)
    TUNE = DEV + TRAIN_SMALL + NEW
    assert len(TUNE) == len(set(TUNE)) == 49 and not set(TUNE) & set(TEST), 'tuning set must be the 49 non-TEST clips'
    C1_TEST_ZEROSHOT = 7.4991                        # B c1_zeroshot (E009 Part B), = E004 7.498

    def as_ann(out):
        return out.speaker_diarization if hasattr(out, 'speaker_diarization') else out

    def true_n(c):
        return len(load_rttm(f'{ROOT}/rttm_truth/{c}.rttm', c).crop(load_uem(f'{ROOT}/rttm_truth/{c}.uem', c)).labels())
    TRUE_N = {c: true_n(c) for c in TUNE + TEST}

    # 1) one heavy pass: segmentation + embeddings are computed once per clip and cached inside the file dicts
    #    (pyannote's own "training" cache), so each parameter setting only re-runs clustering + reconstruction.
    pipe13 = fresh_c1()
    SHIPPED = copy.deepcopy(pipe13.parameters(instantiated=True))
    print('shipped parameters:', SHIPPED)
    pipe13.training = True
    FILES = {c: {'uri': c, 'audio': f'{ROOT}/processed/{c}.wav'} for c in TUNE}
    cache_ready = []

    def ensure_cache():                               # built lazily: a finished search re-runs without it
        if cache_ready:
            return
        t0 = time.time()
        for i, c in enumerate(TUNE):
            pipe13.apply(FILES[c])
            if i % 10 == 9:
                print(f'  cached {i + 1}/{len(TUNE)} clips [{time.time() - t0:.0f}s]', flush=True)
        print(f'cache built in {time.time() - t0:.0f}s')
        cache_ready.append(True)

    def params(th, fa, fb, mdo):
        p = copy.deepcopy(SHIPPED)
        p['clustering']['threshold'], p['clustering']['Fa'], p['clustering']['Fb'] = th, fa, fb
        p['segmentation']['min_duration_off'] = mdo
        return p

    GRID = f'{E13}/e013_grid.csv'
    GF = ['stage', 'threshold', 'Fa', 'Fb', 'min_duration_off', 'DER', 'miss', 'fa', 'conf', 'count_ok', 'under', 'over', 'secs']
    tried = {}                                        # key -> DER; resumable from the grid CSV
    if os.path.exists(GRID):
        for i, r in enumerate(csv.DictReader(open(GRID))):
            tried[tuple(round(float(r[k]), 4) for k in ('threshold', 'Fa', 'Fb', 'min_duration_off'))] = (float(r['DER']), i)

    def trial(stage, th, fa, fb, mdo):
        key = (round(th, 4), round(fa, 4), round(fb, 4), round(mdo, 4))
        if key in tried:
            return tried[key][0]
        ensure_cache()
        t = time.time()
        pipe13.instantiate(params(*key))
        hyps = {c: as_ann(pipe13.apply(FILES[c])) for c in TUNE}
        der, miss, fa_, conf, _ = score(hyps, TUNE, 0.0)
        n = {c: len(hyps[c].labels()) for c in TUNE}
        row = dict(stage=stage, threshold=key[0], Fa=key[1], Fb=key[2], min_duration_off=key[3], DER=f'{der:.4f}',
                   miss=f'{miss:.4f}', fa=f'{fa_:.4f}', conf=f'{conf:.4f}',
                   count_ok=sum(n[c] == TRUE_N[c] for c in TUNE), under=sum(n[c] < TRUE_N[c] for c in TUNE),
                   over=sum(n[c] > TRUE_N[c] for c in TUNE), secs=f'{time.time() - t:.1f}')
        new = not os.path.exists(GRID)
        with open(GRID, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=GF)
            if new:
                w.writeheader()
            w.writerow(row)
        tried[key] = (der, len(tried))
        print(f'  [{stage}] th {key[0]:.3f} Fa {key[1]:.2f} Fb {key[2]:.2f} mdo {key[3]:.2f} -> tune DER {der:6.2f} '
              f'(miss {miss:.2f} fa {fa_:.2f} conf {conf:.2f}) count ok {row["count_ok"]}/49 under {row["under"]} over {row["over"]} [{row["secs"]}s]', flush=True)
        return der

    def best():
        return min(tried, key=lambda k: tried[k])     # (DER, order): ties go to the earlier config

    S = SHIPPED
    th0, fa0, fb0, mdo0 = S['clustering']['threshold'], S['clustering']['Fa'], S['clustering']['Fb'], S['segmentation']['min_duration_off']
    trial('0 shipped', th0, fa0, fb0, mdo0)
    for th in (0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        for fb in (0.2, 0.4, 0.8, 1.6, 3.2):
            trial('1 th x Fb', th, fa0, fb, mdo0)
    th1, _, fb1, _ = best()
    for fa in (0.03, 0.07, 0.15, 0.3):
        trial('2 Fa', th1, fa, fb1, mdo0)
    th2, fa2, fb2, _ = best()
    for th in (th2 - 0.025, th2 + 0.025):
        trial('3 fine th', th, fa2, fb2, mdo0)
    th3, fa3, fb3, _ = best()
    for mdo in (0.0, 0.1, 0.2, 0.4):
        trial('4 min_dur_off', th3, fa3, fb3, mdo)
    WIN = best()
    BEST = params(*WIN)
    json.dump(BEST, open(f'{E13}/e013_params.json', 'w'), indent=1)
    print('winner:', BEST, f'tune DER {tried[WIN][0]:.2f} vs shipped {tried[(round(th0, 4), round(fa0, 4), round(fb0, 4), round(mdo0, 4))][0]:.2f}')
    del FILES, pipe13
    torch.cuda.empty_cache()

    # 2) TEST, once, with a fresh community-1 carrying the winning parameters (same scoring path as every experiment)
    if not (scored('F', 'c1_tuned', '-', 'DEV') and scored('F', 'c1_tuned', '-', 'TEST')):
        tuned = fresh_c1()
        tuned.instantiate(BEST)
        run_and_score('F', 'c1_tuned', '-', tuned)
        torch.cuda.empty_cache()

    # 3) summary against the pre-registered rule
    rows = list(csv.DictReader(open(RES)))
    def row(part, arm, st, col=0.0):
        for r in rows:
            if (r['part'], r['arm'], r['set']) == (part, arm, st) and float(r['collar']) == col:
                return r
    b, f = row('B', 'c1_zeroshot', 'TEST'), row('F', 'c1_tuned', 'TEST')
    bpc, fpc = json.loads(b['per_clip']), json.loads(f['per_clip'])
    d = float(f['DER'])
    worse = [c for c in TEST if fpc[c] > bpc[c] + 0.5]
    e13_verdict = ('HELPED' if d < C1_TEST_ZEROSHOT and not worse else
               'HURT' if d > C1_TEST_ZEROSHOT + 0.5 else 'NO CLEAR CHANGE')
    g = list(csv.DictReader(open(GRID)))
    g0 = g[0]
    gw = min(g, key=lambda r: float(r['DER']))
    L = ['===== E013 - community-1 clustering tuned on the 49 non-TEST clips; TEST (IRD 2) scored once =====',
         f'shipped parameters: {SHIPPED}', f'winning parameters: {BEST}', f'configurations tried: {len(g)}', '',
         'TUNING SET (49 clips, blind, collar 0.0)    DER    miss     fa   conf  count ok  under  over',
         f'  shipped                                {float(g0["DER"]):6.2f} {float(g0["miss"]):6.2f} {float(g0["fa"]):6.2f} {float(g0["conf"]):6.2f}   {g0["count_ok"]:>5}/49 {g0["under"]:>5} {g0["over"]:>5}',
         f'  tuned                                  {float(gw["DER"]):6.2f} {float(gw["miss"]):6.2f} {float(gw["fa"]):6.2f} {float(gw["conf"]):6.2f}   {gw["count_ok"]:>5}/49 {gw["under"]:>5} {gw["over"]:>5}',
         '', 'TEST (frozen), collar 0.0                   DER    miss     fa   conf   per clip']
    for name, r in (('community-1 shipped', b), ('community-1 tuned (E013)', f)):
        L.append(f'  {name:<40}{float(r["DER"]):6.2f} {float(r["miss"]):6.2f} {float(r["fa"]):6.2f} {float(r["conf"]):6.2f}   {r["per_clip"]}')
    b25, f25 = row('B', 'c1_zeroshot', 'TEST', 0.25), row('F', 'c1_tuned', 'TEST', 0.25)
    L += [f'  collar 0.25: shipped {float(b25["DER"]):.2f} -> tuned {float(f25["DER"]):.2f}', '',
          f'VERDICT (pre-registered): {e13_verdict}  (TEST {C1_TEST_ZEROSHOT:.2f} -> {d:.2f}, {d - C1_TEST_ZEROSHOT:+.2f} pp'
          + (f'; worse by >0.5 pp on: {", ".join(worse)}' if worse else '') + ')']
    text = '\n'.join(L)
    open(f'{E13}/e013_summary.txt', 'w').write(text + '\n')
    print('\n' + text)

_run_e013()
