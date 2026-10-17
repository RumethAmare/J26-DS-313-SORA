# overnight.py - runs the queued ADDITIONAL experiments (E010 -> E011 -> E012) one after another, unattended.
# Each stage is isolated: a failure is logged and the next stage still runs. Progress and errors are appended to
# WORK/overnight_log.txt, so they can be read from Drive while the runtime is busy.
#
# Why this file exists (1 Oct 2026, ~02:00): the first E010 attempt trained C synth_real s0 fine, then crashed at
# scoring with "AttributeError: 'dict' object has no attribute 'crop'". Cause: the demo cell exec'd demo_app.py in
# the same namespace, and demo_app.py defines its own score(ref, hyp, uem), which replaced Cell 4's
# score(hyps, clips, collar). Nothing was written to the results file (the crash came before the first row).
# Fix: put Cell 4's score() back, verbatim, before anything is scored. (The Gradio demo in this runtime then uses
# the wrong score(); re-exec demo_app.py if the demo is needed again in this runtime.)
import time, traceback

def _log(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())} UTC  {msg}'
    print(line, flush=True)
    with open(f'{WORK}/overnight_log.txt', 'a') as fh:
        fh.write(line + '\n')

# ---- Cell 4's score(), verbatim ----
def score(hyps, clips, collar=0.0):
    pooled = dict.fromkeys(('confusion', 'missed detection', 'false alarm', 'total'), 0.0)
    per = {}
    for c in clips:
        ref = load_rttm(f'{ROOT}/rttm_truth/{c}.rttm', c)
        uem = load_uem(f'{ROOT}/rttm_truth/{c}.uem', c)
        comp = DiarizationErrorRate(collar=collar, skip_overlap=False)(ref, hyps[c], uem=uem, detailed=True)
        per[c] = sum(comp[k] for k in ('confusion', 'missed detection', 'false alarm')) / comp['total'] * 100
        for k in pooled:
            pooled[k] += comp[k]
    t = pooled['total']
    der = sum(pooled[k] for k in ('confusion', 'missed detection', 'false alarm')) / t * 100
    return der, pooled['missed detection'] / t * 100, pooled['false alarm'] / t * 100, pooled['confusion'] / t * 100, per
# ------------------------------------
assert score.__code__.co_varnames[:3] == ('hyps', 'clips', 'collar')
_log('overnight runner started; Cell 4 score() restored (the demo cell had replaced it)')

exec(open(f'{WORK}/train_patch.py').read(), globals())
for _stage in ('e010_run.py', 'e011_run.py', 'e012_run.py'):
    _t0 = time.time()
    _log(f'START  {_stage}')
    try:
        exec(open(f'{WORK}/{_stage}').read(), globals())
        _log(f'DONE   {_stage} in {(time.time() - _t0) / 60:.1f} min')
    except KeyboardInterrupt:
        _log(f'STOPPED by hand during {_stage}')
        raise
    except BaseException as _e:                       # SystemExit included: log it and move on to the next stage
        _log(f'FAILED {_stage} after {(time.time() - _t0) / 60:.1f} min: {type(_e).__name__}: {_e}')
        for _l in traceback.format_exc().splitlines()[-25:]:
            _log('    ' + _l)
    torch.cuda.empty_cache()
_log('overnight runner finished')
