# queue2.py - 1 Oct ~02:57 SL: runs E013 ahead of the rest of the chain, then resumes overnight.py.
# Why the reorder: after E010 synth_real seeds 0-1, both pre-registered E010 verdicts were already fixed
# (R1 can be neither HELPED nor HURT: seed 0 = 17.76 > 14.29 = min E009-B big, seed 1 = 16.34 < 18.30 = max big;
# R2 failed on both seeds), so seed 2's TEST score cannot change a verdict. Colab warned the runtime has ~50 min
# left, and E013 is the cheapest test with a real chance of beating 7.50. Seed 2 was stopped DURING SCORING
# (not training); its DEV rows are saved and its TEST score is completed when overnight.py resumes E010.
import time, traceback
def _qlog(m):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())} UTC  {m}'
    print(line, flush=True)
    with open(f'{WORK}/overnight_log.txt', 'a') as fh:
        fh.write(line + '\n')
_qlog('START  e013_run.py (moved ahead of the rest of E010: its verdicts were already fixed by seeds 0-1)')
_t = time.time()
try:
    exec(open(f'{WORK}/e013_run.py').read(), globals())
    _qlog(f'DONE   e013_run.py in {(time.time() - _t) / 60:.1f} min')
except KeyboardInterrupt:
    _qlog('STOPPED by hand during e013_run.py')
    raise
except BaseException as _e:
    _qlog(f'FAILED e013_run.py after {(time.time() - _t) / 60:.1f} min: {type(_e).__name__}: {_e}')
    for _l in traceback.format_exc().splitlines()[-25:]:
        _qlog('    ' + _l)
exec(open(f'{WORK}/overnight.py').read(), globals())
