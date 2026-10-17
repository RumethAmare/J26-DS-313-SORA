# E011 - "gentle" fine-tune: same data as E010 synth_real, 10x smaller learning rate (1e-4 instead of Adam's 1e-3).
# Question: all 18 fine-tunes so far (E008/E009/E010) made real-audio DER worse. Is the damage caused by the
# fine-tuning being too aggressive (overwriting what community-1 learned from its large training set)?
# ONE change vs E010 synth_real: the learning rate. Everything else identical (data, DEV, early stopping, seeds).
# Runs after e010_run.py in the same runtime (needs SIM set + synth_real protocol registered). Results: part 'D'.
#
# PRE-REGISTERED (written 1 Oct 2026, before running), TEST pooled DER at collar 0.0:
#   R1 "gentler is better": every D seed beats every E010 synth_real seed (exact p = 0.05)
#   R2 "beats the untouched model": every D seed below community-1 zero-shot (7.50)
#   Expectation: R1 plausible (less forgetting); R2 still unlikely.
import csv, torch
from pyannote.audio.core.model import Model as _BaseModel

LR_GENTLE = 1e-4
_orig_opt = _BaseModel.configure_optimizers
_BaseModel.configure_optimizers = lambda self: torch.optim.Adam(self.parameters(), lr=LR_GENTLE)
try:
    for seed in SEEDS:
        ck = train('D', 'synth_real', seed, BASE_B)
        if scored('D', 'synth_real', seed, 'DEV') and scored('D', 'synth_real', seed, 'TEST'):
            print(f'D synth_real s{seed}: already scored, skipped')
            continue
        run_and_score('D', 'synth_real', seed, swap_segmentation(fresh_c1(), Model.from_pretrained(ck)))
        torch.cuda.empty_cache()
finally:
    _BaseModel.configure_optimizers = _orig_opt       # never leak the change into other experiments

rows = list(csv.DictReader(open(RES)))
def ders(part, arm, st='TEST'):
    return sorted((r['seed'], float(r['DER']), float(r['miss']), float(r['fa']), float(r['conf']), r['per_clip'])
                  for r in rows if r['part'] == part and r['arm'] == arm and r['set'] == st and float(r['collar']) == 0.0)
L = ['===== E011 - gentle fine-tune (lr 1e-4) on synthetic + real, community-1, TEST (IRD 2), collar 0.0 =====', '',
     f'{"arm":<26}{"seed":>5}{"DER":>8}{"miss":>7}{"fa":>7}{"conf":>7}   per clip']
groups = [('B', 'c1_zeroshot', 'community-1 untouched'), ('B', 'big', 'E009-B big (lr 1e-3)'),
          ('C', 'synth_real', 'E010 synth+real (1e-3)'), ('D', 'synth_real', 'E011 synth+real (1e-4)')]
V = {}
for part, arm, name in groups:
    rs = ders(part, arm)
    for s, d, m, fa, cf, pc in rs:
        L.append(f'{name:<26}{s:>5}{d:8.2f}{m:7.2f}{fa:7.2f}{cf:7.2f}   {pc}')
    if len(rs) > 1:
        mu = lambda k: sum(r[k] for r in rs) / len(rs)
        L.append(f'{name + " MEAN":<26}{"":>5}{mu(1):8.2f}{mu(2):7.2f}{mu(3):7.2f}{mu(4):7.2f}   range {min(r[1] for r in rs):.2f}-{max(r[1] for r in rs):.2f}')
    V[(part, arm)] = [r[1] for r in rs]
c, d = V.get(('C', 'synth_real'), []), V.get(('D', 'synth_real'), [])
zs = (V.get(('B', 'c1_zeroshot')) or [7.498])[0]
L.append('')
if c and d:
    L.append('R1 (pre-registered): ' + ('GENTLER IS BETTER (every E011 seed beat every E010 seed, p = 0.05)' if max(d) < min(c)
             else 'GENTLER IS WORSE (every E011 seed worse)' if min(d) > max(c) else 'NO DETECTABLE DIFFERENCE vs E010'))
if d:
    L.append('R2: ' + ('E011 BEATS the untouched model on every seed' if max(d) < zs
                      else f'E011 does NOT beat the untouched model ({min(d):.2f}-{max(d):.2f} vs {zs:.2f})'))
text = '\n'.join(L)
open(f'{WORK}/e011_summary.txt', 'w').write(text + '\n')
print('\n' + text)
