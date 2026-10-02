# E014 - gentle fine-tune (lr 1e-4) on SYNTHETIC-ONLY data. Combines the two least-damaging directions so far:
#   E010 synth-only (lr 1e-3): TEST 15.93 / 17.20 / 13.64, mean 15.59
#   E011 synth+real (lr 1e-4): TEST 14.47 / 13.32 / 14.41, mean 14.07  (pre-registered win over E010 synth+real)
# ONE change vs E010 synth-only: the learning rate (1e-4, exactly as E011). Same synthetic set (seed 0), DEV,
# early stopping, seeds 0-2. Results: part 'G', arm 'synth'. Needs e010_run.py run before in this runtime
# (synthetic set + C3.SpeakerDiarization.synth protocol registered).
#
# PRE-REGISTERED (written 1 Oct 2026 ~14:45 SL, before running), TEST pooled DER at collar 0.0:
#   R1 "synthetic-only is the better gentle recipe": every G seed beats every E011 (D synth_real) seed (p = 0.05)
#   R2 "gentle helps synthetic-only too": every G seed beats every E010 synth-only (C synth) seed (p = 0.05)
#   R3 "beats the untouched model": every G seed below community-1 zero-shot (7.50)
#   Expectation: R2 plausible (it worked for synth+real); R1 a coin flip; R3 unlikely.
import csv, torch
from pyannote.audio.core.model import Model as _BaseModel


def _run_e014():
    lr = 1e-4
    orig = _BaseModel.configure_optimizers
    _BaseModel.configure_optimizers = lambda self: torch.optim.Adam(self.parameters(), lr=lr)
    try:
        for seed in SEEDS:
            ck = train('G', 'synth', seed, BASE_B)
            if scored('G', 'synth', seed, 'DEV') and scored('G', 'synth', seed, 'TEST'):
                print(f'G synth s{seed}: already scored, skipped')
                continue
            run_and_score('G', 'synth', seed, swap_segmentation(fresh_c1(), Model.from_pretrained(ck)))
            torch.cuda.empty_cache()
    finally:
        _BaseModel.configure_optimizers = orig        # never leak the change into other experiments

    rows = list(csv.DictReader(open(RES)))

    def ders(part, arm, st='TEST'):
        return sorted((r['seed'], float(r['DER']), float(r['miss']), float(r['fa']), float(r['conf']), r['per_clip'])
                      for r in rows if r['part'] == part and r['arm'] == arm and r['set'] == st and float(r['collar']) == 0.0)

    L = ['===== E014 - gentle fine-tune (lr 1e-4) on SYNTHETIC ONLY, community-1, TEST (IRD 2), collar 0.0 =====', '',
         f'{"arm":<28}{"seed":>5}{"DER":>8}{"miss":>7}{"fa":>7}{"conf":>7}   per clip']
    groups = [('B', 'c1_zeroshot', 'community-1 untouched'), ('C', 'synth', 'E010 synth-only (1e-3)'),
              ('D', 'synth_real', 'E011 synth+real (1e-4)'), ('G', 'synth', 'E014 synth-only (1e-4)')]
    V = {}
    for part, arm, name in groups:
        rs = ders(part, arm)
        for s, d, m, fa, cf, pc in rs:
            L.append(f'{name:<28}{s:>5}{d:8.2f}{m:7.2f}{fa:7.2f}{cf:7.2f}   {pc}')
        if len(rs) > 1:
            mu = lambda k: sum(r[k] for r in rs) / len(rs)
            L.append(f'{name + " MEAN":<28}{"":>5}{mu(1):8.2f}{mu(2):7.2f}{mu(3):7.2f}{mu(4):7.2f}   range {min(r[1] for r in rs):.2f}-{max(r[1] for r in rs):.2f}')
        V[(part, arm)] = [r[1] for r in rs]
    g, d, c = V.get(('G', 'synth'), []), V.get(('D', 'synth_real'), []), V.get(('C', 'synth'), [])
    zs = (V.get(('B', 'c1_zeroshot')) or [7.498])[0]

    def rule(a, b, better, worse):
        return better if max(a) < min(b) else worse if min(a) > max(b) else 'NO DETECTABLE DIFFERENCE'
    L.append('')
    if g and d:
        L.append('R1 (pre-registered, vs E011): ' + rule(g, d, 'SYNTH-ONLY IS THE BETTER GENTLE RECIPE (p = 0.05)', 'SYNTH-ONLY IS WORSE'))
    if g and c:
        L.append('R2 (pre-registered, vs E010 synth-only): ' + rule(g, c, 'GENTLE HELPS SYNTH-ONLY TOO (p = 0.05)', 'GENTLE IS WORSE HERE'))
    if g:
        L.append('R3: ' + ('E014 BEATS the untouched model on every seed' if max(g) < zs
                          else f'E014 does NOT beat the untouched model ({min(g):.2f}-{max(g):.2f} vs {zs:.2f})'))
    text = '\n'.join(L)
    open(f'{WORK}/e014_summary.txt', 'w').write(text + '\n')
    print('\n' + text)


_run_e014()
