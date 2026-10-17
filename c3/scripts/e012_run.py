# E012 - "panel-style" synthetic set: more 5-6-person conversations and more overlap (~14 %), aimed at the one
# failure every fine-tune showed (the 6-person panel IRD_Recording_2_IIT falls apart). Additional test: the E010
# synthetic set (seed 0) is untouched; this is a SECOND set (seed 1, ids SIM01_*), built from the same 45 training clips.
# ONE change vs E010 synth_real: the synthetic set's style. Same recipe (lr 1e-3), DEV, early stopping, seeds.
# Needs: Cells 3-5 + 10 of the 09 notebook, train_patch.py, and e010_run.py run before (helpers + registry). Part 'E'.
#
# PRE-REGISTERED (written 1 Oct 2026, before running), TEST pooled DER at collar 0.0:
#   R1 "panel-style is better": every E seed beats every E010 synth_real seed (exact p = 0.05)
#   R2 "beats the untouched model": every E seed below community-1 zero-shot (7.50)
#   Also reported: IRD_Recording_2_IIT alone (the 6-person panel), where the damage was concentrated.
import os, sys, csv, json, glob, importlib
import numpy as np, torch

SIM2 = globals().get('SIM2_DIR', '/content/sim_panel')
SIM2_HOURS = globals().get('SIM2_HOURS', 5.0)
if not os.path.exists(f'{SIM2}/stats.json'):
    sys.path.insert(0, WORK)
    import simconv
    importlib.reload(simconv)
    _Corpus, _probs = simconv.Corpus, dict(simconv.N_SPK_PROBS)
    class PanelCorpus(_Corpus):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            g = self.gaps
            self.gaps = np.concatenate([g, g[g < 0], g[g < 0]])      # overlap share of speaker changes 46 % -> 72 %
    try:
        simconv.Corpus = PanelCorpus
        simconv.N_SPK_PROBS = {2: 0.10, 3: 0.20, 4: 0.20, 5: 0.25, 6: 0.25}
        simconv.generate(ROOT, TRAIN_BIG, SIM2, SIM2_HOURS, seed=1)
    finally:
        simconv.Corpus, simconv.N_SPK_PROBS = _Corpus, _probs      # the E010 generator stays exactly as it was
SIM2_STATS = json.load(open(f'{SIM2}/stats.json'))
print('panel-style synthetic set:', SIM2_STATS)
SIM2_IDS = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{SIM2}/wav/*.wav'))
assert all(c.startswith('SIM01_') for c in SIM2_IDS)
for c in SIM2_IDS:
    dst = f'{ROOT}/processed/{c}.wav'
    if not os.path.exists(dst):
        os.symlink(f'{SIM2}/wav/{c}.wav', dst)

with open(f'{ROOT}/db/lists/panel_real_train.lst', 'w') as fh:
    fh.write('\n'.join(TRAIN_BIG + SIM2_IDS) + '\n')
with open(f'{ROOT}/db/lists/panel_real_train.rttm', 'w') as fh:
    for c in TRAIN_BIG:
        for line in open(f'{ROOT}/rttm_truth/{c}.rttm', encoding='utf-8'):
            p = line.split()
            if len(p) >= 8 and p[0] == 'SPEAKER':
                fh.write(' '.join(p[:8] + ['<NA>', '<NA>']) + '\n')
    for c in SIM2_IDS:
        fh.write(open(f'{SIM2}/rttm/{c}.rttm').read())
with open(f'{ROOT}/db/lists/panel_real_train.uem', 'w') as fh:
    for c in TRAIN_BIG:
        fh.write(open(f'{ROOT}/rttm_truth/{c}.uem', encoding='utf-8').read())
    for c in SIM2_IDS:
        fh.write(open(f'{SIM2}/uem/{c}.uem').read())
yml2 = f'{ROOT}/db/database_e012.yml'
with open(yml2, 'w') as fh:
    fh.write(f'''Databases:
  C3: {ROOT}/processed/{{uri}}.wav
Protocols:
  C3:
    SpeakerDiarization:
      panel_real:
        train:
          uri: {ROOT}/db/lists/panel_real_train.lst
          annotation: {ROOT}/db/lists/panel_real_train.rttm
          annotated: {ROOT}/db/lists/panel_real_train.uem
        development:
          uri: {ROOT}/db/lists/dev.lst
          annotation: {ROOT}/db/lists/dev.rttm
          annotated: {ROOT}/db/lists/dev.uem
''')
registry.load_database(yml2)

for seed in SEEDS:
    ck = train('E', 'panel_real', seed, BASE_B)
    if scored('E', 'panel_real', seed, 'DEV') and scored('E', 'panel_real', seed, 'TEST'):
        print(f'E panel_real s{seed}: already scored, skipped')
        continue
    run_and_score('E', 'panel_real', seed, swap_segmentation(fresh_c1(), Model.from_pretrained(ck)))
    torch.cuda.empty_cache()

rows = list(csv.DictReader(open(RES)))
def ders(part, arm, st='TEST'):
    return sorted((r['seed'], float(r['DER']), float(r['miss']), float(r['fa']), float(r['conf']), json.loads(r['per_clip']))
                  for r in rows if r['part'] == part and r['arm'] == arm and r['set'] == st and float(r['collar']) == 0.0)
L = ['===== E012 - panel-style synthetic + real, community-1, TEST (IRD 2), collar 0.0 =====',
     f'panel-style set: {SIM2_STATS}', '',
     f'{"arm":<26}{"seed":>5}{"DER":>8}{"miss":>7}{"fa":>7}{"conf":>7}{"IIT 6-spk":>11}{"Harshana":>10}']
groups = [('B', 'c1_zeroshot', 'community-1 untouched'), ('B', 'big', 'E009-B big (real only)'),
          ('C', 'synth_real', 'E010 synth+real'), ('E', 'panel_real', 'E012 panel-synth+real')]
V = {}
for part, arm, name in groups:
    rs = ders(part, arm)
    for s, d, m, fa, cf, pc in rs:
        L.append(f'{name:<26}{s:>5}{d:8.2f}{m:7.2f}{fa:7.2f}{cf:7.2f}{pc.get("IRD_Recording_2_IIT", float("nan")):11.2f}{pc.get("IRD_Harshana_Silva", float("nan")):10.2f}')
    V[(part, arm)] = [r[1] for r in rs]
c, e = V.get(('C', 'synth_real'), []), V.get(('E', 'panel_real'), [])
zs = (V.get(('B', 'c1_zeroshot')) or [7.498])[0]
L.append('')
if c and e:
    L.append('R1 (pre-registered): ' + ('PANEL-STYLE IS BETTER (every E012 seed beat every E010 seed, p = 0.05)' if max(e) < min(c)
             else 'PANEL-STYLE IS WORSE (every E012 seed worse)' if min(e) > max(c) else 'NO DETECTABLE DIFFERENCE vs E010'))
if e:
    L.append('R2: ' + ('E012 BEATS the untouched model on every seed' if max(e) < zs
                      else f'E012 does NOT beat the untouched model ({min(e):.2f}-{max(e):.2f} vs {zs:.2f})'))
text = '\n'.join(L)
open(f'{WORK}/e012_summary.txt', 'w').write(text + '\n')
print('\n' + text)
