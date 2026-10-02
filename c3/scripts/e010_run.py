# E010 - does SYNTHETIC conversation data make fine-tuning community-1 help (or at least stop hurting)?
# Runs inside the 09_more_data_finetune.ipynb runtime AFTER Cells 3, 4, 5 and 10 (uses ROOT, WORK, HF, DEVICE,
# TRAIN_BIG, DEV, TEST, train(), run_and_score(), swap_segmentation(), fresh_c1(), BASE_B, RES).
#
# Arms (community-1's own segmentation fine-tuned, swapped into an otherwise untouched community-1, as E009 Part B):
#   synth_real = 5 h synthetic conversations + the 45 real training clips (115 min)
#   synth      = 5 h synthetic conversations only
# Synthetic data: simconv.py, built ONLY from the 45 training clips (never DEV, never TEST), seed 0.
# Controls already on file: c1_zeroshot (7.50) and E009 Part B small / big (3 seeds each).
#
# PRE-REGISTERED (written 1 Oct 2026, before running), TEST pooled DER at collar 0.0:
#   R1 "synthetic data helped fine-tuning": every synth_real seed beats every E009-B big seed (exact p = 0.05)
#   R2 "beats the untouched model":        every seed of an arm is below community-1 zero-shot (7.50)
#   Expectation: R1 likely (the 5-6-speaker conversations should undo the damage on the 6-person panel);
#   R2 unlikely (the synthetic voices are the same ~15 people; the main error is voice separation).
import os, sys, csv, json, time, glob, subprocess, importlib
import torch

SIM = globals().get('SIM_DIR', '/content/sim')
SIM_HOURS, SIM_SEED = globals().get('SIM_HOURS', 5.0), 0
ARMS = ['synth_real', 'synth']            # synth_real first: it is the main question

# 1) synthetic data (deterministic: re-running regenerates the identical set)
if not os.path.exists(f'{SIM}/stats.json'):
    try:
        import pyroomacoustics, soundfile
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyroomacoustics', 'soundfile'], check=True)
    sys.path.insert(0, WORK)
    import simconv
    importlib.reload(simconv)
    simconv.generate(ROOT, TRAIN_BIG, SIM, SIM_HOURS, seed=SIM_SEED)
SIM_STATS = json.load(open(f'{SIM}/stats.json'))
print('synthetic set:', SIM_STATS)
SIM_IDS = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{SIM}/wav/*.wav'))
assert not set(SIM_IDS) & set(TEST + DEV), 'synthetic ids collide with real clips'
for c in SIM_IDS:                         # the C3 database looks for audio in ROOT/processed/{uri}.wav
    dst = f'{ROOT}/processed/{c}.wav'
    if not os.path.exists(dst):
        os.symlink(f'{SIM}/wav/{c}.wav', dst)

# 2) protocols: C3.SpeakerDiarization.synth / synth_real (same DEV as E008/E009 for early stopping)
os.makedirs(f'{ROOT}/db/lists', exist_ok=True)
def write_list(name, real, sim):
    with open(f'{ROOT}/db/lists/{name}.lst', 'w') as fh:
        fh.write('\n'.join(real + sim) + '\n')
    with open(f'{ROOT}/db/lists/{name}.rttm', 'w') as fh:
        for c in real:
            for line in open(f'{ROOT}/rttm_truth/{c}.rttm', encoding='utf-8'):
                p = line.split()
                if len(p) >= 8 and p[0] == 'SPEAKER':
                    fh.write(' '.join(p[:8] + ['<NA>', '<NA>']) + '\n')
        for c in sim:
            fh.write(open(f'{SIM}/rttm/{c}.rttm').read())
    with open(f'{ROOT}/db/lists/{name}.uem', 'w') as fh:
        for c in real:
            fh.write(open(f'{ROOT}/rttm_truth/{c}.uem', encoding='utf-8').read())
        for c in sim:
            fh.write(open(f'{SIM}/uem/{c}.uem').read())
write_list('synth_real_train', TRAIN_BIG, SIM_IDS)
write_list('synth_train', [], SIM_IDS)
yml = f'{ROOT}/db/database_e010.yml'
with open(yml, 'w') as fh:
    fh.write(f'''Databases:
  C3: {ROOT}/processed/{{uri}}.wav
Protocols:
  C3:
    SpeakerDiarization:
''' + ''.join(f'''      {arm}:
        train:
          uri: {ROOT}/db/lists/{arm}_train.lst
          annotation: {ROOT}/db/lists/{arm}_train.rttm
          annotated: {ROOT}/db/lists/{arm}_train.uem
        development:
          uri: {ROOT}/db/lists/dev.lst
          annotation: {ROOT}/db/lists/dev.rttm
          annotated: {ROOT}/db/lists/dev.uem
''' for arm in ARMS))
registry.load_database(yml)

# 3) train + score, arm-major so the main question finishes first; everything resumable
for arm in ARMS:
    for seed in SEEDS:
        ck = train('C', arm, seed, BASE_B)
        if scored('C', arm, seed, 'DEV') and scored('C', arm, seed, 'TEST'):
            print(f'C {arm} s{seed}: already scored, skipped')
            continue
        run_and_score('C', arm, seed, swap_segmentation(fresh_c1(), Model.from_pretrained(ck)))
        torch.cuda.empty_cache()

# 4) summary against the pre-registered rules
rows = list(csv.DictReader(open(RES)))
def ders(part, arm, st='TEST'):
    return sorted((r['seed'], float(r['DER']), float(r['miss']), float(r['fa']), float(r['conf']), r['per_clip'])
                  for r in rows if r['part'] == part and r['arm'] == arm and r['set'] == st and float(r['collar']) == 0.0)
L = ['===== E010 - synthetic conversations, community-1 segmentation fine-tuned, TEST (IRD 2), collar 0.0 =====',
     f'synthetic set: {SIM_STATS}', '',
     f'{"arm":<22}{"seed":>5}{"DER":>8}{"miss":>7}{"fa":>7}{"conf":>7}   per clip']
groups = [('B', 'c1_zeroshot', 'community-1 untouched'), ('B', 'small', 'E009-B small (36 min)'),
          ('B', 'big', 'E009-B big (115 min)'), ('C', 'synth', 'E010 synth only (5 h)'),
          ('C', 'synth_real', 'E010 synth + real')]
means = {}
for part, arm, name in groups:
    rs = ders(part, arm)
    for s, d, m, fa, cf, pc in rs:
        L.append(f'{name:<22}{s:>5}{d:8.2f}{m:7.2f}{fa:7.2f}{cf:7.2f}   {pc}')
    if len(rs) > 1:
        mu = lambda k: sum(r[k] for r in rs) / len(rs)
        L.append(f'{name + " MEAN":<22}{"":>5}{mu(1):8.2f}{mu(2):7.2f}{mu(3):7.2f}{mu(4):7.2f}   range {min(r[1] for r in rs):.2f}-{max(r[1] for r in rs):.2f}')
    means[arm] = [r[1] for r in rs]
L.append('')
big, sr_, sy = means.get('big', []), means.get('synth_real', []), means.get('synth', [])
if big and sr_:
    r1 = ('SYNTHETIC DATA HELPED FINE-TUNING (every synth_real seed beat every E009-B big seed, p = 0.05)' if max(sr_) < min(big)
          else 'SYNTHETIC DATA HURT (every synth_real seed worse than every big seed)' if min(sr_) > max(big)
          else 'NO DETECTABLE DIFFERENCE vs E009-B big (seed ranges overlap)')
    L.append(f'R1 (pre-registered): {r1};  mean change vs big {sum(sr_)/len(sr_) - sum(big)/len(big):+.2f} pp')
zs = means.get('c1_zeroshot', [7.498])[0]
for arm, v in (('synth_real', sr_), ('synth', sy)):
    if v:
        L.append(f'R2 {arm}: ' + ('BEATS the untouched model on every seed' if max(v) < zs else
                                   f'does NOT beat the untouched model ({min(v):.2f}-{max(v):.2f} vs {zs:.2f})'))
L += ['', 'DEV (leakage-inflated, sanity only), collar 0.0:']
for part, arm, name in groups:
    d = [x[1] for x in ders(part, arm, 'DEV')]
    if d:
        L.append(f'  {name:<22} mean {sum(d)/len(d):6.2f} (n={len(d)})')
text = '\n'.join(L)
open(f'{WORK}/e010_summary.txt', 'w').write(text + '\n')
print('\n' + text)
