# E006 — does telling community-1 HOW MANY people are talking fix its errors?
# Runs after Cells 3-4 of 09_more_data_finetune.ipynb (uses ROOT, WORK, HF, DEVICE, clip lists, load_rttm, load_uem).
# Conditions (protocol v1: explicit UEM, overlap included, collars 0.0 / 0.25):
#   blind        = community-1 exactly as shipped. This is the HEADLINE condition (and a reproduction check vs E004).
#   known_count  = num_speakers = true count       -> LABELLED condition "participant count known" (trap #2: never the headline)
#   min_count    = min_speakers = true count       -> LABELLED condition "at least N participants"
# PRE-REGISTERED READING (written before running, 30 Sep 2026), on ALL-51 pooled DER at collar 0.0:
#   known_count <= 2/3 of blind          -> counting is the main failure; a count-aware system is a viable honest "after"
#   known_count  > 0.9 x blind           -> counting is NOT the problem; the voices themselves are hard to separate (embedding work)
#   in between                           -> counting explains part of it
import os, csv, time, torch
from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate

E6 = f'{WORK}/e006'
os.makedirs(E6, exist_ok=True)
RES6 = f'{E6}/e006_per_clip.csv'
ALL51 = TEST + DEV + TRAIN_SMALL + NEW
assert len(set(ALL51)) == 51
GROUP = {c: ('NEW' if c in NEW else 'E004') for c in ALL51}
CONDS = ['blind', 'known_count', 'min_count']
F6 = ['cond', 'clip', 'group', 'true_spk', 'found_spk', 'uem_s'] + \
     [f'{k}_c{col}' for col in (0.0, 0.25) for k in ('total', 'conf', 'miss', 'fa')]

def truth(c):
    ref = load_rttm(f'{ROOT}/rttm_truth/{c}.rttm', c)
    uem = load_uem(f'{ROOT}/rttm_truth/{c}.uem', c)
    return ref, uem, len(ref.crop(uem).labels())

done = set()
if os.path.exists(RES6):
    done = {(r['cond'], r['clip']) for r in csv.DictReader(open(RES6))}
print(f'E006: {len(done)} of {len(CONDS) * 51} clip-runs already done')

pipe6 = Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token=HF)
pipe6.to(DEVICE)
t0 = time.time()
for cond in CONDS:
    os.makedirs(f'{E6}/rttm_{cond}', exist_ok=True)
    for c in ALL51:
        if (cond, c) in done:
            continue
        ref, uem, n = truth(c)
        kw = {} if cond == 'blind' else ({'num_speakers': n} if cond == 'known_count' else {'min_speakers': n})
        out = pipe6(f'{ROOT}/processed/{c}.wav', **kw)
        hyp = out.speaker_diarization if hasattr(out, 'speaker_diarization') else out
        with open(f'{E6}/rttm_{cond}/{c}.rttm', 'w') as fh:
            hyp.write_rttm(fh)
        row = dict(cond=cond, clip=c, group=GROUP[c], true_spk=n, found_spk=len(hyp.labels()),
                   uem_s=round(uem.duration(), 3))
        for col in (0.0, 0.25):
            comp = DiarizationErrorRate(collar=col, skip_overlap=False)(ref, hyp, uem=uem, detailed=True)
            for k, key in (('total', 'total'), ('conf', 'confusion'), ('miss', 'missed detection'), ('fa', 'false alarm')):
                row[f'{k}_c{col}'] = round(float(comp[key]), 4)
        new_file = not os.path.exists(RES6)
        with open(RES6, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=F6)
            if new_file:
                w.writeheader()
            w.writerow(row)
        der = 100 * (row['conf_c0.0'] + row['miss_c0.0'] + row['fa_c0.0']) / row['total_c0.0']
        print(f'{cond:<12} {c:<20} spk {n}->{row["found_spk"]}  DER {der:6.2f}   [{time.time() - t0:5.0f}s]')
    torch.cuda.empty_cache()

# ---------------- summary ----------------
rows = list(csv.DictReader(open(RES6)))
def pool(rs, col=0.0):
    T = sum(float(r[f'total_c{col}']) for r in rs)
    f = lambda k: sum(float(r[f'{k}_c{col}']) for r in rs) / T * 100
    return f('conf') + f('miss') + f('fa'), f('miss'), f('fa'), f('conf')
blind = {r['clip']: r for r in rows if r['cond'] == 'blind'}
under = {c for c, r in blind.items() if int(r['found_spk']) < int(r['true_spk'])}
groups = (('ALL 51', lambda c: True), ('E004 old 27', lambda c: GROUP[c] == 'E004'),
          ('NEW 24', lambda c: GROUP[c] == 'NEW'), ('TEST (IRD 2)', lambda c: c in TEST),
          (f'undercounted blind ({len(under)})', lambda c: c in under))
L = ['===== E006 — community-1 with / without the true speaker count (protocol v1) =====',
     'blind = headline. known_count / min_count = LABELLED conditions, never the headline.', '',
     f'{"group":<26}{"condition":<13}{"DER c0":>8}{"miss":>7}{"fa":>7}{"conf":>7}{"DER c.25":>10}{"count ok":>10}']
for gname, sel in groups:
    for cond in CONDS:
        rs = [r for r in rows if r['cond'] == cond and sel(r['clip'])]
        if not rs:
            continue
        d0, m, fa, cf = pool(rs); d25 = pool(rs, 0.25)[0]
        ok = sum(int(r['found_spk']) == int(r['true_spk']) for r in rs)
        L.append(f'{gname:<26}{cond:<13}{d0:8.2f}{m:7.2f}{fa:7.2f}{cf:7.2f}{d25:10.2f}{ok:>6}/{len(rs)}')
    L.append('')
allb = pool([r for r in rows if r['cond'] == 'blind'])[0]
allk = pool([r for r in rows if r['cond'] == 'known_count'])[0]
L.append(f'reproduction: blind ALL-51 = {allb:.2f} (E004b benchmark from saved drafts: 18.07 @ c0.0)')
ratio = allk / allb
reading = ('COUNTING IS THE MAIN FAILURE (known_count <= 2/3 of blind)' if ratio <= 2 / 3 else
           'COUNTING IS NOT THE PROBLEM (known_count > 0.9 x blind)' if ratio > 0.9 else
           'COUNTING EXPLAINS PART OF THE ERROR (between 2/3 and 0.9 of blind)')
L.append(f'known_count / blind = {ratio:.2f}  ->  PRE-REGISTERED READING: {reading}')
L += ['', 'biggest per-clip changes blind -> known_count (collar 0.0):']
chg = []
for c in ALL51:
    k = next((r for r in rows if r['cond'] == 'known_count' and r['clip'] == c), None)
    b = blind.get(c)
    if k and b:
        db = 100 * (float(b['conf_c0.0']) + float(b['miss_c0.0']) + float(b['fa_c0.0'])) / float(b['total_c0.0'])
        dk = 100 * (float(k['conf_c0.0']) + float(k['miss_c0.0']) + float(k['fa_c0.0'])) / float(k['total_c0.0'])
        chg.append((dk - db, c, db, dk, b['true_spk'], b['found_spk']))
for d, c, db, dk, t, f in sorted(chg)[:8] + [('...', '', 0, 0, '', '')] + sorted(chg)[-4:]:
    L.append('  ...' if d == '...' else f'  {c:<20} {db:6.2f} -> {dk:6.2f}  ({d:+.2f})   blind count {t}->{f}')
text = '\n'.join(L)
open(f'{E6}/e006_summary.txt', 'w').write(text + '\n')
print('\n' + text)
