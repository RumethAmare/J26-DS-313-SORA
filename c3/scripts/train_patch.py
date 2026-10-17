# train_patch.py - makes train() safe to resume after a runtime death or an interrupt.
# ModelCheckpoint writes best.ckpt DURING training, so a killed run leaves a checkpoint behind that the original
# train() would treat as finished. A run counts as finished only if train_log.csv has its row (written after fit).
# Unfinished leftovers are moved aside (renamed, never deleted) and the run is trained again from scratch.
import os, csv, time, shutil
if '_train_orig' not in globals():
    _train_orig = train
def _finished(part, arm, seed):
    if not os.path.exists(TRAIN_LOG):
        return False
    return any((r['part'], r['arm'], r['seed']) == (part, arm, str(seed)) for r in csv.DictReader(open(TRAIN_LOG)))
def train(part, arm, seed, base):
    out = f'{WORK}/ckpt/{part}_{arm}_s{seed}'
    if os.path.exists(f'{out}/best.ckpt') and not _finished(part, arm, seed):
        aside = f'{out}_unfinished_{int(time.time())}'
        shutil.move(out, aside)
        print(f'{part} {arm} s{seed}: found an UNFINISHED run, moved to {aside}; training again')
    return _train_orig(part, arm, seed, base)
print('train() now resume-safe')
