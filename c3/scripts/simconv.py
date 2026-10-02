"""simconv.py - C3 E010 synthetic conversation generator.

Builds NEW multi-speaker conversations out of the speaker-pure speech in the C3 TRAINING clips only
(never DEV R0023-R0026, never the frozen IRD TEST clips).

Method, each step from published work:
  * simulated conversations: turns are re-assembled with pause / overlap durations drawn from the
    real corpus's own statistics (Landini et al., Interspeech 2022)
  * speaker augmentation: speed perturbation 0.9 / 1.0 / 1.1 turns one real voice into three
    pseudo-speakers (Yamamoto et al. 2019; Zhou et al. 2024)
  * acoustics: simulated room echo (pyroomacoustics image method) per speaker position, random
    per-speaker level (so there are quiet speakers), and REAL background noise cut from the
    non-speech parts of the same recordings

Identity safety (labels are per-clip, the same friend can appear in many clips under different labels):
  a conversation takes its speakers in up to three CLIP GROUPS. Inside a group: different people from one
  clip, one shared speed factor. Each group gets a DIFFERENT factor (0.9 / 1.0 / 1.1). So the same real
  voice can never appear twice at the same speed inside one conversation.

Outputs (under OUT): wav/<id>.wav (16 kHz mono int16), rttm/<id>.rttm, uem/<id>.uem,
manifest.csv (one row per conversation), stats.json (corpus-level checks).
"""
import os, json, csv, time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, fftconvolve

SR = 16000
SPEEDS = (0.9, 1.0, 1.1)
N_SPK_PROBS = {2: 0.25, 3: 0.30, 4: 0.20, 5: 0.125, 6: 0.125}
CONV_LEN = (150.0, 330.0)          # seconds, similar to the real clips
MIN_TURN, MIN_BC, MAX_BC = 0.8, 0.25, 1.0
MIN_SPK_PURE = 15.0                # a real speaker needs >= 15 s of clean speech to be used


# ------------------------------------------------------------------ reading the real corpus
def read_rttm(path):
    segs = []
    for line in open(path, encoding='utf-8'):
        p = line.split()
        if len(p) >= 8 and p[0] == 'SPEAKER' and float(p[4]) > 0:
            segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return sorted(segs)

def read_uem(path):
    out = []
    for line in open(path, encoding='utf-8'):
        p = line.split()
        if len(p) >= 4:
            out.append((float(p[2]), float(p[3])))
    return out

def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]

def subtract(a, b):
    """intervals a minus union(b); both lists of (s, e)."""
    b = merge(b)
    out = []
    for s, e in a:
        cur = s
        for bs, be in b:
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
            if cur >= e:
                break
        if cur < e:
            out.append((cur, e))
    return out

def intersect(a, b):
    out = []
    for s, e in a:
        for bs, be in b:
            lo, hi = max(s, bs), min(e, be)
            if hi > lo:
                out.append((lo, hi))
    return out


class Corpus:
    def __init__(self, root, clips, log=print):
        self.audio, self.turns, self.bcs, self.noise = {}, {}, {}, {}
        self.gaps, self.bc_rate_num, self.bc_rate_den = [], 0, 0
        self.clip_speakers = {}
        for c in clips:
            x, sr = sf.read(f'{root}/processed/{c}.wav', dtype='float32')
            assert sr == SR and x.ndim == 1, f'{c}: expected 16 kHz mono'
            self.audio[c] = x
            segs = read_rttm(f'{root}/rttm_truth/{c}.rttm')
            uem = read_uem(f'{root}/rttm_truth/{c}.uem')
            labs = sorted({l for _, _, l in segs})
            spk_ok = []
            for lab in labs:
                own = [(s, e) for s, e, l in segs if l == lab]
                other = [(s, e) for s, e, l in segs if l != lab]
                pure = intersect(subtract(merge(own), other), uem)
                pure = [(s + 0.01, e - 0.01) for s, e in pure if e - s > 0.05]
                turns = [(s, e) for s, e in pure if e - s >= MIN_TURN]
                bcs = [(s, e) for s, e in pure if MIN_BC <= e - s <= MAX_BC]
                if sum(e - s for s, e in turns) >= MIN_SPK_PURE:
                    self.turns[(c, lab)] = turns
                    self.bcs[(c, lab)] = bcs
                    spk_ok.append(lab)
            self.clip_speakers[c] = spk_ok
            # turn-taking statistics: gap between consecutive segments of DIFFERENT speakers
            for (s1, e1, l1), (s2, e2, l2) in zip(segs, segs[1:]):
                if l1 != l2 and -3.0 < s2 - e1 < 3.0:
                    self.gaps.append(s2 - e1)
            # back-channel rate: short segments lying inside another speaker's segment
            for s, e, l in segs:
                if e - s <= MAX_BC:
                    self.bc_rate_den += 1
                    if any(l2 != l and s2 <= s and e <= e2 for s2, e2, l2 in segs):
                        self.bc_rate_num += 1
            # background noise: annotated region minus all speech (0.15 s safety margin)
            speech = [(s - 0.15, e + 0.15) for s, e, _ in segs]
            quiet = [(s, e) for s, e in subtract(uem, speech) if e - s >= 0.3]
            if quiet:
                self.noise[c] = np.concatenate([x[int(s * SR):int(e * SR)] for s, e in quiet])
        self.gaps = np.array(self.gaps)
        n_turn_segs = sum(len(v) for v in self.turns.values())
        self.p_bc = min(0.35, self.bc_rate_num / max(1, n_turn_segs))
        log(f'corpus: {len(clips)} clips, {len(self.turns)} usable real speakers, '
            f'{sum(e - s for v in self.turns.values() for s, e in v) / 60:.1f} min clean turns, '
            f'{sum(len(v) for v in self.noise.values()) / SR / 60:.1f} min background noise, '
            f'{len(self.gaps)} speaker-change gaps (overlap share {np.mean(self.gaps < 0):.2f}), '
            f'back-channel prob per turn {self.p_bc:.2f}')

    def cut(self, key, s, e):
        c, _ = key
        return self.audio[c][int(s * SR):int(e * SR)].copy()


# ------------------------------------------------------------------ signal helpers
def speed(x, f):
    if f == 1.0:
        return x
    up, down = {0.9: (10, 9), 1.1: (10, 11)}[f]      # 0.9x speed = longer, 1.1x = shorter
    return resample_poly(x, up, down).astype(np.float32)

def fade(x, ms=10):
    n = min(len(x) // 2, int(SR * ms / 1000))
    if n > 0:
        r = np.linspace(0, 1, n, dtype=np.float32)
        x[:n] *= r
        x[-n:] *= r[::-1]
    return x

def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))

def room_rirs(n_src, rng):
    """One shoebox room, one mic, n_src speaker positions -> list of RIRs (or None = dry)."""
    import pyroomacoustics as pra
    L, W, H = rng.uniform(4, 10), rng.uniform(3, 8), rng.uniform(2.5, 3.5)
    rt60 = rng.uniform(0.2, 0.7)
    e_abs, max_order = pra.inverse_sabine(rt60, [L, W, H])
    room = pra.ShoeBox([L, W, H], fs=SR, materials=pra.Material(e_abs), max_order=min(max_order, 20))
    mic = np.array([rng.uniform(1, L - 1), rng.uniform(1, W - 1), rng.uniform(0.8, 1.5)])
    room.add_microphone(mic)
    for _ in range(n_src):
        for _try in range(50):
            p = np.array([rng.uniform(0.5, L - 0.5), rng.uniform(0.5, W - 0.5), rng.uniform(1.1, 1.8)])
            if 0.5 <= np.linalg.norm(p[:2] - mic[:2]) <= 3.0:
                break
        room.add_source(p)
    room.compute_rir()
    rirs = [np.asarray(room.rir[0][i], dtype=np.float32) for i in range(n_src)]
    return [r / (np.max(np.abs(r)) + 1e-9) for r in rirs], dict(room=[round(L, 2), round(W, 2), round(H, 2)], rt60=round(rt60, 2))


# ------------------------------------------------------------------ one conversation
def pick_speakers(corp, n, rng):
    """Returns list of (key, factor). Speakers come in up to three CLIP GROUPS; inside a group they are
    different people from one clip; each group gets its own speed factor (0.9 / 1.0 / 1.1). So the same
    real voice can never appear twice at the same speed in one conversation (labels are per clip, and the
    same friend may sit in several clips)."""
    factors = list(SPEEDS)
    rng.shuffle(factors)
    clips = [c for c, labs in corp.clip_speakers.items() if len(labs) >= 2]
    rng.shuffle(clips)
    chosen, remaining = [], n
    for f, c in zip(factors, clips):
        labs = list(corp.clip_speakers[c])
        rng.shuffle(labs)
        take = min(remaining, len(labs))
        chosen += [((c, l), f) for l in labs[:take]]
        remaining -= take
        if remaining == 0:
            break
    assert remaining == 0, 'not enough speakers for this conversation size'
    return chosen

def make_conversation(corp, cid, rng, clean=False):
    n = int(rng.choice(list(N_SPK_PROBS), p=list(N_SPK_PROBS.values())))
    spk = pick_speakers(corp, n, rng)
    names = [f'S{i + 1:02d}' for i in range(n)]
    target = rng.uniform(*CONV_LEN)
    w = rng.dirichlet(np.full(n, 1.2))
    w = np.maximum(w, 0.04)
    w /= w.sum()
    tracks = [[] for _ in range(n)]            # list of (start_sec, dry_audio)
    labels = []                                # (start, end, name)
    t_end, prev, prev_start = 0.0, -1, -1.0
    turns_done = np.zeros(n, int)
    while t_end < target or turns_done.min() < 2:
        if turns_done.min() < 2 and t_end >= target:
            cand = [i for i in range(n) if turns_done[i] < 2 and i != prev] or [i for i in range(n) if i != prev]
            i = cand[rng.integers(len(cand))]
        else:
            p = w.copy()
            if prev >= 0:
                p[prev] = 0
            i = int(rng.choice(n, p=p / p.sum()))
        key, f = spk[i]
        pieces, spans, pos = [], [], 0.0      # a turn = 1-3 clean pieces of the same voice, short pauses between
        parts = 1 + (rng.random() < 0.3) + (rng.random() < 0.1)
        for q in range(parts):
            s, e = corp.turns[key][rng.integers(len(corp.turns[key]))]
            if e - s > 12:                     # long monologue piece: take a random window
                s = rng.uniform(s, e - 12)
                e = s + rng.uniform(3, 12)
            a = fade(speed(corp.cut(key, s, e), f))
            pieces.append(a)
            spans.append((pos, pos + len(a) / SR))
            pos += len(a) / SR
            if q < parts - 1:
                z = np.zeros(int(SR * rng.uniform(0.1, 0.5)), np.float32)
                pieces.append(z)
                pos += len(z) / SR
        utt = np.concatenate(pieces)
        gap = float(corp.gaps[rng.integers(len(corp.gaps))])
        start = max(0.0, t_end + gap)
        start = max(start, prev_start + 0.3)   # never start before the previous turn started + 0.3 s
        dur = len(utt) / SR
        tracks[i].append((start, utt))
        for a0, a1 in spans:                   # label each piece, not the pauses between them
            labels.append((start + a0, start + a1, names[i]))
        prev_start = start
        # optional back-channel from someone else, inside this turn
        if n > 1 and dur > 1.5 and rng.random() < corp.p_bc:
            j = int(rng.choice([k for k in range(n) if k != i]))
            kj, fj = spk[j]
            if corp.bcs[kj]:
                s, e = corp.bcs[kj][rng.integers(len(corp.bcs[kj]))]
                bc = fade(speed(corp.cut(kj, s, e), fj))
                bstart = start + rng.uniform(0.3, max(0.31, dur - len(bc) / SR - 0.1))
                tracks[j].append((bstart, bc))
                labels.append((bstart, bstart + len(bc) / SR, names[j]))
        t_end = max(t_end, start + dur)
        turns_done[i] += 1
        prev = i

    total = t_end + 0.5
    N = int(total * SR) + SR
    reverb = (rng.random() < 0.8) and not clean
    rirs, room = room_rirs(n, rng) if reverb else ([None] * n, dict(room=None, rt60=0.0))
    gains_db = rng.uniform(-10, 0, n)
    if n >= 3 and rng.random() < 0.5:          # one clearly quieter person (the case the model fails on)
        gains_db[rng.integers(n)] = rng.uniform(-14, -8)
    mix = np.zeros(N, np.float32)
    for i in range(n):
        dry = np.zeros(N, np.float32)
        for st, a in tracks[i]:
            k = int(st * SR)
            dry[k:k + len(a)] += a[:N - k]
        dry /= (rms(dry[dry != 0]) if np.any(dry) else 1.0)
        wet = fftconvolve(dry, rirs[i])[:N].astype(np.float32) if rirs[i] is not None else dry
        mix += wet * (10 ** (gains_db[i] / 20)) / (rms(wet[np.abs(wet) > 1e-4]) + 1e-9) * rms(dry[dry != 0])
    snr = None
    if rng.random() < 0.85 and corp.noise and not clean:
        src = list(corp.noise)[rng.integers(len(corp.noise))]
        nz = corp.noise[src]
        reps = int(np.ceil(N / max(1, len(nz))))
        nz = np.tile(nz, reps)[:N]
        speech_rms = rms(mix[np.abs(mix) > 1e-4])
        snr = float(rng.uniform(5, 25))
        mix += nz / (rms(nz) + 1e-9) * speech_rms * 10 ** (-snr / 20)
    peak = float(np.max(np.abs(mix)))
    mix *= 0.9 / peak if peak > 0 else 1.0
    labels.sort()
    meta = dict(id=cid, n_spk=n, dur_s=round(total, 2), speakers=';'.join(f'{k[0]}:{k[1]}@{f}' for k, f in spk),
                reverb=reverb, **room, gains_db=';'.join(f'{g:.1f}' for g in gains_db),
                snr_db=None if snr is None else round(snr, 1))
    return mix[:int(total * SR)], labels, meta


# ------------------------------------------------------------------ corpus-level checks
def overlap_and_speech(labels, total):
    ev = sorted([(s, 1) for s, _, _ in labels] + [(e, -1) for _, e, _ in labels])
    act, last, sp, ov = 0, 0.0, 0.0, 0.0
    for t, d in ev:
        if act >= 1:
            sp += t - last
        if act >= 2:
            ov += t - last
        act += d
        last = t
    return sp, ov


def generate(root, clips, out, hours, seed=0, log=print):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    corp = Corpus(root, clips, log)
    for d in ('wav', 'rttm', 'uem'):
        os.makedirs(f'{out}/{d}', exist_ok=True)
    rows, done, k = [], 0.0, 0
    tot_sp = tot_ov = 0.0
    while done < hours * 3600:
        cid = f'SIM{seed:02d}_{k:04d}'
        audio, labels, meta = make_conversation(corp, cid, rng)
        sf.write(f'{out}/wav/{cid}.wav', audio, SR, subtype='PCM_16')
        with open(f'{out}/rttm/{cid}.rttm', 'w') as fh:
            for s, e, lab in labels:
                fh.write(f'SPEAKER {cid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {lab} <NA> <NA>\n')
        dur = len(audio) / SR
        with open(f'{out}/uem/{cid}.uem', 'w') as fh:
            fh.write(f'{cid} 1 0.000 {dur:.3f}\n')
        sp, ov = overlap_and_speech(labels, dur)
        tot_sp += sp
        tot_ov += ov
        meta['overlap_pct'] = round(100 * ov / max(sp, 1e-9), 2)
        rows.append(meta)
        done += dur
        k += 1
    with open(f'{out}/manifest.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    counts = {}
    for r in rows:
        counts[r['n_spk']] = counts.get(r['n_spk'], 0) + 1
    stats = dict(conversations=len(rows), hours=round(done / 3600, 2), overlap_pct=round(100 * tot_ov / tot_sp, 2),
                 speaker_count_distribution=dict(sorted(counts.items())),
                 reverb_share=round(np.mean([r['reverb'] for r in rows]), 2),
                 noise_share=round(np.mean([r['snr_db'] is not None for r in rows]), 2),
                 source_clips=len(clips), seed=seed, minutes_taken=round((time.time() - t0) / 60, 1))
    json.dump(stats, open(f'{out}/stats.json', 'w'), indent=1)
    log(json.dumps(stats))
    return rows, stats


if __name__ == '__main__':
    import sys
    root, out, hours = sys.argv[1], sys.argv[2], float(sys.argv[3])
    clips = sys.argv[4].split(',')
    generate(root, clips, out, hours)
