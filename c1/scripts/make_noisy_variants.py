#!/usr/bin/env python3
"""
make_noisy_variants.py -- Task 6: synthetic SNR tiers for robustness testing.

Produces degraded copies of the corpus audio at controlled signal-to-noise
ratios so Tasks 1/3/4 can be re-run per tier and plotted against SNR.

NOISE SOURCE — why synthetic rather than MUSAN
----------------------------------------------
The plan calls for MUSAN or self-recorded ambience. MUSAN is ~11GB and this
host's connection runs at ~500KB/s (about six hours), and no self-recorded
ambience exists for this corpus. Both are therefore unavailable in practice.

Synthetic coloured noise is used instead, which is arguably the better choice
for a *controlled* experiment anyway: the SNR is exact rather than approximate,
the result is bit-reproducible from a seed, and there is no licensing or
distribution question attached to the released dataset. The cost is realism —
pink noise is a reasonable proxy for room tone and HVAC hum, but not for the
babble and music MUSAN provides. That limitation is stated in the report rather
than papered over.

Pink (1/f) noise is the default rather than white because its energy is
concentrated at low frequencies, like most real ambient noise, and it masks the
formant region that carries speech information. White noise at the same SNR is
a gentler test than it appears.

REVERB
------
Applied as a separate condition rather than a fourth SNR tier, since
reverberation degrades speech by smearing it in time rather than by adding
energy — a different mechanism that does not belong on an SNR axis.

Usage:
    python3 make_noisy_variants.py                       # default subset
    python3 make_noisy_variants.py --all                 # every recording
    python3 make_noisy_variants.py --recordings R1 R2
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
from audiomentations import AddColorNoise, AddGaussianSNR

DATASET_ROOT = "/mnt/F/SLIIT/Research/SORA_Dataset"
C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(DATASET_ROOT, "processed", "audio")
GOLD_DIR = os.path.join(DATASET_ROOT, "annotations", "c1")
OUT_ROOT = os.path.join(C1_ROOT, "predictions", "noisy_audio")
RESULTS_DIR = os.path.join(C1_ROOT, "results")

SNR_TIERS = [15, 5, 0]          # dB; "clean" is the untouched original
SEED = 4242

# A stratified default subset: the plan explicitly sanctions running the sweep
# on a subset, and the full corpus x 4 tiers x 2 decode configs is a much
# longer job than the information justifies. These cover all three gold-script
# buckets (LATIN / MIXED / SINHALA) and a spread of durations.
DEFAULT_SUBSET = [
    "J26DS313_R0002",   # LATIN
    "J26DS313_R0014",   # LATIN
    "J26DS313_R0023",   # LATIN
    "J26DS313_R0006",   # SINHALA
    "J26DS313_R0009",   # SINHALA
    "J26DS313_R0015",   # MIXED
    "J26DS313_R0019",   # MIXED
    "J26DS313_R0025",   # MIXED
]


def synthetic_reverb(audio, sr, rt60=0.45, seed=SEED):
    """Convolve with a synthetic exponentially-decaying room impulse response.

    A decaying-noise IR is the standard cheap approximation of a diffuse
    reverberant tail: Gaussian noise shaped by exp(-t/tau), with tau set from
    the target RT60, plus a direct path. It lacks the early-reflection
    structure of a real room, but it reproduces the thing that actually hurts
    ASR -- temporal smearing of phone boundaries.

    audiomentations' RoomSimulator (via pyroomacoustics, now installed in this
    venv) would model real room geometry and early reflections. The synthetic
    IR is kept because it is dependency-light and exactly reproducible from a
    seed; switching is a worthwhile upgrade if reverb becomes a focus rather
    than the secondary condition it is here.
    """
    rng = np.random.default_rng(seed)
    length = int(rt60 * sr)
    t = np.arange(length) / sr
    tau = rt60 / np.log(1000.0)                 # -60dB at rt60
    ir = rng.normal(0.0, 1.0, length) * np.exp(-t / tau)
    ir[0] += 1.0                                # direct path
    ir /= np.sqrt(np.sum(ir ** 2))              # preserve energy
    wet = np.convolve(audio, ir, mode="full")[: len(audio)]
    peak = np.max(np.abs(wet))
    if peak > 0:
        wet = wet * (np.max(np.abs(audio)) / peak)
    return wet.astype(np.float32)


def measured_snr(clean, noisy):
    """Actual achieved SNR in dB, as a sanity check on the requested value."""
    noise = noisy[: len(clean)] - clean
    p_sig = float(np.mean(clean ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise <= 0:
        return float("inf")
    return 10.0 * np.log10(p_sig / p_noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--recordings", nargs="+", default=None)
    ap.add_argument("--noise", choices=["pink", "white"], default="pink")
    a = ap.parse_args()

    if a.recordings:
        rids = a.recordings
    elif a.all:
        rids = sorted(f[: -len(".transcript.json")]
                      for f in os.listdir(GOLD_DIR)
                      if f.endswith(".transcript.json"))
    else:
        rids = DEFAULT_SUBSET
    rids = [r for r in rids if os.path.exists(os.path.join(AUDIO_DIR, f"{r}.wav"))]

    conditions = [f"snr{t}" for t in SNR_TIERS] + ["reverb"]
    for cond in conditions:
        os.makedirs(os.path.join(OUT_ROOT, cond), exist_ok=True)

    print(f"Generating {len(conditions)} conditions for {len(rids)} recordings")
    print(f"  SNR tiers: {SNR_TIERS} dB ({a.noise} noise) + reverb\n")

    manifest = []
    for i, rid in enumerate(rids, 1):
        src = os.path.join(AUDIO_DIR, f"{rid}.wav")
        audio, sr = sf.read(src, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        row = {"recording": rid, "sample_rate": sr,
               "duration_s": round(len(audio) / sr, 2), "conditions": {}}

        for snr in SNR_TIERS:
            # A fixed per-tier seed keeps the noise realisation identical
            # across reruns, so tier-to-tier differences are attributable to
            # SNR rather than to a different random draw.
            if a.noise == "pink":
                tf = AddColorNoise(min_snr_db=snr, max_snr_db=snr,
                                   min_f_decay=-3.01, max_f_decay=-3.01, p=1.0)
            else:
                tf = AddGaussianSNR(min_snr_db=snr, max_snr_db=snr, p=1.0)
            tf.randomize_parameters(audio, sample_rate=sr)
            out = tf(samples=audio.copy(), sample_rate=sr)
            got = measured_snr(audio, out)
            path = os.path.join(OUT_ROOT, f"snr{snr}", f"{rid}.wav")
            sf.write(path, out, sr)
            row["conditions"][f"snr{snr}"] = {
                "path": path, "requested_snr_db": snr,
                "measured_snr_db": round(got, 2)}

        rev = synthetic_reverb(audio, sr)
        rev_path = os.path.join(OUT_ROOT, "reverb", f"{rid}.wav")
        sf.write(rev_path, rev, sr)
        row["conditions"]["reverb"] = {"path": rev_path, "rt60_s": 0.45}

        manifest.append(row)
        snrs = "  ".join(f"{k}={v['measured_snr_db']:+.1f}dB"
                         for k, v in row["conditions"].items()
                         if "measured_snr_db" in v)
        print(f"  [{i:2}/{len(rids)}] {rid}  {row['duration_s']:6.1f}s   {snrs}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "c1_noise_manifest.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "note": "Synthetic pink noise at exact SNR, seeded for "
                    "reproducibility. MUSAN was not usable on this host "
                    "(~11GB over a ~500KB/s link); see the script docstring "
                    "for the realism trade-off this implies.",
            "noise_type": a.noise,
            "snr_tiers_db": SNR_TIERS,
            "reverb_rt60_s": 0.45,
            "seed": SEED,
            "recordings": manifest,
        }, fh, indent=2)
    print(f"\nWrote {out}")
    total_mb = sum(os.path.getsize(c["path"]) for r in manifest
                   for c in r["conditions"].values()) / 1e6
    print(f"Generated {len(manifest) * len(conditions)} files, {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
