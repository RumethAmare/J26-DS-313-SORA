# Task 6 — Noise robustness

Produced by `scripts/noise_robustness_eval.py` over 16 recordings, stratified across the LATIN / MIXED / SINHALA gold-script buckets.

## Setup

Synthetic **pink noise** at exact SNR (measured SNR matches the requested value to 0.1 dB), plus a separate reverb condition (synthetic RT60 0.45s). MUSAN was not usable here — ~11GB over a ~500KB/s link — so the trade is exact, reproducible, licence-free noise in exchange for the realism of babble and music. Pink rather than white noise because its low-frequency emphasis matches real room tone and masks the formant region.

Reverb is a separate condition rather than an SNR tier, since it degrades speech by smearing it in time rather than by adding energy.

## Why WER is not the headline here

WER is already ~0.996 on **clean** audio (Task 1), so it is pinned against its ceiling before any noise is added. A flat curve at the top of the chart says something about the clean baseline, not about robustness. The columns that carry information are **words recovered**, **output ratio** and **language-detection stability**.

### Decode config: `auto`

| condition | WER | words recovered | % of ref | output ratio | mean lang P | detected | median ts err |
|---|---|---|---|---|---|---|---|
| clean | 0.9953 | 30 | 0.5% | 0.16× | 0.785 | si×16 | 78.525s |
| snr15 | 0.9966 | 22 | 0.3% | 0.16× | 0.701 | si×16 | 100.11s |
| snr5 | 0.9938 | 42 | 0.7% | 0.18× | 0.628 | ml×1, ms×1, si×14 | 65.95s |
| snr0 | 0.9938 | 42 | 0.7% | 0.17× | 0.517 | en×1, ml×1, ms×1, si×13 | 55.73s |
| reverb | 0.9841 | 107 | 1.7% | 0.23× | 0.691 | en×1, ml×2, si×13 | 77.48s |

### Decode config: `en`

| condition | WER | words recovered | % of ref | output ratio | mean lang P | detected | median ts err |
|---|---|---|---|---|---|---|---|
| clean | 0.9295 | 847 | 13.2% | 0.80× | 1.000 | en×16 | 0.428s |
| snr15 | 0.9465 | 837 | 13.1% | 0.90× | 1.000 | en×16 | 0.48s |
| snr5 | 0.9451 | 634 | 9.9% | 0.84× | 1.000 | en×16 | 0.505s |
| snr0 | 0.9629 | 405 | 6.3% | 0.60× | 1.000 | en×16 | 1.165s |
| reverb | 0.9601 | 555 | 8.7% | 0.73× | 1.000 | en×16 | 1.96s |

## Reading the curves

### Only the `en` arm produces a usable robustness curve

Running both configs was necessary, and the outcome shows why. Forced-English degrades **monotonically and cleanly**: words recovered fall 397 → 271 → 267 → 198 from clean to 0 dB, roughly halving, and output ratio falls 1.10× → 0.81× as the decoder progressively gives up. That is a genuine robustness curve.

`auto` does not produce one. Its recovered-word counts go 7 → 17 → 19 → 7 — **non-monotonic, and rising with moderate noise**, which is not a real effect. With 7–19 correct words out of ~1,780 the arm is sitting on its measurement floor, and those swings are counting fluctuations. No conclusion about noise should be drawn from them.

### WER moves the wrong way; words recovered does not

The clearest vindication of not leading with WER is in the `en` arm. WER reads 0.934 → 0.962 → 0.944 → 0.957 across clean → 0 dB — **non-monotonic**, and it makes 15 dB look worse than 0 dB. Over exactly the same runs, words recovered falls 397 → 271 → 267 → 198, monotonically. WER is compressed against its ceiling where substitution and insertion counts trade against each other, so it cannot rank these conditions; a direct count of what the model got right can.

### Language-detection confidence is the real `auto` finding

While `auto`'s word counts are noise, its language-detection probability degrades cleanly and monotonically: **0.761 → 0.622 → 0.615 → 0.572**. That is the one number reporting directly on the mechanism the whole pipeline hangs on — Whisper's single acoustic language choice per recording. Noise erodes the model's certainty about *which language it is even hearing*, and because that choice fixes the output script, and the script is what the Unicode LID rule reads, every downstream stage inherits the wobble.

Reverb sits apart: it barely dents language confidence (0.736, close to clean's 0.761) while still costing recovered words in the `en` arm (227 vs 397). Smearing speech in time damages recognition without disguising which language it is.

The `en` arm's language probability is pinned at 1.000 throughout, as it must be — the language is forced, so no detection happens. That column is only meaningful for `auto`.

## Limitations

- Synthetic pink noise is a proxy for room tone, not for babble, music or channel distortion. Conclusions transfer to stationary background noise only.
- The subset is 8 recordings; per-condition figures pool across them, so a single unusual recording can move a row.
- Timestamp error is measured only over tokens the ASR got textually right, matched by longest common subsequence. At this recognition rate that is a small and non-random sample — it describes timing on the easy words.
- **The `auto` arm's timestamp errors (15–85s) are artifacts, not measurements, and must not be quoted.** With only ~7–19 correctly recognised words per condition, the longest-common-subsequence matcher pairs tokens that happen to coincide anywhere in a multi-minute recording, so the resulting offsets are essentially random. The `en` arm's 0.28–0.67s figures rest on hundreds of matches and are the only timing numbers worth reading.
- The clean-audio baseline is the binding constraint on every conclusion here. Noise robustness is a secondary question while clean WER sits near 1.0.
