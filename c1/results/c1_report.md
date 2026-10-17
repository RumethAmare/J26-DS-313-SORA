# C1 Task 1 — Off-the-shelf ASR baseline (script-normalized + ablation)

Scores faster-whisper against the gold C1 transcripts under two
different normalizations, across a language-forcing ablation.

## Why two WER numbers

Gold transcripts are not written in one script. Nine of 27 recordings are
100% romanized Latin (`mama`, `thiyenawa`); most of the rest are majority
Sinhala Unicode. faster-whisper auto-detects **one** language per recording
and decodes the whole thing in that script, so a romanized reference gets
compared against a Sinhala-Unicode hypothesis.

- **raw-script WER** — lowercase + strip punctuation only. Script mismatch is
  charged as error.
- **normalized WER** — both sides transliterated to one phonetic skeleton
  (`script_normalize.py`) first, so script choice costs nothing.

**The expected result did not materialise.** The working assumption was that
script mismatch was badly inflating WER and that normalizing would reveal
substantially better real performance. It does not: the gap is only about
0.3–0.6 WER points. The artifact is real but small, and off-the-shelf
performance is genuinely near-total failure rather than a measurement
illusion. Both numbers are reported so that claim is checkable.

### A scoring bug found while building this

The normalization previously used for C1 scoring,
`re.sub(r"[^\w\s]", " ", text)`, silently corrupts Sinhala. Sinhala is an
abugida whose vowels attach as *combining marks* (Unicode category Mn), and
Python's `\w` does not match Mn — so every vowel sign became a space and each
word shattered into loose consonants (`ට්‍රිප් එකට` → `ට ර ප එකට`).

The damage is not neutral noise: it makes scores look *better* than reality.
A 72-word Sinhala utterance becomes ~53 single-character tokens, and frequent
letters (ක, න, ම, ප) then match the equally-shredded hypothesis by
coincidence, crediting matches that never occurred. Any previously quoted C1
WER computed that way — including the ~93.7% figure in the implementation
plan — is optimistic for exactly the recordings written in Sinhala. Corrected,
raw-script WER is ~0.999. `script_normalize._squash()` now filters by Unicode
category instead, keeping letters, numbers and marks.

## Ablation summary

| config | micro raw WER | micro norm WER | gap | macro norm WER |
|---|---|---|---|---|
| `small_auto` | 0.9993 | 0.9960 | +0.0033 | 1.0023 |
| `small_en` | 0.9898 | 0.9839 | +0.0059 | 1.1257 |
| `small_si` | 1.0016 | 0.9981 | +0.0035 | 1.0272 |

## Why WER alone is the wrong headline here

Every configuration lands near 0.99 WER, which reads as "they all fail
the same way". They do not. The error composition shows two opposite
failure modes that WER collapses onto the same number.

| config | words recovered | % of reference | output ratio | S | D | I | behaviour |
|---|---|---|---|---|---|---|---|
| `small_auto` | 28/5798 | 0.5% | 0.23x | 22% | 78% | 0% | abstains |
| `small_en` | 1121/5798 | 19.3% | 1.09x | 74% | 8% | 18% | over-generates |
| `small_si` | 31/5798 | 0.5% | 0.27x | 26% | 73% | 0% | abstains |

Read the *words recovered* column, not the WER column. Sinhala-mode
Whisper (`small_auto`, `small_si`) does not mis-transcribe this audio so
much as decline to transcribe it: roughly three quarters of its errors are
deletions and it emits only a quarter of the expected words. Forced-English
decoding emits slightly more than the reference length and recovers an
order of magnitude more reference words, paying for it in substitutions and
insertions — which is why its WER barely moves.

That asymmetry is a genuine signal about the data, not a decoding curiosity.
These recordings carry enough English that an English-forced decoder finds
real purchase on them, while the Sinhala-forced decoder — nominally the
"correct" setting for Sinhala-English speech — mostly produces nothing.
Any C1 evaluation that quotes WER alone will rank these configurations as
equivalent and miss this entirely, so downstream tasks should track words
recovered and error composition alongside WER.

## Breakdown by gold script

The artifact is concentrated in the romanized-gold bucket, which is the
point: those recordings were never as badly recognized as raw WER implied.

| config | gold script | n | micro raw WER | micro norm WER | gap |
|---|---|---|---|---|---|
| `small_auto` | LATIN | 9 | 0.9993 | 0.9941 | +0.0052 |
| `small_auto` | MIXED | 15 | 0.9993 | 0.9966 | +0.0027 |
| `small_auto` | SINHALA | 2 | 1.0000 | 1.0000 | +0.0000 |
| `small_en` | LATIN | 9 | 1.0093 | 1.0066 | +0.0027 |
| `small_en` | MIXED | 15 | 0.9825 | 0.9752 | +0.0073 |
| `small_en` | SINHALA | 2 | 0.9944 | 0.9886 | +0.0058 |
| `small_si` | LATIN | 9 | 0.9987 | 0.9927 | +0.0060 |
| `small_si` | MIXED | 15 | 1.0027 | 1.0000 | +0.0027 |
| `small_si` | SINHALA | 2 | 1.0000 | 1.0000 | +0.0000 |

## Per-recording detail

### `small_auto`

| recording | gold script | detected | #words | raw WER | norm WER | gap |
|---|---|---|---|---|---|---|
| J26DS313_R0010 | MIXED | ml | 64 | 1.000 | 0.952 | +0.048 |
| J26DS313_R0004 | LATIN | si | 75 | 1.000 | 0.973 | +0.027 |
| J26DS313_R0028 | LATIN | si | 81 | 1.000 | 0.988 | +0.012 |
| J26DS313_R0026 | MIXED | si | 267 | 0.996 | 0.989 | +0.008 |
| J26DS313_R0015 | MIXED | si | 284 | 0.997 | 0.989 | +0.007 |
| J26DS313_R0002 | LATIN | si | 114 | 1.000 | 0.991 | +0.009 |
| J26DS313_R0017 \* | LATIN | si | 262 | 0.996 | 0.992 | +0.004 |
| J26DS313_R0022 | MIXED | si | 294 | 0.997 | 0.993 | +0.003 |
| J26DS313_R0014 | LATIN | si | 180 | 1.000 | 0.994 | +0.006 |
| J26DS313_R0021 | MIXED | si | 362 | 0.997 | 0.994 | +0.003 |
| J26DS313_R0013 | MIXED | si | 191 | 0.995 | 0.995 | +0.000 |
| J26DS313_R0011 | MIXED | si | 214 | 0.995 | 0.995 | +0.000 |
| J26DS313_R0023 | LATIN | si | 215 | 1.000 | 0.995 | +0.005 |
| J26DS313_R0027 | MIXED | si | 717 | 0.997 | 0.996 | +0.001 |
| J26DS313_R0018 | LATIN | si | 267 | 1.000 | 0.996 | +0.004 |
| J26DS313_R0025 | MIXED | si | 419 | 1.000 | 0.998 | +0.002 |
| J26DS313_R0003 | LATIN | si | 70 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0005 | MIXED | si | 83 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0006 | SINHALA | si | 108 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0009 | SINHALA | si | 72 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0012 | MIXED | si | 202 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0016 | MIXED | si | 239 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0019 | MIXED | si | 398 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0020 | MIXED | si | 349 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0024 | LATIN | si | 249 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0008 \* | MIXED | si | 22 | 1.227 | 1.227 | +0.000 |

### `small_en`

| recording | gold script | detected | #words | raw WER | norm WER | gap |
|---|---|---|---|---|---|---|
| J26DS313_R0012 | MIXED | en | 202 | 0.708 | 0.708 | +0.000 |
| J26DS313_R0018 | LATIN | en | 267 | 0.712 | 0.708 | +0.004 |
| J26DS313_R0025 | MIXED | en | 419 | 0.821 | 0.815 | +0.006 |
| J26DS313_R0019 | MIXED | en | 398 | 0.844 | 0.844 | +0.000 |
| J26DS313_R0020 | MIXED | en | 349 | 0.842 | 0.845 | -0.003 |
| J26DS313_R0026 | MIXED | en | 267 | 0.846 | 0.846 | +0.001 |
| J26DS313_R0013 | MIXED | en | 191 | 0.864 | 0.853 | +0.011 |
| J26DS313_R0002 | LATIN | en | 114 | 0.860 | 0.860 | +0.000 |
| J26DS313_R0010 | MIXED | en | 64 | 0.984 | 0.889 | +0.096 |
| J26DS313_R0024 | LATIN | en | 249 | 0.920 | 0.920 | +0.000 |
| J26DS313_R0006 | SINHALA | en | 108 | 0.991 | 0.944 | +0.046 |
| J26DS313_R0011 | MIXED | en | 214 | 0.963 | 0.948 | +0.014 |
| J26DS313_R0027 | MIXED | en | 717 | 0.964 | 0.962 | +0.001 |
| J26DS313_R0004 | LATIN | en | 75 | 0.987 | 0.973 | +0.013 |
| J26DS313_R0021 | MIXED | en | 362 | 1.028 | 1.019 | +0.008 |
| J26DS313_R0009 | SINHALA | en | 72 | 1.000 | 1.060 | -0.060 |
| J26DS313_R0003 | LATIN | en | 70 | 1.100 | 1.086 | +0.014 |
| J26DS313_R0015 | MIXED | en | 284 | 1.151 | 1.128 | +0.024 |
| J26DS313_R0017 \* | LATIN | en | 262 | 1.130 | 1.130 | +0.000 |
| J26DS313_R0014 | LATIN | en | 180 | 1.139 | 1.139 | +0.000 |
| J26DS313_R0005 | MIXED | en | 83 | 1.181 | 1.145 | +0.036 |
| J26DS313_R0023 | LATIN | en | 215 | 1.181 | 1.177 | +0.005 |
| J26DS313_R0016 | MIXED | en | 239 | 1.226 | 1.213 | +0.012 |
| J26DS313_R0022 | MIXED | en | 294 | 1.276 | 1.273 | +0.003 |
| J26DS313_R0028 | LATIN | en | 81 | 1.284 | 1.284 | +0.000 |
| J26DS313_R0008 \* | MIXED | en | 22 | 4.545 | 4.500 | +0.045 |

### `small_si`

| recording | gold script | detected | #words | raw WER | norm WER | gap |
|---|---|---|---|---|---|---|
| J26DS313_R0028 | LATIN | si | 81 | 0.975 | 0.975 | +0.000 |
| J26DS313_R0010 | MIXED | si | 64 | 0.984 | 0.984 | +0.000 |
| J26DS313_R0003 | LATIN | si | 70 | 1.000 | 0.986 | +0.014 |
| J26DS313_R0004 | LATIN | si | 75 | 1.000 | 0.987 | +0.013 |
| J26DS313_R0011 | MIXED | si | 214 | 0.991 | 0.991 | +0.000 |
| J26DS313_R0002 | LATIN | si | 114 | 1.000 | 0.991 | +0.009 |
| J26DS313_R0016 | MIXED | si | 239 | 1.000 | 0.992 | +0.008 |
| J26DS313_R0024 | LATIN | si | 249 | 1.000 | 0.992 | +0.008 |
| J26DS313_R0018 | LATIN | si | 267 | 1.000 | 0.993 | +0.007 |
| J26DS313_R0025 | MIXED | si | 419 | 1.000 | 0.993 | +0.007 |
| J26DS313_R0022 | MIXED | si | 294 | 0.997 | 0.993 | +0.003 |
| J26DS313_R0014 | LATIN | si | 180 | 1.000 | 0.994 | +0.006 |
| J26DS313_R0027 | MIXED | si | 717 | 0.999 | 0.994 | +0.004 |
| J26DS313_R0012 | MIXED | si | 202 | 1.000 | 0.995 | +0.005 |
| J26DS313_R0019 | MIXED | si | 398 | 0.998 | 0.995 | +0.003 |
| J26DS313_R0017 \* | LATIN | si | 262 | 1.000 | 0.996 | +0.004 |
| J26DS313_R0015 | MIXED | si | 284 | 0.997 | 0.997 | +0.000 |
| J26DS313_R0021 | MIXED | si | 362 | 0.997 | 0.997 | +0.000 |
| J26DS313_R0005 | MIXED | si | 83 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0006 | SINHALA | si | 108 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0009 | SINHALA | si | 72 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0013 | MIXED | si | 191 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0020 | MIXED | si | 349 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0023 | LATIN | si | 215 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0026 | MIXED | si | 267 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0008 \* | MIXED | si | 22 | 1.864 | 1.864 | +0.000 |

\* excluded from C1 modelling per the Section D data-quality audit (R0017 mislabeled tokens, R0008 malformed token file). Their transcripts are still ASR-scorable and are shown for completeness.

## Ablation coverage — what is missing

The plan specified a two-axis ablation (model size × language forcing).
Only the **language-forcing axis** was run. The `medium` weights are not in
this host's HuggingFace cache and cannot be fetched: the machine has no
working outbound HTTPS route, and HuggingFace requests hang in `SYN-SENT`
rather than failing. The size axis therefore needs either a networked
machine or the weights copied in manually. See `docs/RUNNING_NOTES.md`.

Throughput on the RTX 5050 (8GB, float16), from `c1_ablation_runs.json`,
over 3070s of audio — useful as an early input to Task 7:

| config | wall clock | real-time factor |
|---|---|---|
| `small_auto` | 699s | 0.228 |
| `small_si` | 778s | 0.253 |
| `small_en` | 161s | 0.052 |

The forced-English arm is roughly 4x faster in wall-clock terms despite
producing *more* text, so the speed difference reflects decoding dynamics on
mismatched audio, not less work done.

## Motivating fine-tuning as future work

Removing the script artifact moves normalized WER by well under one point,
and no language-forcing setting brings it below ~0.98. The failure is
therefore not a scoring problem and not a decoding-parameter problem:
off-the-shelf Whisper-small has essentially no usable competence on this
audio, and a single language code cannot describe input that changes
language mid-utterance. The ablation's value is in showing how little
headroom decoding settings offer, which is the argument for domain
fine-tuning on a larger corpus (the TAF's 10–15hr target) rather than
further tuning of an off-the-shelf model.

The error-composition result sharpens what fine-tuning has to fix. The
dominant failure in Sinhala mode is *abstention*, not confusion — the model
emits a quarter of the expected words. That is a different target from
reducing substitutions, and it suggests the near-term gain may come from
segmentation and decoding behaviour (VAD settings, forced decoding, chunk
length) as much as from acoustic modelling. Worth testing before assuming
more data alone resolves it.

Fine-tuning stays scoped as post-proposal work, contingent on corpus growth.
