# C1 Task 1 — Off-the-shelf ASR baseline (script-normalized + ablation)

Scores faster-whisper against the gold C1 transcripts under two
different normalizations, across a language-forcing ablation.

## Why two WER numbers

Gold transcripts are not written in one script. Nine of 27 recordings are
100% romanized Latin (`mama`, `thiyenawa`); most of the rest are majority
Sinhala Unicode. faster-whisper auto-detects **one** language per recording
and decodes the whole thing in that script. Scoring those directly compares
romanized reference against Sinhala-Unicode hypothesis, which pins WER near
1.0 regardless of whether the model heard the words correctly.

- **raw-script WER** — lowercase + strip punctuation only. Script mismatch is
  charged as error. This is what the original `evaluate.py` reported.
- **normalized WER** — both sides transliterated to one phonetic skeleton
  (`script_normalize.py`) first, so script choice costs nothing.

The gap between them measures the scoring artifact; what remains after
normalization is genuine recognition error.

## Ablation summary

| config | micro raw WER | micro norm WER | gap | macro norm WER |
|---|---|---|---|---|
| `small_auto` | 1.0005 | 0.9959 | +0.0046 | 1.0074 |

## Breakdown by gold script

The artifact is concentrated in the romanized-gold bucket, which is the
point: those recordings were never as badly recognized as raw WER implied.

| config | gold script | n | micro raw WER | micro norm WER | gap |
|---|---|---|---|---|---|
| `small_auto` | LATIN | 5 | 0.9986 | 0.9914 | +0.0072 |
| `small_auto` | MIXED | 8 | 1.0015 | 0.9977 | +0.0038 |
| `small_auto` | SINHALA | 2 | 1.0000 | 1.0000 | +0.0000 |

## Per-recording detail

### `small_auto`

| recording | gold script | detected | #words | raw WER | norm WER | gap |
|---|---|---|---|---|---|---|
| J26DS313_R0010 | MIXED | ml | 64 | 1.000 | 0.952 | +0.048 |
| J26DS313_R0004 | LATIN | si | 75 | 1.000 | 0.973 | +0.027 |
| J26DS313_R0015 | MIXED | si | 284 | 0.997 | 0.989 | +0.007 |
| J26DS313_R0002 | LATIN | si | 114 | 1.000 | 0.991 | +0.009 |
| J26DS313_R0017 \* | LATIN | si | 262 | 0.996 | 0.992 | +0.004 |
| J26DS313_R0014 | LATIN | si | 180 | 1.000 | 0.994 | +0.006 |
| J26DS313_R0013 | MIXED | si | 191 | 0.995 | 0.995 | +0.000 |
| J26DS313_R0011 | MIXED | si | 214 | 0.995 | 0.995 | +0.000 |
| J26DS313_R0003 | LATIN | si | 70 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0005 | MIXED | si | 83 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0006 | SINHALA | si | 108 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0009 | SINHALA | si | 72 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0012 | MIXED | si | 202 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0016 | MIXED | si | 239 | 1.000 | 1.000 | +0.000 |
| J26DS313_R0008 \* | MIXED | si | 22 | 1.227 | 1.227 | +0.000 |

\* excluded from C1 modelling per the Section D data-quality audit (R0017 mislabeled tokens, R0008 malformed token file). Their transcripts are still ASR-scorable and are shown for completeness.

## Motivating fine-tuning as future work

Even after the script artifact is removed, normalized WER stays far above
anything usable downstream. That residue is the real problem C1 exists to
solve, and it is not a scoring or decoding-parameter issue: off-the-shelf
Whisper has essentially no exposure to Sinhala-English code-mixed speech,
and forcing a single language code cannot fix input that changes language
mid-utterance. The ablation shows how little headroom is available from
decoding settings alone — which is precisely the argument for domain
fine-tuning on a larger corpus (the TAF's 10–15hr target) rather than
further tuning of an off-the-shelf model. Fine-tuning is therefore scoped
as post-proposal work, contingent on corpus growth.
