# Phase 0 — Baseline decisions

Decoding-mode and model-size comparison for C1, measured on the real corpus. This file and its companion `baseline_decisions.json` are the single source of truth every later phase reads from (C1_FINAL_IMPLEMENTATION.md §8).

## Decision

- **Decode mode:** `en`
- **Model size:** `medium`
- **Best config:** `medium/en` — eval-split folded WER **0.841**, raw WER **0.847**
- **Criterion:** eval-split micro WER under script-folded normalization. Script-folded scoring measures recognition quality independently of which script the corpus uses, because fine-tuning can change the output script but cannot recover words the model never heard.

## Harness check

`small`/`auto` under the legacy normalizer scores 0.9525 against the prior baseline's 0.9372 (delta 0.0153) — reproduced. Source: `SORA_Dataset/results/c1_report.md`.

## All configs

| config | raw WER | folded WER | raw CER | folded CER | eval folded WER | RTF |
|---|---|---|---|---|---|---|
| `medium/en` | 0.9000 | 0.8938 | 0.6507 | 0.5881 | 0.8410 | 0.0562 |
| `small/en` | 0.9852 | 0.9764 | 0.7099 | 0.6318 | 0.8548 | 0.0557 |
| `large-v3/en` | 0.9201 | 0.9147 | 0.6559 | 0.5836 | 0.8675 | 0.1728 |
| `large-v3/auto` | 0.9712 | 0.9648 | 0.8349 | 0.7400 | 0.9619 | 0.9719 |
| `large-v3/si` | 0.9722 | 0.9651 | 0.8258 | 0.7308 | 0.9735 | 0.9387 |
| `medium/auto` | 1.0009 | 0.9977 | 0.8913 | 0.8040 | 0.9934 | 0.5971 |
| `medium/si` | 0.9966 | 0.9944 | 0.8814 | 0.7897 | 0.9934 | 0.5761 |
| `small/si` | 0.9976 | 0.9941 | 0.8996 | 0.8180 | 0.9934 | 0.2528 |
| `small/auto` | 0.9976 | 0.9946 | 0.9021 | 0.8192 | 0.9939 | 0.2483 |

## Error composition (folded, whole corpus)

Where the WER actually comes from. Deletions dominating means the model emitted nothing; substitutions dominating means it emitted the wrong words. The two call for opposite fixes.

| config | hit | sub | del | ins |
|---|---|---|---|---|
| `medium/en` | 0.237 | 0.638 | 0.125 | 0.131 |
| `small/en` | 0.188 | 0.674 | 0.138 | 0.164 |
| `large-v3/en` | 0.249 | 0.700 | 0.052 | 0.163 |
| `large-v3/auto` | 0.045 | 0.347 | 0.608 | 0.010 |
| `large-v3/si` | 0.051 | 0.374 | 0.575 | 0.016 |
| `medium/auto` | 0.006 | 0.287 | 0.706 | 0.004 |
| `medium/si` | 0.007 | 0.285 | 0.708 | 0.001 |
| `small/si` | 0.007 | 0.230 | 0.764 | 0.001 |
| `small/auto` | 0.005 | 0.209 | 0.786 | 0.000 |

## Script-mismatch cost (raw WER - folded WER)

How many WER points each config loses purely to script mismatch rather than to misrecognition.

| config | mixed | native | romanized |
|---|---|---|---|
| `medium/en` | 0.004 | 0.008 | 0.000 |
| `small/en` | 0.008 | 0.010 | 0.002 |
| `large-v3/en` | 0.004 | 0.007 | 0.002 |
| `large-v3/auto` | 0.007 | 0.002 | 0.023 |
| `large-v3/si` | 0.026 | 0.004 | 0.014 |
| `medium/auto` | 0.000 | 0.002 | 0.006 |
| `medium/si` | 0.000 | 0.002 | 0.002 |
| `small/si` | 0.007 | 0.003 | 0.006 |
| `small/auto` | 0.007 | 0.002 | 0.005 |

## Caveats

- Scored on 52 min of audio; C1 §6.2 assumes ~1 h eval plus 15-20 h train. Treat every figure as provisional until the corpus grows.
- Folded WER is a floor, not a headline: script_fold is deliberately many-to-one and merges genuinely distinct words.
- Whisper segmentation does not align to gold utt_ids, so scoring is recording-level concatenation, which absorbs segmentation error into WER.

Eval split (7 recordings, locked in `data/manifest.csv`): J26DS313_R0010, J26DS313_R0012, J26DS313_R0014, J26DS313_R0019, J26DS313_R0020, J26DS313_R0023, J26DS313_R0025
