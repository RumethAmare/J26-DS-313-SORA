# Whisper fine-tuning — epoch-by-epoch report

Produced by `scripts/report_finetune.py` from `results/c1_asr_finetune_<model>.json`. Training: `scripts/finetune_whisper.py`.

## Setup

- **Method:** LoRA (rank 32) on attention and feed-forward layers; base weights frozen in bf16. Same for all three models.
- **Language token:** `en`. Learning rate 0.001, linear warm-up then decay, effective batch 16 clips.
- **Data**, split by recording: train 47 recordings / 479 clips / 1.59 h of speech; validation 6 / 63 / 0.23 h; test 10 / 101 / 0.40 h.
- **Per epoch:** one pass over the training clips, then validation loss (the model's average error at predicting the validation transcripts, token by token; lower is better). The adapter is saved whenever validation loss improves; training stops after 2 epochs without improvement (max 10).
- **Word error rate is not measured per epoch.** It was measured once per model, on the saved best-epoch adapter, against the test set — see the final section.

## Validation loss by epoch, all models

| epoch | `small` | `medium` | `large-v3` |
|---|---|---|---|
| 0 | 2.3954 | 2.3794 | 2.1337 |
| 1 | 1.3972 | 1.2608 | 1.1452 |
| 2 | 1.1254 | 1.1166 | 1.0786 |
| 3 | 1.0772 | **1.0510** ←best | **1.0421** ←best |
| 4 | **1.0652** ←best | 1.1080 | 1.0471 |
| 5 | 1.0792 | 1.1288 | 1.1038 |
| 6 | 1.1038 | stopped | stopped |

Epoch 0 is the off-the-shelf model before any training.

## `small`

| epoch | train loss | val loss | change in val loss | val − train gap | epoch time | cumulative | |
|---|---|---|---|---|---|---|---|
| 0 | — | 2.3954 | — | — | — | — | before training |
| 1 | 1.5801 | 1.3972 | -0.9982 | -0.183 | 2.5 min | 2.5 min |  |
| 2 | 0.8051 | 1.1254 | -0.2718 | +0.320 | 2.4 min | 4.9 min |  |
| 3 | 0.5211 | 1.0772 | -0.0482 | +0.556 | 1.6 min | 6.5 min |  |
| 4 | 0.3717 | **1.0652** | -0.0120 | +0.694 | 1.6 min | 8.1 min | **best — saved** |
| 5 | 0.2786 | 1.0792 | +0.0140 | +0.801 | 1.6 min | 9.7 min | val worse |
| 6 | 0.1922 | 1.1038 | +0.0246 | +0.912 | 1.6 min | 11.3 min | val worse |

- Validation loss fell from 2.395 to 1.065 at epoch 4 (56% lower). Epoch 1 alone delivered 75% of that drop.
- After epoch 4, training loss kept falling (0.372 → 0.192) while validation loss rose (1.065 → 1.104): the model began memorising the training clips. Early stopping ended the run at epoch 6 of 10.
- Gap between validation and training loss at the best epoch: 0.69, widening to 0.91 by the last epoch.
- Total training time 11 min; peak GPU memory 4.2 GiB.

## `medium`

| epoch | train loss | val loss | change in val loss | val − train gap | epoch time | cumulative | |
|---|---|---|---|---|---|---|---|
| 0 | — | 2.3794 | — | — | — | — | before training |
| 1 | 1.3873 | 1.2608 | -1.1186 | -0.127 | 5.1 min | 5.1 min |  |
| 2 | 0.6823 | 1.1166 | -0.1442 | +0.434 | 4.2 min | 9.3 min |  |
| 3 | 0.4347 | **1.0510** | -0.0656 | +0.616 | 4.3 min | 13.6 min | **best — saved** |
| 4 | 0.3124 | 1.1080 | +0.0570 | +0.796 | 9.4 min | 23.0 min | val worse |
| 5 | 0.2340 | 1.1288 | +0.0208 | +0.895 | 4.7 min | 27.7 min | val worse |

- Validation loss fell from 2.379 to 1.051 at epoch 3 (56% lower). Epoch 1 alone delivered 84% of that drop.
- After epoch 3, training loss kept falling (0.435 → 0.234) while validation loss rose (1.051 → 1.129): the model began memorising the training clips. Early stopping ended the run at epoch 5 of 10.
- Gap between validation and training loss at the best epoch: 0.62, widening to 0.89 by the last epoch.
- Epoch times varied from 4.2 to 9.4 min (slowest: epoch 4). Every epoch does the same work, so the variation is the laptop (GPU memory pressure from other apps, or background load), not the training.
- Total training time 28 min; peak GPU memory 3.7 GiB.

## `large-v3`

| epoch | train loss | val loss | change in val loss | val − train gap | epoch time | cumulative | |
|---|---|---|---|---|---|---|---|
| 0 | — | 2.1337 | — | — | — | — | before training |
| 1 | 1.2493 | 1.1452 | -0.9885 | -0.104 | 28.0 min | 28.0 min |  |
| 2 | 0.6488 | 1.0786 | -0.0666 | +0.430 | 17.6 min | 45.6 min |  |
| 3 | 0.4482 | **1.0421** | -0.0365 | +0.594 | 66.9 min | 112.5 min | **best — saved** |
| 4 | 0.3442 | 1.0471 | +0.0050 | +0.703 | 22.7 min | 135.2 min | val worse |
| 5 | 0.2599 | 1.1038 | +0.0567 | +0.844 | 64.9 min | 200.1 min | val worse |

- Validation loss fell from 2.134 to 1.042 at epoch 3 (51% lower). Epoch 1 alone delivered 91% of that drop.
- After epoch 3, training loss kept falling (0.448 → 0.260) while validation loss rose (1.042 → 1.104): the model began memorising the training clips. Early stopping ended the run at epoch 5 of 10.
- Gap between validation and training loss at the best epoch: 0.59, widening to 0.84 by the last epoch.
- Epoch times varied from 17.6 to 66.9 min (slowest: epoch 3). Every epoch does the same work, so the variation is the laptop (GPU memory pressure from other apps, or background load), not the training.
- Total training time 200 min; peak GPU memory 4.7 GiB.

## Test-set result of each model's best epoch

10 held-out recordings, 3,490 reference words. WER and character error rate are script-normalised (Sinhala Unicode vs romanized is not penalised). Greedy decoding, same settings before and after.

| model | best epoch | | WER | char. error rate | words recovered | output ÷ reference | extra words |
|---|---|---|---|---|---|---|---|
| `small` | — | off-the-shelf | 2.256 | 1.511 | 13.2% | 1.97× | 4,841 |
| `small` | 4 | fine-tuned | **0.934** | 0.589 | 48.6% | 1.33× | 1,466 |
| `medium` | — | off-the-shelf | 1.113 | 0.821 | 17.3% | 0.90× | 998 |
| `medium` | 3 | fine-tuned | **0.956** | 0.623 | 54.8% | 1.38× | 1,758 |
| `large-v3` | — | off-the-shelf | 1.100 | 0.819 | 18.3% | 0.88× | 987 |
| `large-v3` | 3 | fine-tuned | **0.663** | 0.384 | 53.2% | 1.08× | 682 |

## Reading across the three models

- **All three peaked early** — `small` at epoch 4, `medium` at epoch 3, `large-v3` at epoch 3. With 1.6 h of training speech the models extract most of what the data offers in 3–4 passes, then start memorising it. More training data, not more epochs, is what would move these numbers.
- **Best validation losses are close** (`small` 1.065, `medium` 1.051, `large-v3` 1.042), yet test WER differs much more. Validation loss scores each next token given the correct previous ones; WER scores free-running transcription, where `large-v3` loops far less (682 extra words vs 1,466 for `small` and 1,758 for `medium`). Validation loss is the right signal for choosing an epoch, not for ranking models.
- **Bigger models started lower and fell less steeply.** `large-v3`'s off-the-shelf validation loss was already the lowest (2.13 vs ~2.38), consistent with its stronger multilingual pre-training.

## Limitations

- One run per model with one seed; no variance estimate. Small differences (a few hundredths of validation loss, a few points of words recovered) are within run-to-run noise.
- The validation set is 6 recordings / 63 clips, so the best-epoch choice itself is noisy: `large-v3`'s epochs 3 and 4 differ by 0.005.
- Test WER uses greedy decoding on utterance clips, not faster-whisper on full recordings, so it is not directly comparable with Task 1's figures.
