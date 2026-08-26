# Task 3 — SI/EN/OTHER classifier: heuristic vs fastText hybrid

Produced by `scripts/train_lid_fasttext.py`. 5-fold cross-validation, folded by
**recording** so no conversation contributes tokens to both train and test
(verified by an assertion in the script). 5,545 gold tokens across 25
recordings, after excluding R0017 (all tokens mislabeled EN) and R0008
(malformed token file) per the Section D audit.

## Result

| method | scope | accuracy | macro-F1 | SI F1 | EN F1 | OTHER F1 |
|---|---|---|---|---|---|---|
| Method A — heuristic | all tokens | 0.8994 ± 0.0319 | 0.5985 | 0.9251 | 0.8705 | 0.0000 |
| **Hybrid — rules + fastText** | all tokens | **0.9318 ± 0.0378** | 0.6231 | 0.9514 | 0.9180 | 0.0000 |
| heuristic | Latin only | 0.8442 ± 0.0500 | 0.5438 | 0.7465 | 0.8848 | 0.0000 |
| **Method B — fastText** | Latin only | **0.9146 ± 0.0510** | 0.5971 | 0.8524 | 0.9388 | 0.0000 |

The gain concentrates where it should: Latin-script SI F1 moves from **0.7465
to 0.8524**, because that is where the heuristic was weakest.

### The error bars overlap; the comparison is still sound

Mean ± std across folds understates the evidence, because both methods are
scored on the *same* folds and fold difficulty varies widely (per-fold
heuristic accuracy spans 0.844–0.924). The paired comparison is the right one:

| fold | heuristic | hybrid | delta |
|---|---|---|---|
| 0 | 0.9165 | 0.9420 | +0.0255 |
| 1 | 0.9064 | 0.9576 | +0.0512 |
| 2 | 0.9241 | 0.9621 | +0.0380 |
| 3 | 0.9062 | 0.9363 | +0.0301 |
| 4 | 0.8440 | 0.8675 | +0.0235 |

**The hybrid wins on 5 of 5 folds**, with a minimum gain of +0.0235. The
improvement is consistent in direction on every split, which overlapping error
bars would otherwise obscure.

## Why fastText only sees Latin-script tokens

Routing tokens by script first is not an optimisation, it is what makes the
comparison meaningful. Heuristic accuracy by script:

| script | accuracy | errors |
|---|---|---|
| Sinhala | 97.7% | 60 |
| Latin | 83.9% | 430 |
| numeric | 77.0% | 59 |

Sinhala and Tamil script are resolved by Unicode range with no ambiguity, so a
classifier there would only be learning to imitate an already-exact rule. All
the available headroom is in the 430 Latin-script errors:

| confusion | count | examples |
|---|---|---|
| EN → SI | 273 | `Sampath`, `NIC`, `Galle`, `Dehiwala`, `Nawaloka`, `20th` |
| SI → EN | 142 | `me`, `da`, `ah`, `one`, `Aa` |
| OTHER → SI/EN | 15 | see below |

63% are proper nouns and acronyms that `wordfreq` does not know, so the
heuristic falls through to its romanized-Sinhala default. 33% are the reverse
— romanized Sinhala colliding with genuine English words, the `mama` problem.
Capitalisation is the strongest available cue for the first group, so the
fastText input carries explicit case markers alongside the lower-cased
surface; lower-casing before n-gramming is necessary because otherwise `Galle`
and `galle` share no subwords at all, which this little data cannot afford.

## `OTHER` is not a learnable class here, and not for the reason the plan assumed

OTHER F1 is **0.0000 for every method tried, including the heuristic**. That is
a property of the labels.

The plan treats OTHER as Tamil and credits the Unicode Tamil rule with ~100%
precision. Measured against the corpus:

- **There are zero Tamil-script tokens.** The Tamil rule never fires on real
  data.
- Of 78 OTHER tokens after exclusions, **58 are purely numeric** — the
  annotation drift already documented in Section D.3, not a language.
- Of the remaining 20, most are identifiers and English ordinals:
  `8812304567V.` (NIC), `MASIT2024/0892.`, `LN2024NG00445`,
  `hasitha.94@gmail.com.`, `21st,`, `1st,`, `14th.`, `No.`, `1%`.
- The genuinely Tamil items are written in **Sinhala script**
  (`පෙරිය` *periya*, `ඉල්ලයි` *illai*) or romanized (`puriyam.`, `sari`) —
  never in Tamil script.

That leaves 15 Latin-script OTHER examples spread across an incoherent set of
concepts. No classifier learns a category that is defined by four unrelated
things and has 15 examples. Three-way training is still run so the output
stays schema-compatible, and OTHER is reported at 0.00 rather than quietly
dropped.

**This directly changes Task 5.** Its stated plan — keep the Unicode rule for
native-script Tamil, add romanized-Tamil detection — rests on a premise the
data contradicts. There is no native-script Tamil to keep, and the Tamil that
exists is hiding inside Sinhala script where the `unicode_sinhala` rule
confidently mislabels it SI. Task 5 should be rescoped around that before any
detector is built. Two prerequisites are worth settling first: whether OTHER
means "Tamil" or "not-SI-and-not-EN" (the annotations currently use it for
both), and whether the 58 numeric OTHER tokens should be relabelled under the
Section D.3 rule.

## Model size: 37× smaller for free

fastText defaults to 2,000,000 hash buckets, sized for web-scale corpora. With
2,674 training tokens that yields a 400MB model — 50MB quantized — to store a
few thousand distinct character n-grams. Sweeping the bucket count:

| buckets | quantized size | CV accuracy (Latin) |
|---|---|---|
| 2,000,000 | 50.09 MB | 0.9135 |
| 200,000 | 5.09 MB | 0.9179 |
| **50,000** | **1.34 MB** | **0.9161** |
| 20,000 | 0.59 MB | 0.9062 |
| 5,000 | 0.22 MB | 0.9093 |

Accuracy varies by less than half a standard deviation (±0.05) across a 400×
range, so the capacity is simply unused. 50,000 buckets is the default: a 37×
size reduction against stock fastText at no measured cost. The shipped model
is **1.3 MB**, which is a useful early data point for Task 7's quantization
work and comfortably inside the offline deployment budget.

## Scope of the claim

- Cross-validation numbers are the honest ones. The `lid_hybrid.py` demo runs
  the final model over tokens that were in its training set and is
  illustrative only.
- Fold sizes are uneven by token count (390–638 Latin tokens) because folding
  is by recording and balanced on English ratio, not token count. With 25
  recordings that is unavoidable; it is why per-fold variance is reported.
- 0.93 accuracy is on **gold tokens**. End-to-end on ASR output it would be far
  worse and largely meaningless — the token stream analysis shows 80% of
  predicted tokens get labelled by the Unicode rule reading Whisper's chosen
  script rather than the spoken language.

## XLM-R / mBERT as future work

Unchanged from the plan, and the measurements support it. The usable signal is
2,674 Latin-script tokens across two effective classes. A transformer
fine-tuned at that scale would overfit and produce numbers that do not
generalise, while fastText trains in seconds, quantizes to 1.3MB and needs no
GPU at inference. Revisit once the corpus approaches the TAF's 10–15hr target.

Reproduce:

```bash
python3 scripts/train_lid_fasttext.py          # 5-fold CV + final model
python3 scripts/lid_hybrid.py                  # predictor demo
```
