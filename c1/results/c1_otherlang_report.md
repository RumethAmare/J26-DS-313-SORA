# Task 5 — Tamil (`OTHER`) flagging

Produced by `scripts/otherlang_flag.py`; machine-readable numbers in
`c1_otherlang_eval.json`; curated ground truth in `data/tamil_ground_truth.json`.

`OTHER` denotes **Tamil specifically** (a national language of Sri Lanka
alongside Sinhala), not a generic "not-SI-and-not-EN" bucket. The annotations
do not currently behave that way, which is most of what this task had to
work around.

## Re-scoping: three of the plan's premises don't hold

The plan's Task 5 reads: *"Unicode rule handles native-script Tamil at ~100%
precision — keep as-is. Real gap is romanized Tamil […] flag a Latin-script
token as candidate-OTHER if its likelihood under both is below a percentile
threshold."*

**1. There is no native-script Tamil.** Zero tokens in the Tamil Unicode block
across the whole corpus. The rule being "kept as-is" never fires; its ~100%
precision is vacuous.

**2. Romanized Tamil is the minority case.** Of 9 genuinely-Tamil tokens, only
3 are romanized. **Five are Tamil written in Sinhala script** — a single
complete sentence in R0015:

> `ඉදු ඔරුව පෙරිය ප්‍රචන ඉල්ලයි`
> → *idhu oru periya pirachanai illai* → "this is not a big problem"

The plan does not consider this case, and it is the dominant one.

**3. Gold `OTHER` is not a usable evaluation target.** Of 78 `OTHER` tokens,
**58 are purely numeric** and **11 are identifiers or English ordinals**
(`8812304567V.`, `LN2024NG00445`, `hasitha.94@gmail.com.`, `21st,`, `No.`,
`1%`). Only 9 are Tamil. Scoring "precision/recall against gold OTHER" as the
plan proposes would measure a detector against a class that is **88%
not-Tamil**. Ground truth was curated manually instead.

## Design: romanize first, detect second

Because the Tamil here lives in two scripts, the useful move was to reuse
`script_normalize.normalize_scripted()` — built in Task 1 for WER scoring — to
map Sinhala script into the same Latin skeleton romanized text already
occupies (`පෙරිය` → `periya`, `සරි` → `sari`). **One** detector in romanized
space then covers both cases, rather than two models.

Three signals: foreign-script anomaly, a romanized-Tamil gazetteer, and a
discriminative character-trigram log-likelihood ratio (Tamil vs the better of
SI/EN). The plan's outlier rule is implemented too, for comparison.

## Results

Evaluation uses **raw counts, not F1** — with 9 positives an F1 moves ~0.1 per
token and would communicate false precision.

| threshold pct | LOO recall | full-lexicon | false positives | FP rate |
|---|---|---|---|---|
| 99.9 | 3/9 | 5/9 | 4 | 0.07% |
| 99.0 | 3/9 | 5/9 | 19 | 0.34% |
| 98.0 | 3/9 | 5/9 | 37 | 0.67% |
| **95.0** | **5/9** | **5/9** | **76** | **1.37%** |
| 90.0 | 5/9 | 5/9 | 160 | 2.89% |

*LOO* removes the target word from the lexicon — a pessimistic bound.
*full-lexicon* keeps it, which is what a real Tamil dictionary would do, since
these are all common Tamil words.

**The plan's outlier-only rule recovers 2/9 with 558 false positives (10.06%
of the corpus).** The discriminative approach reaches 5/9 at 1.37%. Framing
the problem as "Tamil vs the others" rather than "unlike SI and EN" is worth
roughly an order of magnitude in false-positive rate — the outlier signal
fires on any unusual token, and most unusual tokens are not Tamil.

### Two bugs found while fixing this detector

Both were mine, and both are the same class of error — a smoothing artifact
masquerading as signal.

*Unseen short tokens scored as maximally Tamil.* Digits and single letters
have n-grams present in **no** model, so every model returns its add-1 floor —
and the Tamil model, trained on the fewest words, has the smallest
`total + vocab` and therefore the **highest** floor. ~380 of an initial 426
false positives were digits. Fixed by restricting scoring to alphabetic tokens
of length ≥ 3, and by requiring **positive** Tamil evidence (a minimum share of
the token's n-grams actually observed in Tamil) rather than a high margin
alone. False positives fell from 7.68% to 1.17%.

*Thresholds were calibrated over tokens the detector never scores*, which
pushed the 99th percentile to an unreachable +1.58. Now calibrated over
candidates only.

## Annotation inconsistency this exposed

The same Tamil word is labelled two different ways depending on the script it
was written in:

| recording | token | script | gold label |
|---|---|---|---|
| R0013 | `සරි` | Sinhala | **SI** |
| R0016 | `sari` | Latin | **OTHER** |
| R0016 | `sari.` | Latin | **OTHER** |

Tamil *சரி* ("ok/right") in both cases. This surfaced mechanically: the
gazetteer prunes entries colliding with corpus SI/EN words, and `sari` was
pruned because of the R0013 token — which is why full-lexicon recall caps at
5/9 rather than 7/9. The detector is being penalised by a labelling error.

**Related, and more serious for downstream consumers:** all five
Sinhala-script Tamil tokens are labelled `SI` at **0.99 confidence** by the
`unicode_sinhala` rule. Confidently wrong is the worst failure mode for a
signal C2/C3/C4 are meant to trust. The rule reports script, and script does
not imply language — the same conflation identified in the token-stream
analysis.

## What limits this, honestly

**Lexicon coverage, not model capacity.** Every leave-one-out miss is a
near-miss in vocabulary:

| token | normalized | lexicon has | gap |
|---|---|---|---|
| `ඔරුව` | `oruva` | `oru` | inflected form absent |
| `ඉල්ලයි.` | `ilayi` | `ilai` (from *illai*) | spelling variant |
| `puriyam.` | `puriyam` | `puriyum` | vowel variant |
| `sari` | `sari` | pruned | annotation collision |

A real romanized-Tamil word list, or a transliterated Tamil frequency list,
would help far more than a larger model. `data/tamil_romanized_lexicon.txt` is
hand-written and small; expanding it is the highest-leverage next step.

**Nine positives is not enough to validate anything.** These numbers indicate
direction, not performance. Any confidence interval on 9 examples spans most
of the possible range.

**The detector cannot run end-to-end.** Whisper decodes into Sinhala script for
25 of 26 recordings and never emits romanized Tamil, so on ASR output this has
nothing to fire on. It is scoped to gold/human-transcribed text, consistent
with the Phase 3 finding that the LID task is near-degenerate on ASR output.

## Recommended follow-ups

1. **Relabel the 58 numeric `OTHER` tokens** under the Section D.3 rule. Until
   then `OTHER` conflates Tamil with phone numbers and every metric on that
   class is contaminated.
2. **Fix the `sari` inconsistency** and audit for other same-word/different-label
   pairs across scripts.
3. **Expand the Tamil lexicon**, including inflected forms and Sinhala-script
   Tamil spellings.
4. **Lower `unicode_sinhala` confidence** when a token also scores as Tamil, so
   the 0.99-on-wrong-label case stops propagating downstream.
