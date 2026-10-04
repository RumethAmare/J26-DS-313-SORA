# Task 5 — Tamil (`OTHER`) flagging

Produced by `scripts/otherlang_flag.py`; machine-readable numbers in
`c1_otherlang_eval.json`; curated ground truth in `data/tamil_ground_truth.json`.
Scope: 14,010 gold tokens across the 48 usable recordings (original batch plus
the October 2026 batch; exclusions in `scripts/corpus.py`).

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
across all 14,010 tokens. The rule being "kept as-is" never fires; its ~100%
precision is vacuous.

**2. Tamil in Sinhala script is the most common form.** Of 14 genuinely-Tamil
tokens, **8 are written in Sinhala script**, 5 are romanized and 1 uses
Kannada/Malayalam codepoints. Five of the eight are one sentence in R0015:

> `ඉදු ඔරුව පෙරිය ප්‍රචන ඉල්ලයි`
> → *idhu oru periya pirachanai illai* → "this is not a big problem"

The new batch adds three more — `වනක්කම්` (*vanakkam*), `රොම්බ` (*romba*,
"very") and `නන්ද්‍රි` (*nandri*, "thanks") in R0049. The plan does not
consider this case at all.

**3. Gold `OTHER` is not a usable evaluation target, and the new batch made it
worse.** Of 743 `OTHER` tokens, **722 are numerals** (one annotator group tags
every digit OTHER — see `c1_lid_report.md`) and **11 are identifiers or
ordinals** (`8812304567V.`, `LN2024NG00445`, `21st,`, `No.`). Only **10** are
Tamil — 1.3% of the class. And gold misses Tamil in the other direction too:
4 of the 14 Tamil tokens are labelled SI. Ground truth is therefore curated
manually.

## Design: romanize first, detect second

Because the Tamil here lives in several scripts, the detector reuses
`script_normalize.normalize_scripted()` — built in Task 1 for WER scoring — to
map Sinhala script into the same Latin skeleton romanized text already
occupies (`පෙරිය` → `periya`, `වනක්කම්` → `vanakam`). **One** detector in
romanized space then covers every case, rather than one model per script.

Three signals: foreign-script anomaly, a romanized-Tamil gazetteer, and a
discriminative character-trigram log-likelihood ratio (Tamil vs the better of
SI/EN). The plan's outlier rule is implemented too, for comparison.

## Results

Evaluation uses **raw counts, not F1** — with 14 positives an F1 moves ~0.07
per token and would communicate false precision.

| threshold pct | LOO recall | full-lexicon | false positives | FP rate |
|---|---|---|---|---|
| 99.9 | 2/14 | 9/14 | 9 | 0.06% |
| **99.0** | **3/14** | **10/14** | **37** | **0.26%** |
| 98.0 | 3/14 | 10/14 | 40 | 0.29% |
| 95.0 | 5/14 | 10/14 | 108 | 0.77% |
| 90.0 | 6/14 | 10/14 | 341 | 2.43% |

*LOO* removes the target word from the lexicon — a pessimistic bound.
*full-lexicon* keeps it, which is what a real Tamil dictionary would do, since
these are all common Tamil words.

**At the 99th percentile the detector finds 10 of 14 Tamil tokens while
flagging 0.26% of the corpus** — on the first batch alone it managed 5/9 at
0.34%. Most of the recall gain is that the five new positives are common
greetings and courtesies already in the lexicon, not a better detector; the
false-positive *rate* is slightly lower at every threshold, though the raw
count is higher on a corpus 2.5× the size.

The LOO column is harsher than it looks: `vanakkam` accounts for 3 of the 14
tokens, so removing it from the lexicon loses all three at once. LOO is a
bound on unseen *words*, and these are mostly a few common greetings.

**The plan's outlier-only rule recovers 1/14 with 1,571 false positives (11.2%
of the corpus).** Framing the problem as "Tamil vs the others" rather than
"unlike SI and EN" is worth about two orders of magnitude in false-positive
rate — the outlier signal fires on any unusual token (`you`, `much`, `zero`),
and most unusual tokens are not Tamil.

### Two bugs found while building this detector

Both are the same class of error — a smoothing artifact masquerading as
signal.

*Unseen short tokens scored as maximally Tamil.* Digits and single letters
have n-grams present in **no** model, so every model returns its add-1 floor —
and the Tamil model, trained on the fewest words, has the smallest
`total + vocab` and therefore the **highest** floor. ~380 of an initial 426
false positives were digits. Fixed by restricting scoring to alphabetic tokens
of length ≥ 3, and by requiring **positive** Tamil evidence (a minimum share of
the token's n-grams actually observed in Tamil) rather than a high margin
alone.

*Thresholds were calibrated over tokens the detector never scores*, which
pushed the 99th percentile to an unreachable value. Now calibrated over
candidates only.

## Annotation inconsistency this exposed

The same Tamil word is labelled differently depending on script and annotator:

| recording | token | script | gold label |
|---|---|---|---|
| R0013 | `සරි` | Sinhala | **SI** |
| R0016 | `sari`, `sari.` | Latin | **OTHER** |
| R0050 | `wanakkam` | Latin | **OTHER** |
| R0052 | `Vanakkam.` | Latin | **SI** |
| R0049 | `වනක්කම්` | Sinhala | **SI** |

The gazetteer prunes entries that collide with corpus SI/EN words, and 11 were
pruned this run (`sari`, `nala`, `nama`, `nan`, `ne`, …). `sari` is pruned
because of the R0013 token, which is why full-lexicon recall caps at 10/14:
the detector is penalised by a labelling error.

**More serious for downstream consumers:** all 8 Sinhala-script Tamil tokens
are labelled `SI` at **0.99 confidence** by the `unicode_sinhala` rule.
Confidently wrong is the worst failure mode for a signal C2/C3/C4 are meant to
trust. The rule reports script, and script does not imply language.

## What limits this, honestly

**Lexicon coverage, not model capacity.** Every full-lexicon miss is a
vocabulary gap:

| token | normalized | lexicon has | gap |
|---|---|---|---|
| `ඔරුව` | `oruva` | `oru` | inflected form absent |
| `ඉල්ලයි.` | `ilayi` | `ilai` (from *illai*) | spelling variant |
| `sari`, `sari.` | `sari` | pruned | annotation collision |

Tamil in excluded recordings shows the same pattern — `nalla` (pruned as a
collision with `nala`) and `visayam` (absent) in R0065. A real romanized-Tamil
word list would help far more than a larger model;
`data/tamil_romanized_lexicon.txt` is hand-written and small.

**Fourteen positives is still not enough to validate anything.** These numbers
indicate direction, not performance, and five of the fourteen are two words
(`vanakkam`, `sari`).

**The detector cannot run end-to-end.** Whisper never emits romanized Tamil and
the new recordings have no audio in the dataset repo yet, so this is scoped to
gold/human-transcribed text.

## Recommended follow-ups

1. **Agree one numeral rule** and relabel. 722 of 743 `OTHER` tokens are
   numerals; until that is fixed every metric on the class is contaminated.
2. **Tag Tamil as OTHER consistently** — `vanakkam`, `romba`, `nandri`,
   `sari` are currently SI in some recordings and OTHER in others.
3. **Expand the Tamil lexicon**, including inflected forms and Sinhala-script
   Tamil spellings.
4. **Lower `unicode_sinhala` confidence** when a token also scores as Tamil, so
   the 0.99-on-wrong-label case stops propagating downstream.
