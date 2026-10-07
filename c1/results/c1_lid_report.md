# Task 3 — SI/EN/OTHER classifier: heuristic vs fastText hybrid

Produced by `scripts/train_lid_fasttext.py`. 5-fold cross-validation, folded by
**recording** so no conversation contributes tokens to both train and test
(verified by an assertion in the script). **25,582 gold tokens across 66
recordings** (data as of 2026-10-05: R0002–R0075 and R0100–R0105), after
excluding 12 recordings whose labels are unusable — see
[Data used](#data-used-and-what-was-excluded) below.

## Result

| method | scope | accuracy | macro-F1 | SI F1 | EN F1 | OTHER F1 |
|---|---|---|---|---|---|---|
| Method A — heuristic | all tokens | 0.8967 ± 0.0080 | 0.5974 | 0.9518 | 0.8405 | 0.0000 |
| **Hybrid — rules + fastText** | all tokens | **0.9445 ± 0.0103** | 0.6498 | 0.9879 | 0.9202 | 0.0414 |
| heuristic | excl. numerals | 0.9332 ± 0.0106 | 0.6149 | 0.9521 | 0.8925 | 0.0000 |
| **Hybrid** | excl. numerals | **0.9837 ± 0.0085** | 0.8590 | 0.9883 | 0.9762 | 0.6126 |
| heuristic | Latin only | 0.8281 ± 0.0230 | 0.4715 | 0.5183 | 0.8963 | 0.0000 |
| **Method B — fastText** | Latin only | **0.9645 ± 0.0173** | 0.8226 | 0.8488 | 0.9799 | 0.6389 |

The gain concentrates where it should: Latin-script SI F1 moves from **0.5183
to 0.8488**, because romanized Sinhala is where the heuristic is weakest.

**Read the "excl. numerals" rows as the language-ID result.** The all-token
rows are dragged down by numerals, whose gold label depends on which annotator
labelled the recording rather than on anything in the token (next section).
With numerals set aside the hybrid is right on **98.4%** of tokens.

The OTHER F1 of ~0.6 is **not** a Tamil-detection capability: the Latin-script
OTHER examples it learns from are mostly slashed numbers and percentages
(`48/2`, `45%`), which carry an obvious orthographic signal. Tamil is Task 5's
job (`c1_otherlang_report.md`).

### How the result has moved as data arrived

| | first run | 2026-10-04 | **2026-10-05** |
|---|---|---|---|
| recordings / tokens | 25 / 5,545 | 48 / 14,010 | **66 / 25,582** |
| hybrid, all tokens | 0.9318 ± 0.0378 | 0.9232 ± 0.0218 | **0.9445 ± 0.0103** |
| hybrid, excl. numerals | — | 0.9735 ± 0.0143 | **0.9837 ± 0.0085** |
| fastText, Latin only | 0.9146 ± 0.0510 | 0.9535 ± 0.0270 | **0.9645 ± 0.0173** |
| Latin-script training tokens | 2,674 | 5,816 | 9,037 |

Every added batch has raised accuracy and shrunk the fold-to-fold spread — the
standard deviation is now a quarter of the first run's. The classifier is
still data-limited, which argues for more clean annotation over a bigger model.

### The paired comparison

Both methods are scored on the *same* folds, so the paired view is the right
one:

| fold | heuristic | hybrid | delta |
|---|---|---|---|
| 0 | 0.8850 | 0.9442 | +0.0592 |
| 1 | 0.8931 | 0.9356 | +0.0425 |
| 2 | 0.9061 | 0.9486 | +0.0425 |
| 3 | 0.8992 | 0.9596 | +0.0604 |
| 4 | 0.8999 | 0.9346 | +0.0347 |

**The hybrid wins on 5 of 5 folds**, mean +0.0479, minimum +0.0347.

## Numerals: still two conventions

| recordings | numerals tagged EN | OTHER | SI |
|---|---|---|---|
| R0002–R0028 | 197 | 58 | 1 |
| R0029–R0049 | 0 | 715 | 0 |
| R0050–R0061 | 88 | 4 | 10 |
| R0068–R0075 | 0 | 242 | 0 |
| R0100–R0105 | — | — | — (numbers written as words) |

Every recording added since 2026-10-04 tags digits **OTHER**, so OTHER is now
the majority convention (1,019 of 1,315 numerals), but the older recordings
still tag them EN. The rule-based tagger says EN, so heuristic accuracy on
numerals is **22%** — a measure of which convention a recording follows, not a
model failure. Settling one rule in `DATA_CONTRACT.md` and relabelling the
minority (about 300 tokens) would remove this entirely.

## Data used, and what was excluded

`scripts/audit_gold.py` checks every gold file. Twelve recordings are
excluded, with reasons kept in `scripts/corpus.py`:

| kind | recordings | evidence |
|---|---|---|
| malformed file | R0008 | junk timestamps |
| bulk mislabel | R0017, R0054, R0055, R0056, R0059 | every token tagged EN despite code-mixed speech |
| bulk mislabel | R0058 | 369/373 tokens tagged SI, incl. plain English |
| partial mislabel | R0051, R0062, R0063, R0064, R0065 | 14–51% of unambiguous Sinhala function words (`eka`, `mata`, `oyata`) tagged EN |

R0052 and R0053 were re-annotated on 2026-10-04; R0053 had been excluded (bulk
EN, near-duplicate of R0052) and now passes the audit, so it is back in.

**Sensitivity — keeping the five partially-mislabelled recordings in**
(`SORA_C1_KEEP_PARTIAL=1`, saved as `c1_lid_eval_with_partial.json`):

| | excluded (reported) | kept |
|---|---|---|
| hybrid, all tokens | 0.9445 ± 0.0103 | 0.9223 ± 0.0212 |
| hybrid, excl. numerals | 0.9837 ± 0.0085 | 0.9581 ± 0.0245 |
| fastText, Latin only | 0.9645 ± 0.0173 | 0.9078 ± 0.0438 |

Keeping them costs two to six points and more than doubles the spread. Draft
corrections for all 11 label-error recordings are in `predictions/pretag/`
(`scripts/pretag_relabel.py`, summary in `c1_pretag_summary.md`).

## Why fastText only sees Latin-script tokens

Heuristic accuracy by script:

| script | tokens | accuracy | errors |
|---|---|---|---|
| Sinhala | 15,230 | 99.6% | 60 |
| Latin | 9,037 | 82.8% | 1,558 |
| numeric | 1,315 | 21.7% | 1,030 (convention, see above) |

Sinhala and Tamil script are resolved by Unicode range with no ambiguity, so a
classifier there would only be learning to imitate an already-exact rule. The
real headroom is in the Latin-script errors:

| confusion | count | examples |
|---|---|---|
| EN → SI | 1,339 | `NIC`, `madam`, `Colombo`, `verify`, `SSCL`, `Galle` |
| SI → EN | 179 | `mama`, `me`, `Ah`, `na`, `number`, `da` |
| OTHER → SI/EN | 40 | slashed numbers, percentages, identifiers, a few Tamil words |

86% are English words — mostly proper nouns and acronyms — that `wordfreq`
does not know, so the heuristic falls through to its romanized-Sinhala
default. The rest are romanized Sinhala colliding with English words (`mama`
is in `wordfreq`). Capitalisation is the strongest available cue for the
first group, so the fastText input carries explicit case markers alongside
the lower-cased surface.

## `OTHER` is still not a learnable class

- **Zero Tamil-script tokens** in 25,582. The Unicode Tamil rule never fires.
- Of 1,064 OTHER tokens, **1,041 are numerals** (98%) and 11 are identifiers
  or ordinals (`8812304567V.`, `LN2024NG00445`, `21st,`, `No.`).
- **12 OTHER tokens are Tamil**, and gold labels at least 4 more SI
  (`romba`, `nandri`, `vanakkam` in Sinhala script). Tamil appears in Sinhala
  script, romanized, and once in Kannada/Malayalam codepoints — never in
  Tamil script.

## Model size

fastText defaults to 2,000,000 hash buckets, sized for web-scale corpora.
Sweeping the bucket count on the first batch:

| buckets | quantized size | CV accuracy (Latin) |
|---|---|---|
| 2,000,000 | 50.09 MB | 0.9135 |
| 200,000 | 5.09 MB | 0.9179 |
| **50,000** | **1.34 MB** | **0.9161** |
| 20,000 | 0.59 MB | 0.9062 |
| 5,000 | 0.22 MB | 0.9093 |

Accuracy varied by less than half a standard deviation across a 400× range, so
50,000 buckets is the default. The retrained model is **1.3 MB** on 9,037
Latin tokens; quantization costs no measurable accuracy (Task 7,
`c1_latency_report.md`). The bucket sweep was not re-run on the larger corpus.

## Scope of the claim

- Cross-validation numbers are the honest ones. The `lid_hybrid.py` demo runs
  the final model over tokens that were in its training set and is
  illustrative only.
- Fold sizes are uneven by token count (1,288–2,233 Latin tokens) because
  folding is by recording and balanced on English ratio.
- Many recordings come from shared scenario templates (e.g. R0012/R0013 are
  two "Senaratne and Associates" conversations). Folding by recording keeps a
  conversation out of its own test set but not its template's vocabulary, so
  these numbers are mildly optimistic for genuinely new topics.
- All figures are on **gold tokens**. End-to-end on ASR output they would be
  far worse — see `c1_token_stream_notes.md`.

## XLM-R / mBERT as future work

Still future work. 9,037 Latin-script tokens is better, and the learning curve
above is still rising, but fastText trains in seconds, quantizes to 1.3 MB and
needs no GPU. A transformer becomes worth testing once the corpus nears the
TAF's 10–15 hr target and the numeral convention is fixed.

Reproduce:

```bash
python scripts/audit_gold.py                         # which recordings are usable
python scripts/train_lid_fasttext.py                 # 5-fold CV + final model
SORA_C1_KEEP_PARTIAL=1 python scripts/train_lid_fasttext.py   # sensitivity run
python scripts/lid_hybrid.py                         # predictor demo
```
