# Task 3 — SI/EN/OTHER classifier: heuristic vs fastText hybrid

Produced by `scripts/train_lid_fasttext.py`. 5-fold cross-validation, folded by
**recording** so no conversation contributes tokens to both train and test
(verified by an assertion in the script). **14,010 gold tokens across 48
recordings**: the original batch (R0002–R0028) plus the October 2026 batch
(R0031–R0061), after excluding 13 recordings whose labels are unusable — see
[Data used](#data-used-and-what-was-excluded) below.

## Result

| method | scope | accuracy | macro-F1 | SI F1 | EN F1 | OTHER F1 |
|---|---|---|---|---|---|---|
| Method A — heuristic | all tokens | 0.8807 ± 0.0208 | 0.5944 | 0.9437 | 0.8397 | 0.0000 |
| **Hybrid — rules + fastText** | all tokens | **0.9232 ± 0.0218** | 0.6328 | 0.9789 | 0.9027 | 0.0168 |
| heuristic | excl. numerals | 0.9276 ± 0.0119 | 0.6140 | 0.9444 | 0.8977 | 0.0000 |
| **Hybrid** | excl. numerals | **0.9735 ± 0.0143** | 0.7514 | 0.9796 | 0.9658 | 0.3089 |
| heuristic | Latin only | 0.8487 ± 0.0271 | 0.5192 | 0.6543 | 0.9033 | 0.0000 |
| **Method B — fastText** | Latin only | **0.9535 ± 0.0270** | 0.7285 | 0.8709 | 0.9715 | 0.3431 |

The gain concentrates where it should: Latin-script SI F1 moves from **0.6543
to 0.8709**, because romanized Sinhala is where the heuristic is weakest.

**Read the "excl. numerals" rows as the language-ID result.** The all-token
rows are dragged down by numerals, whose gold label depends on who annotated
the recording rather than on anything in the token (next section). With
numerals set aside the hybrid is right on 97.4% of tokens.

### Compared with the first run (27 recordings)

| | first run | now |
|---|---|---|
| recordings / tokens | 25 / 5,545 | 48 / 14,010 |
| hybrid, all tokens | 0.9318 ± 0.0378 | 0.9232 ± 0.0218 |
| fastText, Latin only | 0.9146 ± 0.0510 | 0.9535 ± 0.0270 |
| Latin-script training tokens | 2,674 | 5,816 |

More than doubling the Latin-script training data lifted fastText by four
points and **halved the fold-to-fold spread**. The all-token figure dipped
slightly only because the new batch carries ~650 numerals labelled OTHER that
no token-level rule can reproduce.

### The paired comparison

Mean ± std across folds understates the evidence, because both methods are
scored on the *same* folds. The paired view:

| fold | heuristic | hybrid | delta |
|---|---|---|---|
| 0 | 0.8740 | 0.9163 | +0.0423 |
| 1 | 0.8542 | 0.8992 | +0.0450 |
| 2 | 0.9067 | 0.9486 | +0.0419 |
| 3 | 0.8961 | 0.9436 | +0.0475 |
| 4 | 0.8725 | 0.9085 | +0.0360 |

**The hybrid wins on 5 of 5 folds**, mean +0.0425, minimum +0.0360.

## Numerals: the annotation convention flipped twice

| batch | numerals tagged EN | OTHER | SI |
|---|---|---|---|
| R0002–R0028 | 197 | 58 | 1 |
| R0031–R0049 | 0 | 654 | 0 |
| R0050–R0061 | 78 | 4 | 10 |

One annotator group tags every digit OTHER, the others mostly EN. Heuristic
accuracy on numerals is **27%** (727 of 1,002 wrong) — not a model failure but
a measurement of which convention a recording was labelled under. This is the
Section D.3 numeric-tagging drift: 59 disagreeing numerals in the first batch,
over 700 now, and the single largest source of "error" in the all-token numbers. It needs a
team decision (one rule for numerals, written into `DATA_CONTRACT.md`), not a
better classifier.

## Data used, and what was excluded

`scripts/audit_gold.py` checks every gold file. Thirteen recordings are
excluded, with reasons kept in `scripts/corpus.py`:

| kind | recordings | evidence |
|---|---|---|
| malformed file | R0008 | junk timestamps |
| bulk mislabel | R0017, R0053, R0054, R0055, R0056, R0059 | every token tagged EN despite code-mixed speech |
| bulk mislabel | R0058 | 369/373 tokens tagged SI, incl. plain English |
| partial mislabel | R0051, R0062, R0063, R0064, R0065 | 14–51% of unambiguous Sinhala function words (`eka`, `mata`, `oyata`) tagged EN |

R0053 is also a near-duplicate of R0052 (94% sequence match), so keeping it
would have leaked the same conversation across folds.

**Sensitivity — keeping the five partially-mislabelled recordings in**
(`SORA_C1_KEEP_PARTIAL=1`, saved as `c1_lid_eval_with_partial.json`):

| | excluded (reported) | kept |
|---|---|---|
| hybrid, all tokens | 0.9232 ± 0.0218 | 0.8892 ± 0.0519 |
| hybrid, excl. numerals | 0.9735 ± 0.0143 | 0.9320 ± 0.0473 |
| fastText, Latin only | 0.9535 ± 0.0270 | 0.8818 ± 0.0705 |

Keeping them costs four to seven points and more than doubles the spread,
concentrated in the two folds that hold them (heuristic 0.78 on both). Their
gold labels disagree with themselves on words like `ekak`, so the model is
being scored against noise. Draft corrections for all 12 mislabelled
recordings are in `predictions/pretag/` (`scripts/pretag_relabel.py`).

## Why fastText only sees Latin-script tokens

Heuristic accuracy by script:

| script | tokens | accuracy | errors |
|---|---|---|---|
| Sinhala | 7,192 | 99.2% | 60 |
| Latin | 5,816 | 84.9% | 880 |
| numeric | 1,002 | 27.4% | 727 (convention, see above) |

Sinhala and Tamil script are resolved by Unicode range with no ambiguity, so a
classifier there would only be learning to imitate an already-exact rule. The
real headroom is in the Latin-script errors:

| confusion | count | examples |
|---|---|---|
| EN → SI | 680 | `NIC`, `madam`, `Colombo`, `SSCL`, `15th`, `Galle`, `SLIIT` |
| SI → EN | 178 | `mama`, `me`, `Ah`, `na`, `da`, `dan` |
| OTHER → SI/EN | 22 | identifiers, percentages, a few Tamil words |

Three quarters are proper nouns and acronyms that `wordfreq` does not know, so
the heuristic falls through to its romanized-Sinhala default. The rest are
romanized Sinhala colliding with English words (`mama` is in `wordfreq`).
Capitalisation is the strongest available cue for the first group, so the
fastText input carries explicit case markers alongside the lower-cased surface.

## `OTHER` is still not a learnable class

- **Zero Tamil-script tokens** in 14,010. The Unicode Tamil rule never fires.
- Of 743 OTHER tokens, **~720 are numerals** (above) and 11 are identifiers or
  ordinals (`8812304567V.`, `LN2024NG00445`, `21st,`, `No.`).
- **Only 11 OTHER tokens are Tamil**, and gold misses at least 7 more that are
  labelled SI or EN (`Vanakkam`, `romba`, `nandri`, `nalla visayam`). Tamil
  appears in Sinhala script, romanized, and once in Kannada/Malayalam
  codepoints — never in Tamil script.

fastText's OTHER F1 of 0.31–0.34 comes from ~22 Latin-script examples and
should not be quoted as a capability. Task 5 handles Tamil against curated
ground truth instead (`c1_otherlang_report.md`).

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
50,000 buckets is the default: a 37× size reduction at no measured cost. The
retrained model is **1.3 MB** on 5,816 Latin tokens. The sweep was not re-run
on the larger corpus.

## Scope of the claim

- Cross-validation numbers are the honest ones. The `lid_hybrid.py` demo runs
  the final model over tokens that were in its training set and is
  illustrative only.
- Fold sizes are uneven by token count (982–1,337 Latin tokens) because folding
  is by recording and balanced on English ratio.
- Many recordings come from shared scenario templates (e.g. R0012/R0013 are
  two "Senaratne and Associates" conversations). Folding by recording keeps a
  conversation out of its own test set but not its template's vocabulary, so
  these numbers are mildly optimistic for genuinely new topics.
- All figures are on **gold tokens**. End-to-end on ASR output they would be
  far worse — see `c1_token_stream_notes.md`.

## XLM-R / mBERT as future work

Still future work. The usable signal is now 5,816 Latin-script tokens across
two effective classes — better, but a transformer fine-tuned at that scale
would still be hard to validate, while fastText trains in seconds, quantizes to
1.3 MB and needs no GPU. Revisit once the corpus nears the TAF's 10–15 hr
target and the numeral convention is fixed.

Reproduce:

```bash
python scripts/audit_gold.py                         # which recordings are usable
python scripts/train_lid_fasttext.py                 # 5-fold CV + final model
SORA_C1_KEEP_PARTIAL=1 python scripts/train_lid_fasttext.py   # sensitivity run
python scripts/lid_hybrid.py                         # predictor demo
```
