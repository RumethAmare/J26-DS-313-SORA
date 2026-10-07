# Task 4 — Switch-point detection and code-mixing index

Produced by `scripts/switch_detection_eval.py`.

Scope: **25,582 gold tokens across 66 recordings**, after excluding 12 recordings with unusable labels (reasons in `scripts/corpus.py`, evidence in `results/c1_gold_audit.md`).

## Two deviations from the plan, both forced by the data

**Gold switches are re-derived from gold `lang`, not read from the
annotated `switch` field.** The plan expected that field to serve as the
target. It cannot: in the audit of the original 27 recordings, at utterance
boundaries where the language actually changes, gold marks `switch=true`
102 times and `switch=false` 100 times, plus 27 tokens flagged as switches
with no language change at all. Scoring against it would measure annotator
inconsistency. Full analysis in
[SWITCH_FIELD_AUDIT.md](../docs/SWITCH_FIELD_AUDIT.md).

**The primary metric is computed on gold tokens.** End-to-end switch F1 is
bounded by the ASR before switch detection contributes anything — on the
original recordings the baseline emits 1,280 tokens against gold's 5,818.
It is reported below, separately and labelled, so an ASR failure is not
attributed to this task.

## Switch-point F1 on gold tokens

Positions, not token labels: a sequence can be 90% correct token-wise and
still miss every switch, because switches are rare and sit exactly where
the labels are hardest.

| LID method | convention | P | R | F1 | gold switches | predicted |
|---|---|---|---|---|---|---|
| heuristic | within-utterance | 0.8772 | 0.7901 | 0.8314 | 8544 | 7696 |
| hybrid | within-utterance | 0.9642 | 0.9318 | 0.9477 | 8544 | 8257 |
| heuristic | cross-utterance | 0.8801 | 0.7882 | 0.8316 | 9207 | 8246 |
| hybrid | cross-utterance | 0.9638 | 0.9286 | 0.9459 | 9207 | 8871 |

The `cross-utterance` rows are a sensitivity check. That convention is
linguistically closer to what code-switching means, but it conflates a
speaker change with a code-switch, because `tokens.jsonl` carries no
speaker field. Resolving it properly needs a join against the C3 RTTM.

### Switch detection amplifies LID errors

Switch detection trains nothing of its own — it is a deterministic
function of the language sequence — so all of its error comes from LID.
But it does not inherit that error one-for-one. A single mislabelled
token in the middle of a monolingual run creates **two** spurious
switches, one entering the error and one leaving it.

The LID accuracies below are pooled over all tokens, so they differ
slightly from `c1_lid_report.md`'s fold-averaged figures. Same
predictions, micro vs macro averaging.

| method | LID token accuracy | switch F1 | gap |
|---|---|---|---|
| heuristic | 0.8965 | 0.8314 | +0.0651 |
| hybrid | 0.9443 | 0.9477 | -0.0034 |

Token accuracy here includes numerals, whose gold convention is
inconsistent across annotators (EN in some recordings, OTHER in others).
Those errors come in contiguous runs — a phone number read digit by
digit — so they cost token accuracy on every digit but switch F1 only
at the run's two edges. That is why the gap is narrower than LID
accuracy alone would suggest (heuristic +0.0651, hybrid -0.0034).

The amplification still cuts both ways: improving LID accuracy
by **+0.0478** moved switch F1 by
**+0.1163** — about **2.43×**
the token-level gain.

That ratio is the practical argument for Task 3's classifier. Judged on
token accuracy alone the hybrid looks like a modest 5-point
improvement; judged on the switch points that actually define
code-switching, it is worth about 2.43× that. Switch F1, not token
accuracy, is the honest headline metric for this sub-objective.

## Code-mixing index (corpus characterization)

Das & Gambäck (2014): `CMI = 100 × (N − max_L) / N` per utterance, where
N is language-dependent tokens and `max_L` the dominant language's count.
0 is monolingual; higher is more evenly mixed.

- utterances measured: **1791**
- mean CMI: **28.21** (median 31.03, sd 17.21)
- mean CMI excluding numeric tokens: **26.22**
- monolingual utterances (CMI = 0): **16.5%**
- utterances with CMI > 25: **58.7%**

Both variants are given because Section D.3 found gold's numeric tagging
inconsistent, and the October 2026 batches made it worse: R0002–R0028 and
R0050–R0061 tag numerals mostly EN, while R0029–R0049 and R0068–R0075 tag
every one OTHER (see `c1_lid_report.md`). Numerals therefore inject
arbitrary label mass into the mix.

### Most and least code-mixed recordings

| recording | utterances | mean CMI | excl. numeric | monolingual utts |
|---|---|---|---|---|
| J26DS313_R0074 | 19 | 40.23 | 38.23 | 5.3% |
| J26DS313_R0052 | 25 | 39.54 | 39.91 | 0.0% |
| J26DS313_R0070 | 47 | 39.34 | 34.77 | 6.4% |
| J26DS313_R0075 | 22 | 37.86 | 32.97 | 9.1% |
| J26DS313_R0071 | 35 | 36.52 | 31.20 | 8.6% |
| … | | | | |
| J26DS313_R0027 | 74 | 13.72 | 12.94 | 40.5% |
| J26DS313_R0103 | 13 | 11.59 | 11.59 | 30.8% |
| J26DS313_R0104 | 17 | 11.38 | 11.38 | 23.5% |
| J26DS313_R0004 | 9 | 11.26 | 11.26 | 44.4% |
| J26DS313_R0102 | 17 | 10.41 | 10.41 | 35.3% |

## End-to-end (ASR → LID → switch) — ASR-bounded

Predicted tokens are aligned to gold by maximum temporal overlap, since
the two sequences differ in length and content. Only recordings with
both an ASR token stream and gold timestamps contribute — currently the
original batch, since the new recordings have no audio in the dataset
repo yet.

- gold tokens matched to any ASR token: **2806/5444 (51.5%)**
- precision 0.3942, recall 0.0226, **F1 0.0427**

This number describes the ASR, not the switch-detection logic. 48% of gold tokens
overlap no ASR token at all, and an overlapping token is usually not the
right word, so most gold switches are unreachable regardless of how good
the LID is.
Read the gold-token table above for the quality of this task's own
contribution, and `c1_report.md` for why the ASR is the bottleneck.

## What this says about the sub-objective

Switch detection needs no model of its own — it is a deterministic function
of the language sequence, so its accuracy is entirely inherited from LID.
That makes it a useful diagnostic rather than a component to optimise: it
amplifies LID errors, because one mislabelled token in a monolingual run
creates *two* spurious switches. That amplification is why switch F1,
not token accuracy, is the honest metric to quote for code-switching
capability.
