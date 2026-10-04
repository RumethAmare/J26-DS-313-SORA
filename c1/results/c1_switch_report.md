# Task 4 — Switch-point detection and code-mixing index

Produced by `scripts/switch_detection_eval.py`.

Scope: **14,010 gold tokens across 48 recordings**, after excluding 13 recordings with unusable labels (reasons in `scripts/corpus.py`, evidence in `results/c1_gold_audit.md`).

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
| heuristic | within-utterance | 0.8735 | 0.8047 | 0.8377 | 4864 | 4481 |
| hybrid | within-utterance | 0.9447 | 0.9132 | 0.9287 | 4864 | 4702 |
| heuristic | cross-utterance | 0.8765 | 0.8015 | 0.8373 | 5340 | 4883 |
| hybrid | cross-utterance | 0.9458 | 0.9112 | 0.9282 | 5340 | 5145 |

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
| heuristic | 0.8810 | 0.8377 | +0.0433 |
| hybrid | 0.9236 | 0.9287 | -0.0051 |

Token accuracy here includes numerals, whose gold convention is
inconsistent across annotators (EN in some recordings, OTHER in others).
Those errors come in contiguous runs — a phone number read digit by
digit — so they cost token accuracy on every digit but switch F1 only
at the run's two edges. That is why the gap is narrower than LID
accuracy alone would suggest (heuristic +0.0433, hybrid -0.0051).

The amplification still cuts both ways: improving LID accuracy
by **+0.0426** moved switch F1 by
**+0.0910** — about **2.14×**
the token-level gain.

That ratio is the practical argument for Task 3's classifier. Judged on
token accuracy alone the hybrid looks like a modest 4-point
improvement; judged on the switch points that actually define
code-switching, it is worth about 2.14× that. Switch F1, not token
accuracy, is the honest headline metric for this sub-objective.

## Code-mixing index (corpus characterization)

Das & Gambäck (2014): `CMI = 100 × (N − max_L) / N` per utterance, where
N is language-dependent tokens and `max_L` the dominant language's count.
0 is monolingual; higher is more evenly mixed.

- utterances measured: **1271**
- mean CMI: **27.38** (median 30.00, sd 17.26)
- mean CMI excluding numeric tokens: **25.42**
- monolingual utterances (CMI = 0): **17.9%**
- utterances with CMI > 25: **56.3%**

Both variants are given because Section D.3 found gold's numeric tagging
inconsistent, and the October 2026 batch made it worse: the first batch
tags numerals mostly EN, R0031–R0049 tag every one OTHER, R0050 onward
EN again. Numerals therefore inject arbitrary label mass into the mix.

### Most and least code-mixed recordings

| recording | utterances | mean CMI | excl. numeric | monolingual utts |
|---|---|---|---|---|
| J26DS313_R0052 | 42 | 37.14 | 36.46 | 4.8% |
| J26DS313_R0041 | 32 | 35.78 | 30.24 | 12.5% |
| J26DS313_R0013 | 14 | 34.27 | 33.28 | 7.1% |
| J26DS313_R0024 | 24 | 34.17 | 31.37 | 12.5% |
| J26DS313_R0042 | 32 | 33.59 | 25.13 | 6.2% |
| … | | | | |
| J26DS313_R0060 | 19 | 18.53 | 18.99 | 31.6% |
| J26DS313_R0010 | 9 | 17.80 | 17.80 | 44.4% |
| J26DS313_R0057 | 25 | 16.33 | 14.66 | 48.0% |
| J26DS313_R0027 | 74 | 13.72 | 12.94 | 40.5% |
| J26DS313_R0004 | 9 | 11.26 | 11.26 | 44.4% |

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
