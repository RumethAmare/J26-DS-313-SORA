# Task 4 — Switch-point detection and code-mixing index

Produced by `scripts/switch_detection_eval.py`.

## Two deviations from the plan, both forced by the data

**Gold switches are re-derived from gold `lang`, not read from the
annotated `switch` field.** The plan expected that field to serve as the
target. It cannot: at utterance boundaries where the language actually
changes, gold marks `switch=true` 102 times and `switch=false` 100 times,
plus 27 tokens flagged as switches with no language change at all. Scoring
against it would measure annotator inconsistency. Full analysis in
[SWITCH_FIELD_AUDIT.md](../docs/SWITCH_FIELD_AUDIT.md).

**The primary metric is computed on gold tokens.** End-to-end switch F1 is
bounded by the ASR before switch detection contributes anything — the
baseline emits 1,280 tokens against gold's 5,818. It is reported below,
separately and labelled, so an ASR failure is not attributed to this task.

## Switch-point F1 on gold tokens

Positions, not token labels: a sequence can be 90% correct token-wise and
still miss every switch, because switches are rare and sit exactly where
the labels are hardest.

| LID method | convention | P | R | F1 | gold switches | predicted |
|---|---|---|---|---|---|---|
| heuristic | within-utterance | 0.8188 | 0.7622 | 0.7895 | 1850 | 1722 |
| hybrid | within-utterance | 0.8753 | 0.8654 | 0.8703 | 1850 | 1829 |
| heuristic | cross-utterance | 0.8233 | 0.7565 | 0.7885 | 2045 | 1879 |
| hybrid | cross-utterance | 0.8790 | 0.8631 | 0.8710 | 2045 | 2008 |

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
slightly from `c1_lid_report.md`'s 0.8994 / 0.9318, which average the
five per-fold accuracies. Same predictions, micro vs macro averaging.

| method | LID token accuracy | switch F1 | gap |
|---|---|---|---|
| heuristic | 0.9010 | 0.7895 | +0.1115 |
| hybrid | 0.9347 | 0.8703 | +0.0644 |

Switch F1 sits roughly 6–11 points below LID accuracy for both
methods, and the amplification cuts both ways: improving LID accuracy
by **+0.0337** moved switch F1 by
**+0.0808** — about **2.4×**
the token-level gain.

That ratio is the practical argument for Task 3's classifier. Judged on
token accuracy alone the hybrid looks like a modest three-point
improvement; judged on the switch points that actually define
code-switching, it is worth several times that. Switch F1, not token
accuracy, is the honest headline metric for this sub-objective.

## Code-mixing index (corpus characterization)

Das & Gambäck (2014): `CMI = 100 × (N − max_L) / N` per utterance, where
N is language-dependent tokens and `max_L` the dominant language's count.
0 is monolingual; higher is more evenly mixed.

- utterances measured: **550**
- mean CMI: **25.41** (median 27.27, sd 17.40)
- mean CMI excluding numeric tokens: **24.71**
- monolingual utterances (CMI = 0): **21.3%**
- utterances with CMI > 25: **52.2%**

Both variants are given because Section D.3 found gold's numeric tagging
inconsistent (192 EN / 61 OTHER / 1 SI for comparable tokens), so numerics
inject arbitrary label mass into the mix.

### Most and least code-mixed recordings

| recording | utterances | mean CMI | excl. numeric | monolingual utts |
|---|---|---|---|---|
| J26DS313_R0013 | 14 | 34.27 | 33.28 | 7.1% |
| J26DS313_R0024 | 24 | 34.17 | 31.37 | 12.5% |
| J26DS313_R0022 | 22 | 32.83 | 31.53 | 0.0% |
| J26DS313_R0015 | 25 | 32.26 | 31.86 | 8.0% |
| J26DS313_R0012 | 16 | 31.50 | 31.92 | 12.5% |
| … | | | | |
| J26DS313_R0023 | 24 | 21.76 | 19.33 | 37.5% |
| J26DS313_R0002 | 14 | 19.21 | 20.27 | 28.6% |
| J26DS313_R0010 | 9 | 17.80 | 17.80 | 44.4% |
| J26DS313_R0027 | 74 | 13.72 | 12.94 | 40.5% |
| J26DS313_R0004 | 9 | 11.26 | 11.26 | 44.4% |

## End-to-end (ASR → LID → switch) — ASR-bounded

Predicted tokens are aligned to gold by maximum temporal overlap, since
the two sequences differ in length and content.

- gold tokens matched to any ASR token: **2806/5444 (51.5%)**
- precision 0.3942, recall 0.0226, **F1 0.0427**

This number describes the ASR, not the switch-detection logic. Roughly
four fifths of gold tokens have no corresponding ASR token at all, so
most gold switches are unreachable regardless of how good the LID is.
Read the gold-token table above for the quality of this task's own
contribution, and `c1_report.md` for why the ASR is the bottleneck.

## What this says about the sub-objective

Switch detection needs no model of its own — it is a deterministic function
of the language sequence, so its accuracy is entirely inherited from LID.
That makes it a useful diagnostic rather than a component to optimise: it
amplifies LID errors, because one mislabelled token in a monolingual run
creates *two* spurious switches. That amplification is why switch F1 sits
well below LID token accuracy and is the honest metric to quote for
code-switching capability.
