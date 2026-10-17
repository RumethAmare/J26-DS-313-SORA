# Audit: the gold `switch` field is unreliable (affects Task 4)

The plan's Task 4 says switch-point detection is "derived, not separately
trained: a switch point is any adjacent token pair where `lang[i] != lang[i-1]`
within an utterance (matches the gold `switch` field already in the schema —
audit whether it's populated correctly)."

This is that audit. **The gold `switch` field does not match that definition,
and it does not match any single consistent alternative either.** Task 4 must
not score against it as annotated.

## What was measured

Across all 27 gold `*.tokens.jsonl` files (5,818 tokens), the annotated
`switch` value disagrees with the plan's derived definition on 137 tokens
(2.4%). 129 of those 137 sit on `tok_id 0` — the first token of an utterance,
where the plan's within-utterance rule always yields `false`.

That pattern suggested gold might be using a *cross-utterance* convention
instead: a first token counts as a switch when the previous utterance ended in
a different language. That would be a legitimate, arguably better definition,
since a speaker moving from a Sinhala utterance to an English one really has
code-switched.

It is not what gold does. Re-deriving under the cross-utterance convention
gives 135 mismatches instead of 137 — no meaningful improvement, because the
new rule fixes ~102 tokens and breaks ~100 others.

## The actual behaviour

Gold's treatment of the first token of an utterance, classified by what the
*previous* utterance ended in:

| situation | count | reading |
|---|---|---|
| previous utterance ended in a DIFFERENT language, `switch=true` | 102 | cross-utterance convention |
| previous utterance ended in a DIFFERENT language, `switch=false` | 100 | within-utterance convention |
| previous utterance ended in the SAME language, `switch=false` | 330 | consistent under both |
| previous utterance ended in the SAME language, `switch=true` | 27 | inconsistent under both |
| first utterance of a recording (no predecessor) | 27 | n/a |

The 102 / 100 split is the finding. At utterance boundaries the annotation is
effectively a coin flip between the two conventions, and a further 27 tokens
are marked as switches with no language change at all on either reading.

Within utterances the field is fine: only 8 of the 137 disagreements fall on
non-first tokens.

## Consequence for Task 4

Switch-point F1 scored directly against this field would be measuring
annotation noise at every utterance boundary. Roughly 200 boundary tokens are
annotated arbitrarily, against ~2,012 total `switch=true` tokens — large enough
to move the metric substantially and in a way that has nothing to do with model
quality.

**Recommendation: re-derive the gold switch sequence from the gold `lang`
field rather than reading the annotated `switch` field.** `lang` is the primary
annotation and is far more trustworthy — the untrained heuristic already agrees
with it on 88% of tokens, and it is what the token-level LID task is scored on
anyway. Deriving both sides with the same explicit rule makes switch-point F1
measure the model instead of the annotators' inconsistency.

The convention chosen must then be stated, because it changes the target:

- **within-utterance** (what `build_token_stream.py` implements, per the plan):
  first token of each utterance is never a switch. Well-defined and simple;
  under-counts genuine switches that happen across a turn boundary.
- **cross-utterance**: carry the previous token's language across utterance
  boundaries. Linguistically closer to what code-switching means, but conflates
  speaker changes with code-switching unless speaker identity is consulted —
  and `tokens.jsonl` carries no speaker field, so that would need joining
  against `<rid>.transcript.json` or the C3 RTTM.

Within-utterance is the current default because it is the plan's stated rule
and needs no external join. If Task 4 wants the cross-utterance variant, it
should take speaker turns into account rather than treating every utterance
boundary as continuous speech.

## Recommended follow-up (not done here)

Fixing the gold field is a re-annotation task, not a code change, and is out of
scope for Task 2. The cheap intermediate step is a `validate.py` check that
warns when `switch` disagrees with the value derived from `lang` — the same
shape as the numeric-tagging check Section D already proposes. It would have
caught this at annotation time.

Reproduce with:

```bash
python3 scripts/validate_token_stream.py --gold
```
