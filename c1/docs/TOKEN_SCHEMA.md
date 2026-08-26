# C1 Extended Token Schema (Task 2)

Extends the shared `<rid>.tokens.jsonl` contract in
`SORA_Dataset/docs/DATA_CONTRACT.md` with the confidence fields the TAF asks
for. `SORA_Dataset` is treated as read-only here, so this document is the
authoritative spec for C1 *predictions*; the shared contract should be updated
by the team before any release that includes these fields.

## Current shared schema (gold annotations)

```json
{"utt_id", "tok_id", "token", "lang", "start", "end", "switch"}
```

`lang ∈ {SI, EN, OTHER}`, `start`/`end` in seconds.

## Extended schema (C1 predictions)

```json
{
  "utt_id":          "J26DS313_R0015_u001",
  "tok_id":          1,
  "token":           "Golden",
  "lang":            "EN",
  "start":           1.43,
  "end":             1.91,
  "switch":          true,
  "asr_confidence":  0.87,
  "lang_confidence": 0.94,
  "lang_method":     "wordfreq_en"
}
```

The first seven fields are unchanged, so anything reading the gold format
keeps working. Three fields are added.

### `asr_confidence` — did we hear the word?

Per-word probability from faster-whisper, obtained with
`word_timestamps=True` (`word.probability`). Range 0–1.

### `lang_confidence` — is the language label right?

Confidence in the `lang` tag, from whichever LID method produced it. Range
0–1.

**These two are deliberately separate fields and must not be merged.** They
answer different questions and fail independently. A token can be heard
perfectly and tagged wrongly (a clearly-articulated romanized Sinhala word
that also exists in English), or heard badly and tagged confidently (a
mangled token that is still unambiguously Sinhala script). Collapsing them
into one number destroys exactly the distinction C2/C3/C4 need in order to
decide whether to trust a token's text or its language.

### `lang_method` — which rule fired?

Provenance for the label, so a consumer can tell a deterministic
script-match from a frequency guess without re-deriving it. This matters
because Task 3 will add a second method (fastText) alongside the heuristic,
and mixed-provenance output is otherwise unauditable.

| value | meaning |
|---|---|
| `unicode_sinhala` | token contains Sinhala script — deterministic |
| `unicode_tamil` | token contains Tamil script — deterministic |
| `numeric` | token is purely numeric |
| `wordfreq_en` | Latin script, English by corpus frequency |
| `fallback_si` | Latin script, no English evidence — assumed romanized Sinhala |

## How `lang_confidence` is calibrated for the heuristic

Confidence reflects how *decisive the rule that fired actually is*, not how
common the label is. The heuristic's rules differ enormously in reliability,
and a single flat confidence would hide that.

| rule | confidence | reasoning |
|---|---|---|
| `unicode_sinhala` / `unicode_tamil` | 0.99 | Script presence is unambiguous. Near-certain by construction. |
| `numeric` | 0.50 | Deliberately low. Gold itself is inconsistent here — `"8"`→EN but `"075"`→OTHER — so this is annotation-guideline drift, not a confident prediction. Section D of the plan flags it; the confidence should say so rather than paper over it. |
| `wordfreq_en` | 0.50–0.95 | Scaled by how far the word's English frequency sits above the 1e-5 threshold, in decades. A word barely over the line gets ~0.5; a common word two decades above gets 0.95. |
| `fallback_si` | 0.50–0.90 | Negative evidence only — "not recognisably English". Scaled by how far *below* threshold, and capped lower than the EN ceiling because absence of English evidence is weaker than presence of Sinhala script. |

### Known miscalibration

The fallback cannot detect romanized Sinhala that collides with a real
English word. `mama` (Sinhala "I") carries genuine English frequency, so it
is tagged EN with *high* confidence — confidently wrong. The confidence
signal does not catch this class of error and should not be trusted to.
Task 3's fastText classifier targets precisely these ambiguous Latin-script
cases, and Task 5 handles romanized Tamil the same way.

`OTHER` is only ever produced by the Tamil-script rule, so romanized Tamil is
currently invisible to this method — consistent with the 0% OTHER recall
already measured in the LID baseline.

## `switch`

Derived, not predicted: `switch = (tok_id > 0) and (lang[i] != lang[i-1])`
within an utterance. The first token of each utterance is always `false`.

**The gold `switch` field does not reliably follow this rule, or any other.**
It disagrees with the derivation on 137 of 5,818 tokens (2.4%), and 129 of
those sit on the first token of an utterance. That looked like gold using a
cross-utterance convention instead, but it is not: at utterance boundaries
where the language actually changes, gold marks `switch=true` 102 times and
`switch=false` 100 times — a coin flip — plus 27 tokens marked as switches
with no language change at all.

Task 4 must therefore **re-derive the gold switch sequence from gold `lang`**
rather than scoring against the annotated field, or switch-point F1 will
largely measure annotator inconsistency at turn boundaries. Full analysis and
reproduction steps in [SWITCH_FIELD_AUDIT.md](SWITCH_FIELD_AUDIT.md).
