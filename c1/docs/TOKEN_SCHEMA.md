# C1 token schema — SI / EN / NUM / OTHER

Proposed schema for per-token language identification, and the output contract
Stage D emits to Components 2 and 3. Derived from a full audit of all 27
recordings (5 818 tokens); reproduce with `python -m src.audit_labels`.

## The four labels

| Label | Means | Decided by |
|---|---|---|
| `SI` | Sinhala, in either script | Annotation / model |
| `EN` | English, in either script | Annotation / model |
| `NUM` | Numerals and numeric identifiers | Deterministic rule |
| `OTHER` | **Tamil** | Annotation / model |

`OTHER` means Tamil specifically — Sri Lanka's other national language — not
"anything else". This matches the intent behind `docs/DATA_CONTRACT.md`; the
`NUM` class is what makes it true in practice.

## Why NUM was added

The three-label schema was not being applied consistently. Of 92 `OTHER`
tokens, 81 were numerals, phone numbers, NIC numbers or reference codes;
separately, 227 numeric tokens sat in `EN` and 15 in `SI`. The same
phenomenon was split across three classes on no stated principle.

The measured cost is in `SORA_Dataset/results/c1_lid_eval.json`: `OTHER`
scored precision 0.00, recall 0.00, **F1 0.00**. Not a modelling failure — the
class had no learnable definition.

Splitting `NUM` out moves 323 tokens (5.6 % of the corpus) and leaves `OTHER`
with 11 tokens that are actually about language:

```
label     before    after    delta
SI          3338     3323      -15
EN          2388     2161     -227
NUM            0      323     +323
OTHER         92       11      -81
```

`NUM` is also directly useful downstream: C4 needs NIC and phone-number spans
for PII redaction, and C2 needs them for action-item extraction. One rule
serves three components.

### The NUM rule

Strip leading and trailing punctuation. The token is `NUM` if what remains
contains a digit and no `@`.

Deliberately mechanical, so it never needs adjudicating. It captures bare
numerals (`1990`), grouped amounts (`15,000`), decimals (`1.5`), percentages
(`70%`), times (`10:00`), ordinals (`21st`), phone numbers (`0777998876`),
NICs (`8812304567V`) and reference codes (`CEB/KW/2024/008834`). The `@`
exclusion keeps email addresses out — they carry digits but are not numbers,
and they belong in a review bucket rather than a language class.

## The 11 remaining OTHER tokens

Small enough to list, which is the point — they can be verified by hand.

| Recording | Script | Token |
|---|---|---|
| R0012 | Latin | `puriyam.` |
| R0013 | Latin | `ಪುರಿಯമ്.` |
| R0015 | Sinhala | `ඉදු`, `ඔරුව`, `පෙරිය`, `ප්‍රචන`, `ඉල්ලයි.` |
| R0016 | Latin | `sari`, `sari.` |
| R0018 | Latin | `hasitha.94@gmail.com.`, `No.` |

Two open items for the annotation team:

- `hasitha.94@gmail.com.` and `No.` are not Tamil. The email belongs in a
  review bucket; `No.` is English.
- **There are zero Tamil-script tokens in the corpus.** Every Tamil token is
  written either romanized or in Sinhala script, so a script-based feature can
  never find them. With 11 examples, `OTHER` cannot be learned from this data
  at all — it needs either targeted Tamil collection or an external lexicon,
  and until then its F1 should be reported as not-measurable rather than 0.00.

## English loanwords in Sinhala script

56 tokens (43 distinct) are labelled `EN` but written in Sinhala script:
`හොටෙල්` (hotel), `රූම්ස්` (rooms), `එන්අයිසී` (NIC), `නම්බර්` (number),
`ඇඩ්රස්` (address), `ට්‍රිප්` (trip).

**This is correct annotation, not drift.** It is the genuinely hard case for
per-token LID: the script says Sinhala and the word is English. Any classifier
built on script features alone will get all 56 wrong, so `lid_features.py`
needs lexical and character-n-gram features rather than a script flag.

## Two systematically mislabelled recordings

Bigger than the `NUM` drift, and found by a check that needs no external
lexicon: compare each token's label against how the same word (folded, so
`කොහොමද` and `kohomada` count as one) is labelled in every *other* recording.
A token is only checkable when it appears elsewhere with enough occurrences to
form a majority.

The corpus splits cleanly — 25 recordings contradict the rest on ≤7% of their
checkable tokens, which is ordinary ambiguity (`me` is both English *me* and
Sinhala *මේ*; `one` collides with *ඕන*). Two do not:

| Recording | Rate | Direction | Examples |
|---|---|---|---|
| **R0017** | **55%** of 179 | Sinhala labelled `EN` | `mama`, `Oyata`, `ekak`, `oneda?`, `eken,` |
| **R0023** | **20%** of 128 | English labelled `SI` | `complaint`, `sir,`, `file`, `Phone`, `number` |

R0017 is the worse case and the manifest agrees with the error — it declares
`EN+OTHER+OTHER`, so nothing cross-checks it. The transcript is plainly
code-mixed ("Good morning Ceylon Electricity Board Kaduwela office **eken, mama
Chaminda. Oyata connection ekak oneda?**"); its romanized Sinhala was read as
English, at the manifest level and the token level together.

Both must be corrected **before** any CRF training or LID evaluation. A
recording annotated on a different reading of the label set does not add noise,
it adds a competing definition — it poisons training and makes the reported F1
unfalsifiable. Note that R0023 sits in the **eval** split, so this affects the
headline LID number directly. Correcting labels does not change the split, so
`data/manifest.csv` stays locked.

## The `switch` field

Annotated `switch=true` on 2 012 of 5 818 tokens (34.6 %). Deriving it
instead as "language differs from the previous token in the same utterance"
gives 1 877 (32.3 %), agreeing with the annotation **97.6 %** of the time —
136 tokens are flagged as switches without a language change, 1 the reverse.

Close, but not identical, so `switch` does not mean exactly "the language
changed". Two consequences:

- Stage D **derives** switch markers from the language sequence rather than
  copying the field, so the definition is one line of code rather than 27
  annotators' judgement.
- Switch-point accuracy (§9 of the implementation document) needs that
  definition stated before it is reported, or the metric is unfalsifiable.

## Output contract (Stage D)

Extends `docs/DATA_CONTRACT.md` with the confidence score §1 requires:

```json
{"utt_id": "J26DS313_R0002_u001", "tok_id": 0, "token": "hello",
 "lang": "EN", "start": 1.19, "end": 1.49, "conf": 0.92, "switch": false}
```

| Field | Type | Notes |
|---|---|---|
| `utt_id` | string | `<rid>_u###`, zero-padding identical across all layers |
| `tok_id` | int | 0-based within the utterance |
| `token` | string | Surface form, in the decoded script |
| `lang` | enum | `SI` / `EN` / `NUM` / `OTHER` |
| `start`, `end` | float | Seconds into the recording, from Whisper word timestamps |
| `conf` | float | 0–1, the ASR word probability |
| `switch` | bool | Derived: `lang` differs from the previous token in the utterance |

`conf` is additive, so the output stays readable by anything expecting the
existing contract. Adding `NUM` to the `lang` enum is the one change that needs
team sign-off before `validate.py` will accept it.

## Status

Proposed. `src/audit_labels.py` writes `data/label_relabel_patch.jsonl` with all
323 changes for review; **nothing in `SORA_Dataset` has been modified.** Applying
the patch is a Phase 1 team decision.
