# C2 schema — clean text, summary, action items

The output contract C2 emits to Component 4, and the gold format it is scored
against. Gold lives in `SORA_Dataset/annotations/c2/`, per
`docs/DATA_CONTRACT.md`.

## Input (from C1 + C3)

`annotations/c1/<rid>.transcript.json` — speaker-attributed, code-mixed
utterances:

```json
{"utt_id": "J26DS313_R0002_u001", "speaker": "S1", "start": 1.19, "end": 4.98,
 "text": "hello sampath bank customer care amashi speaking kohomada help karanne"}
```

## Output 1 — `<rid>.translation.json`

One record per transcript utterance, same `utt_id`, a full word-for-word clean
rendering in each language:

```json
{"recording_id": "J26DS313_R0002",
 "utterance_records": [
   {"utt_id": "J26DS313_R0002_u001",
    "clean_en": "Hello, Sampath Bank customer care, Amashi speaking. How can I help you?",
    "clean_si": "ආයුබෝවන්, සම්පත් බැංකුවේ පාරිභෝගික සේවය, අමාෂි කතා කරන්නේ. ..."}]}
```

`clean_en` and `clean_si` are a **record**, not a paraphrase: every statement in
the utterance survives, only disfluencies and code-mixing are removed.
Identifiers (NIC, phone, account numbers) are copied verbatim — C4 locates them
by character offset in these exact strings.

## Output 2 — `<rid>.summary.json`

```json
{"recording_id": "J26DS313_R0002",
 "summaries": {"en": {"text": "..."}, "si": {"text": "..."}},
 "action_items": [
   {"intent": "Block the lost debit card", "owner": "Amashi (agent)",
    "receiver": null, "deadline": "immediate"}]}
```

| Field | Meaning |
|---|---|
| `intent` | What will be done, as a short English verb phrase |
| `owner` | Who committed to it |
| `receiver` | The person it is directed at (`"Call Suresh"` → `"Suresh"`); `null` if none |
| `deadline` | When, as spoken (`"Friday"`, `"immediate"`); `null` if none |

**Indirect commitments count.** `"mama call karannam"` ("I'll call") and
`"balamu"` ("let's see/check") are commitments even though no English modal
appears. Missing these is the failure the proposal names for cloud tools, so
gold must include them.

## Format drift found in the corpus

`python src/audit.py` reports the full census. As of 2026-10-08, across 78
recordings, the canonical shape above is used by 66; the rest use
`utterances` instead of `utterance_records`, a bare list, `recording` instead
of `recording_id`, flat `summary`/`summary_si` strings, or a `summaries` list.
48 action items are plain strings rather than objects. `src/corpus.py` reads
every one of these shapes so nothing is dropped, and the audit lists which file
uses which, so they can be fixed at source.

**New annotations should use the canonical shape above.**

## Normalization (C2 stage 1)

Applied to the code-mixed transcript before translation. Deterministic, so it
is unit-tested rather than learned. See `src/normalize.py`.

| Rule | Example |
|---|---|
| Split hyphenated Sinhala clitics off English stems | `meeting-eka` → `meeting eka` |
| Split Latin stems glued to Sinhala-script suffixes | `bankඑකට` → `bank එකට` |
| Drop filler tokens | `uh`, `umm`, `hmm`, `අ` |
| Tidy whitespace and stray punctuation | `hari ,  ok` → `hari, ok` |

Repeated words are **not** collapsed by default: Sinhala uses reduplication
meaningfully (`podi podi`, `hemin hemin`), and a rule cannot tell that from a
stutter.
