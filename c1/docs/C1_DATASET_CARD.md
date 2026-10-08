# SORA C1 code-switching token dataset — dataset card

**Version 0.1.0** · built by `c1/scripts/build_dataset_release.py` · files in
`c1/release/` · source data: `SORA_Dataset` at commit `75f4469` (2026-10-05).

Token-level Sinhala–English code-switching annotations for Sri Lankan
conversational speech, with C1's automatic language labels, switch points and
Tamil flags alongside the human labels. Built for the other SORA components
(C2 translation, C3 diarization, C4 PII) to consume.

## At a glance

| | usable | all (incl. excluded) |
|---|---|---|
| recordings | **66** (59 with audio) | 78 |
| gold tokens | **25,582** | 28,974 |
| utterances | 1,791 | — |
| audio | **3.2 h** | — |
| language mix (gold) | SI 16,224 · EN 8,294 · OTHER 1,064 | — |
| switch points | 8,544 | — |
| mean code-mixing index | 28.2 per utterance (median 31.0) | — |

58.7% of utterances have a code-mixing index above 25 and only 16.5% are
monolingual, so this is densely mixed speech rather than mostly-monolingual
speech with occasional English words.

## Files

```
c1/release/
  MANIFEST.json          per-recording status + statistics, totals, provenance
  tokens/<rid>.jsonl     one record per gold token (78 files)
  asr/<rid>.tokens.jsonl ASR token stream from Task 2 (61 files)
```

### `tokens/<rid>.jsonl` — one record per gold token

| field | meaning | source |
|---|---|---|
| `recording`, `utt_id`, `tok_id` | identifiers | gold |
| `token` | surface form, as annotated | gold |
| `start`, `end` | seconds; **null** for 868 tokens in 9 recordings | gold |
| `lang` | human label: `SI`, `EN` or `OTHER` | gold |
| `switch` | language differs from the previous token in the same utterance, **derived from `lang`** | C1 Task 4 |
| `switch_annotated` | the annotators' own `switch` field — kept for reference, not reliable (see below) | gold |
| `script` | `sinhala`, `latin`, `tamil` or `numeric` | C1 |
| `numeric` | token is purely digits — see the numeral caveat | C1 |
| `lang_pred`, `lang_pred_confidence`, `lang_pred_method` | C1's automatic label, confidence and the rule/model that produced it | C1 Task 3 |
| `lang_pred_source` | `out_of_fold` (usable recordings) or `final_model` (excluded) | C1 |
| `tamil_flag`, `tamil_reason` | Tamil candidate, and which signal fired | C1 Task 5 |

### `asr/<rid>.tokens.jsonl` — what off-the-shelf ASR produces

Whisper-`small` word-level output with timestamps, `asr_confidence`, and the
same language labelling, in the schema of `docs/TOKEN_SCHEMA.md`. It is
included so consumers can build against the end-to-end format, **not** as
usable content: it contains 4,230 tokens for recordings whose gold has 23,677
(18%). See *ASR quality* below.

### `MANIFEST.json`

Per recording: `status` (`usable`, `usable_no_audio`, `excluded`),
`exclusion_reason`, token and utterance counts, language mix, switch count,
code-mixing index, agreement between `lang_pred` and `lang`, Tamil flag count,
null-timestamp count, Sinhala script convention, outstanding audit issues,
audio duration and ASR token count. Plus corpus totals and provenance (commits,
model path and SHA-256, thresholds).

## How the C1 fields were produced, and how good they are

All figures are 5-fold cross-validation folded by **recording**, so no
conversation is in both training and test.

**`lang_pred` — hybrid language ID.** Sinhala script → `SI` by Unicode rule;
pure digits → numeric rule; Latin script → fastText classifier
(`models/lid_latin.ftz`, 1.3 MB). For usable recordings every label comes from
a model that never saw that recording, so the confidences behave as they
would on new data. Agreement with gold in this release:

| scope | accuracy |
|---|---|
| all tokens | **0.944** |
| excluding numerals | **0.984** |
| Latin-script tokens (where the model works) | 0.965 |

**`switch`.** A deterministic function of the label sequence. Computed from
C1's labels instead of gold, switch-point F1 against gold is **0.948**
(`results/c1_switch_report.md`). In this file `switch` is derived from the
*gold* labels, so it is exact with respect to `lang`.

**`tamil_flag`.** Gazetteer + character n-gram detector in a romanized space
that covers Tamil written in Sinhala script as well as Latin. At the chosen
operating point (99th percentile) it flags 70 usable tokens: **12 of the 16
known Tamil tokens**, plus 58 others. Treat it as a candidate list for human
review, not a label.

## Known limitations — read before using

1. **12 recordings are excluded** (status `excluded`, reason in the manifest):
   - R0008 — malformed timestamps.
   - R0017, R0054, R0055, R0056, R0059 — every token tagged EN despite
     code-mixed speech. R0058 — nearly every token tagged SI.
   - R0051, R0062–R0065 — 14–51% of unambiguous Sinhala function words
     (`eka`, `mata`, `oyata`) tagged EN.

   Their gold `lang` should not be trusted; `lang_pred` is the useful field.
   Draft corrections for 11 of them are in `c1/predictions/pretag/`.
2. **Numerals follow two conventions.** R0002–R0028 and R0050–R0061 tag digits
   mostly `EN`; R0029–R0049 and R0068–R0075 tag every digit `OTHER`. R0100–R0105
   write numbers as words. Use the `numeric` field to set numerals aside: they
   are ~5% of tokens and most of the disagreement between `lang_pred` and
   `lang`.
3. **`OTHER` is not "Tamil".** The project defines OTHER as Tamil, but of
   1,064 OTHER tokens, 1,041 are numerals, 11 are identifiers or ordinals, and
   12 are Tamil. At least 6 Tamil tokens are labelled SI instead — including
   `රොම්බ නල්ල` (*romba nalla*, "very good") in R0005, found while building
   this release and not yet in `data/tamil_ground_truth.json`.
4. **Two Sinhala script conventions.** 55 usable recordings write Sinhala in
   Sinhala Unicode, 10 romanized (`mama`, `thiyenawa`), 1 mixed. Models must
   handle both; string matching across recordings must transliterate first
   (`scripts/script_normalize.py`).
5. **Null timestamps.** 868 tokens in 9 recordings have `start`/`end` null:
   R0073 (217), R0103 (172), R0075 (161), R0045 (114), R0035 (85), R0101 (80),
   R0068 (25), R0046 (13), R0043 (1). Text and labels are still usable;
   timing analyses must skip them.
6. **`switch_annotated` is unreliable.** At utterance boundaries where the
   language changes, annotators marked a switch about half the time (audit of
   the first 27 recordings, `docs/SWITCH_FIELD_AUDIT.md`). Use `switch`.
7. **No speaker field.** Tokens carry no speaker ID, so a switch at an
   utterance boundary may be a speaker change. Join with C3's RTTM for
   speaker-aware analysis.
8. **Shared scenario templates.** Many recordings are role-plays from the same
   script templates (bank call, clinic booking, insurance claim). Splitting by
   recording is necessary but not sufficient for testing on genuinely new
   topics.
9. **Bookkeeping.** 51 recordings are not yet in
   `manifests/recordings_current.csv`; R0062–R0065 lack the `J26DS313_`
   prefix in the source repo (canonicalised here); 7 usable recordings have
   no audio (R0007, R0050, R0052, R0053, R0057, R0060, R0061).
10. **Consent metadata** is incomplete in the source manifest (`FILL_IN`
    owner/consent fields). Confirm consent status before any use outside the
    project.

## ASR quality

The `asr/` layer and anything built on it inherits off-the-shelf Whisper's
performance on this audio, which is near-total failure: word error rate
≈ 0.99 under auto-detection, with the model producing only ~16% of the
expected words and recovering under 1% of them correctly. Forcing English
recovers more (~13–19% of words) but over-generates. Full analysis:
`results/c1_report.md` (Task 1) and `results/c1_noise_robustness.md` (Task 6).

> A `medium`-model re-run of Task 1 on all 61 recordings with audio is in
> progress; this section will be updated with its results.

Fine-tuning on a larger corpus (the TAF's 10–15 hour target) is the expected
fix; the current 3.2 hours is not enough.

## Intended use

- Training and evaluating token-level language identification and
  code-switch detection for Sinhala–English.
- Characterising code-mixing in Sri Lankan conversational speech.
- A development target for C2–C4 to build against the C1 output format.

**Not suitable for:** ASR training as-is (3.2 h, with the label issues above),
Tamil language identification (16 positive examples), or any claim about
spontaneous speech — most recordings are scripted role-plays.

## Rebuilding

```bash
python c1/scripts/audit_gold.py             # check the gold data first
python c1/scripts/build_dataset_release.py  # writes c1/release/
```

The build is deterministic given the same source commit: fastText trains
single-threaded with a fixed seed, and folds are assigned deterministically.
Provenance (both repo commits, model SHA-256, Tamil threshold) is recorded in
`MANIFEST.json`.
