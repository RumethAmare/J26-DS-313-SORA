# C1 Dataset Card — Sinhala-English Code-Switched Token Corpus

**Sub-objective C1**, project J26-DS-313 (SORA). Owner: H.J.R.S. Amarasiri
(IT23151888).

Built by `scripts/build_dataset_release.py`. Release at
`c1/release/c1_dataset/`; machine-readable statistics in its `MANIFEST.json`.

---

## What this is

Per-token language annotation for Sinhala-English code-mixed conversational
speech, with timestamps, switch-point markers, model predictions and quality
flags.

It ships in **two layers**, and the distinction matters more than anything
else in this document.

| layer | tokens | what it is | use it? |
|---|---|---|---|
| `gold/` | 5,818 (5,545 usable) | Human-annotated tokens enriched with C1 outputs | **Yes — this is the dataset** |
| `asr/` | 1,280 | End-to-end pipeline output | Only to inspect the pipeline |

The `asr/` layer is off-the-shelf faster-whisper output measured at
**~0.98–1.00 WER with 22% token coverage**. It is retained so the pipeline is
reproducible and inspectable. It is not training data, and every record in it
carries a `reliability` field saying so.

## Corpus statistics (usable recordings only)

- **5,545 tokens** across **25 recordings**, ~51 minutes of audio
- Language mix: **SI 3,320 (59.9%) · EN 2,147 (38.7%) · OTHER 78 (1.4%)**
- Class imbalance **SI:OTHER = 42.6 : 1**
- **1,850 switch points**, a switch rate of 33.4%
- Code-Mixing Index: mean **25.4**, median 27.3 (sd 17.4); 21.3% of utterances
  are monolingual, 52.2% score above 25

This is a genuinely, heavily code-mixed corpus — a third of all tokens sit at a
language boundary. That is the corpus's main value.

## Record schema

```json
{
  "recording": "J26DS313_R0015", "utt_id": "...", "tok_id": 1,
  "token": "Golden", "start": 1.43, "end": 1.91, "script": "latin",
  "gold":      {"lang": "EN", "switch": true, "switch_annotated": true},
  "predicted": {"lang": "EN", "switch": true, "lang_confidence": 0.65,
                "lang_method": "wordfreq_en", "provenance": "out_of_fold"},
  "tamil_flag":{"flagged": false, "reason": null, "margin": -0.31,
                "is_ground_truth_tamil": false},
  "quality":   {"usable_for_training": true, "exclusion_reason": null}
}
```

Two provenance guarantees worth stating explicitly:

- **`gold.switch` is derived from `gold.lang`, not copied from the annotation.**
  The annotated field is preserved separately as `switch_annotated` so the
  discrepancy is auditable. See limitation 4.
- **`predicted.lang` is out-of-fold** on usable recordings: each token was
  labelled by a fastText model trained without that token's recording. The
  shipped model (`models/lid_latin.ftz`) is trained on everything, so using it
  to label the release would leak its own training data into any metric
  computed from it.

---

## Known limitations

Ordered by how much damage each can do to work built on this data.

### 1. The corpus is ~51 minutes, against the TAF's 10–15 hour target

Everything below follows from this. 5,545 tokens is enough to build and test a
pipeline and to characterise the phenomenon; it is not enough to train a
model that generalises, and no result here should be read as a performance
claim for a deployable system.

### 2. `OTHER` does not mean what the schema says

`OTHER` is intended to denote **Tamil**, a national language of Sri Lanka
alongside Sinhala. The annotations do not use it that way. Of 78 `OTHER`
tokens:

- **58 are purely numeric** — annotation drift (limitation 5), not a language
- **11 are identifiers or English ordinals** — `8812304567V.`,
  `MASIT2024/0892.`, `LN2024NG00445`, `hasitha.94@gmail.com.`, `21st,`, `No.`,
  `1%`
- **9 are genuinely Tamil**

So the class is **88% not-Tamil**. Do not train or evaluate a Tamil detector
against the raw `OTHER` label. Curated ground truth for the 9 real Tamil
tokens is in `data/tamil_ground_truth.json`.

Related sub-findings:

- **There are zero Tamil-script tokens in the corpus.** Any approach relying
  on a Tamil Unicode range will never fire.
- **Most Tamil is written in Sinhala script.** Five of the nine Tamil tokens
  form one sentence in R0015 — `ඉදු ඔරුව පෙරිය ප්‍රචන ඉල්ලයි` =
  *idhu oru periya pirachanai illai* = "this is not a big problem".
- **The same word is labelled inconsistently across scripts.** Tamil *சரி*
  ("ok") appears as `සරි` tagged **SI** in R0013 and as `sari` tagged
  **OTHER** in R0016.
- **`unicode_sinhala` labels Sinhala-script Tamil as `SI` with confidence
  0.99** — confidently wrong, which is the worst failure mode for a confidence
  signal that C2/C3/C4 are meant to trust.

### 3. Two recordings are excluded (included in the release, flagged)

| recording | reason |
|---|---|
| `J26DS313_R0017` | every token tagged `EN`, including obvious romanized Sinhala (`mama`, `gedara`, `thiyenawa`) |
| `J26DS313_R0008` | token file structurally malformed — junk in timestamps |

Filter on `quality.usable_for_training` to get the clean 5,545-token set. They
are shipped rather than deleted so the corpus stays auditable.

R0017's problem is visible independently in this release: its out-of-fold LID
agreement is **0.454**, against a mean of **0.912** across usable recordings
and a worst usable case of 0.619. Its CMI of 7.2 (versus a corpus mean of
25.4) is likewise an artifact of everything being tagged one language.

### 4. The annotated `switch` field is unreliable at utterance boundaries

Where the language genuinely changes at an utterance boundary, gold marks
`switch=true` **102 times** and `switch=false` **100 times** — a coin flip.
A further 27 tokens are marked as switches with no language change at all.
Within utterances the field is fine (8 disagreements out of 137).

This release therefore **derives** `gold.switch` from `gold.lang` and keeps
the annotation as `switch_annotated`. Scoring switch-point F1 against the
annotated field would largely measure annotator inconsistency. Full analysis:
`docs/SWITCH_FIELD_AUDIT.md`.

### 5. Numeric tokens are tagged inconsistently

Corpus-wide, purely numeric tokens are tagged **EN 192 · OTHER 61 · SI 1**,
with the same short numeral appearing under different labels in different
recordings. This is annotation-guideline drift, not something a model can fix.
A going-forward rule is proposed in `docs/DATA_QUALITY_NOTES.md` (D.3).

### 6. Manifest status does not match validation

All 27 recordings are marked `VALIDATED` in
`SORA_Dataset/manifests/recordings_current.csv`, but **6 fail `validate.py`**
(R0008, R0019, R0020, R0021, R0025, R0026 — mostly UEM/RTTM running past the
audio). A corrected manifest is at `results/recordings_current_corrected.csv`
for a teammate to review and apply; the shared repo was not modified.

### 7. Gold transcripts mix two scripts

Nine of 27 recordings are 100% romanized Latin (`mama`, `thiyenawa`); most of
the rest are majority Sinhala Unicode. This is a corpus property, not an
error, but it means any string comparison must normalise script first —
`scripts/script_normalize.py` does this. Note that this normaliser is
deliberately lossy: it merges 22% of the vocabulary, mostly correct
cross-script unifications (162 of 181 merges) but 19 genuine losses such as
`karana`/`karanna`.

### 8. The ASR layer is not usable data

Off-the-shelf faster-whisper on this domain: **~0.996 normalized WER**, 22%
token coverage, and only 86 predicted switch points against gold's 2,012 — a
switch-recall ceiling of 4.3% before any downstream stage runs. The cause is
structural: Whisper picks **one** language acoustically per recording (25 of
26 → `si`) and decodes everything into that script, so 80.6% of predicted
tokens get labelled by a Unicode rule reading Whisper's script choice rather
than the language actually spoken.

---

## What was measured, and where

All headline metrics are measured on the **gold layer**, because the ASR layer
cannot support them (limitation 8).

| task | metric | result |
|---|---|---|
| 1 — ASR baseline | normalized WER | 0.9946 (raw-script 0.9983) |
| 3 — LID | token accuracy, 5-fold CV by recording | heuristic 0.8994 ± 0.032 → **hybrid 0.9318 ± 0.038** |
| 4 — switch points | switch F1 (positions) | heuristic 0.7895 → **hybrid 0.8703** |
| 5 — Tamil flagging | leave-one-out recall | 3/9 at 0.34% FP; 5/9 at 1.37% FP |
| 6 — noise | words recovered, clean → 0 dB | 397 → 198 (forced-English) |
| 7 — latency | RTF, `small`/float16 GPU | 0.0303 (33× real time) |

Cross-validation folds are by **recording**, never by token: tokens from one
conversation share speaker, topic and vocabulary, so a token-level split
leaks.

### Two measurement cautions

**Switch F1, not token accuracy, is the honest headline for code-switching.**
Switch detection amplifies LID errors roughly **2.4×** — one mislabelled token
mid-run creates *two* spurious switches — so a 3.4-point LID gain became an
8.2-point switch-F1 gain.

**Single-run ASR accuracy comparisons are unreliable on this corpus.** Two
identical runs of the same configuration differed by up to **24%** in words
recovered. Whisper beam-searches over acoustics it barely models, so numerical
noise cascades into different hallucinations. Any accuracy comparison here
needs repeats; Task 7 uses three.

---

## Intended use

**Appropriate:** characterising Sinhala-English code-switching; developing and
comparing per-token LID methods; switch-point and CMI analysis; a seed set for
a larger collection effort; as C2/C3/C4's input contract.

**Not appropriate:** training a deployable ASR or LID model; any claim about
Tamil detection performance (9 examples); benchmarking against other corpora
(too small, and the annotation issues above are not comparable); treating the
`asr/` layer as ground truth.

## Reproducing

```bash
python3 scripts/rerun_validate_all.py        # D.1 manifest audit
python3 scripts/evaluate_normalized.py       # Task 1 WER + ablation
python3 scripts/build_token_stream.py        # Task 2 ASR token stream
python3 scripts/train_lid_fasttext.py        # Task 3 LID, 5-fold CV
python3 scripts/switch_detection_eval.py     # Task 4 switch F1 + CMI
python3 scripts/otherlang_flag.py            # Task 5 Tamil flagging
python3 scripts/make_noisy_variants.py       # Task 6 SNR tiers
python3 scripts/noise_robustness_eval.py     #        robustness sweep
python3 scripts/benchmark_latency.py         # Task 7 latency/quantization
python3 scripts/build_dataset_release.py     # Task 8 this release
```

Environment caveats (offline HuggingFace, venv-local CUDA libraries,
unbuffered logging through the re-exec) are documented in
`docs/RUNNING_NOTES.md`.

## Priority fixes before the next release

1. Relabel the 58 numeric `OTHER` tokens under the D.3 rule.
2. Re-annotate R0017; repair R0008's token file.
3. Resolve the `switch` field convention at utterance boundaries and
   re-annotate accordingly.
4. Fix the `sari` cross-script inconsistency and audit for others like it.
5. Apply the corrected manifest statuses.
6. Grow the corpus toward the TAF's 10–15 hour target — the single change that
   would most improve every number in this document.
