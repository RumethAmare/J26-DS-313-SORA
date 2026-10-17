# Component 4 — Offline PII Detection and Redaction

J26-DS-313 · Owner: O. S. Jayathilaka (IT23247390)

Given the single-language record and summary produced by Component 2, find every
personal identifier, replace each one with a stable placeholder, and emit the
redacted text plus a separately stored re-identification map. **This component
never processes audio** — it is a text-only privacy gate at the end of the
pipeline.

Proposal: `docs/` in this folder carries the schema and rule specifications.

## Why this is a separate component

C2 is a generative task scored on the quality of the text it writes. C4 is a
discriminative detection task scored on whether every identifier was found.
Different models, different losses, different failure costs — and privacy
behaviour has to stay independently auditable, so the two are not merged.

## The two contributions

1. **First PII detection and redaction system for Sinhala-English code-mixed
   text**, together with the first Sinhala-English PII-annotated dataset.
   Priority is claimed for the language pair only — PII redaction and offline
   operation are both established for other languages.
2. **Cross-script entity resolution for code-switched conversation**: a
   Latin-script and a Sinhala-script mention of one individual resolve to a
   single `entity_id` and therefore receive an identical redaction label in
   every output document. "Fernando" and "ප්‍රනාන්දු" share no characters, so
   string comparison cannot associate them; redacting one and missing the other
   leaves the person identifiable and voids the protection.

Role classification (private individual vs organisational representative) is a
schema field required by FR7. It is **not** a novelty claim.

## Label set

`PERSON` · `NIC` · `PHONE` · `ADDRESS` · `ORG` · `ACCOUNT` · `DOB`

`ORG` is detected but **not** redacted — organisation names are not personal
data. Everything else is redacted. `EMAIL` and `LOCATION` also occur in the
corpus and are carried through the pipeline; see `docs/PII_SCHEMA.md`.

## Structured vs unstructured

The split drives the whole architecture, and the baseline confirms it matters:

| Kind | Labels | Method | Why |
|---|---|---|---|
| Structured | NIC, PHONE, ACCOUNT, DOB | Deterministic rules | Fixed format, learnable without data |
| Unstructured | PERSON, ADDRESS, ORG | Fine-tuned multilingual encoder | No shared surface pattern |

Transcribed speech renders NIC and phone numbers as **spoken digit sequences**
(`"9 5 3 2 0 1 4 5 6 v"`, not `953201456V`). The rule layer is written against
that form first, because that is what the pipeline actually produces.

## Baseline to beat

Established on the shared corpus at 27 recordings / 309 annotated spans
(456 train, 130 eval utterances, 5 fully held-out recordings), strict span
matching — start_char, end_char and label must all match exactly:

| | Precision | Recall | F1 |
|---|---|---|---|
| Micro | 0.341 | 0.174 | **0.231** |
| Macro | 0.419 | 0.257 | 0.289 |

29 false positives against 71 false negatives: the model **under-detects**. For
a privacy component that is the dangerous direction, since a false negative is a
disclosure while a false positive only costs readability.

Targets: overall F1 above 0.60, PERSON above 0.50, cross-script linking accuracy
above 0.85. **Recall is the primary metric.**

## Data lives elsewhere

Annotations come from the shared `SORA_Dataset` repository, which this component
treats as **read-only**.

    SORA_DATASET_ROOT   default ../../SORA_Dataset

The corpus is still growing, so the rule layer, the cross-script resolver and
the evaluation harness are developed against **synthetic** Singlish text
generated from Sri Lankan identifier formats. Synthetic data is used for
development and unit tests only; every reported metric comes from the real
held-out recordings.

    python src/synthetic.py        # -> data/synthetic/train/ and data/synthetic/test/

| Split | Recordings | Used for |
|---|---|---|
| `train` | 50 | building rules, training models |
| `test` | 60 | scoring only; frozen, never tuned against |

The test split is held out from train in three ways: its sentence templates
never occur in train, its names/places/organisations never occur in train,
and it adds what real speech does — answers in a separate turn with no
keyword, fillers, lowercase, a dropped NIC `V`, a repeated digit, "zero
binduwai" restatements and ASR ordinal errors (`9rd`). Each noisy span
records the noise applied, so errors can be attributed.

Each synthetic recording follows the corpus layout (transcript, clean English,
clean Sinhala, summary), is marked `annotation_source: "synthetic"`, and must
pass `src/validate.py` before it is written. The set is deterministic for a
given seed.

## Running it

**Live demo for the panel** (offline, local web page):

    python src/demo.py                                   # open http://127.0.0.1:8765

**End to end — C2 output in, shareable output and re-identification map out**
(FR1, FR9, FR8):

    python src/pipeline.py --recording J26DS313_R0022 --offline-check --measure
    python src/pipeline.py --c2-dir <dir> --c1-dir <dir> --rid <recording id>

Writes `out/<rid>.redacted.json` and `out/<rid>.detections.jsonl` (both
git-ignored: system output on real data may still hold a missed identifier)
and `reid_map/<rid>.reid.json`. `--offline-check` blocks all network access
(NFR1); `--measure` reports latency and peak memory (NFR4, NFR5).

**Training** — complete real recordings only (C1, C2 and C4 present, at least
95% of spans matching their text), plus the synthetic train split; the five
held-out recordings are never used:

    python src/transformer_ner.py train --data both      # final XLM-RoBERTa model (~2 h, CPU, local only)
    python src/ner.py train --data both                  # spaCy model: comparison / fallback (~30 min)
    python src/roles.py train                            # person-role classifier (FR7)
    python src/roles.py evaluate

**Individual parts:**

    python src/redact.py --text "mage NIC eka 953201456V, number eka 0771234567"
    python src/redact.py --synthetic SYN_T0002 --all-docs --save-map
    python src/evaluate.py --source synthetic-test --system rules+propagation
    python src/evaluate.py --source real-eval --labels proposal --save
    python src/resolve.py --source real-eval --save      # cross-script linking + ablation
    python src/evaluate.py --source real-eval --system pipeline:xlmr_both --labels proposal
    python src/redact.py --synthetic SYN_T0002 --model both

Presidio comparison (SO2) — needs `python -m spacy download en_core_web_lg` once:

    python src/evaluate.py --source real-eval --system presidio --labels proposal --save

Models are written to the git-ignored `models/` and rebuilt from `ner.py` and
its seed; `--data real|synthetic|both` selects the training data.

One entity gets one placeholder across transcript, clean_en, clean_si and
summary. The re-identification map is written only to the git-ignored
`reid_map/`; `write_reid_map` refuses any path git would commit (NFR2).

## Setup

    python -m pip install -r requirements.txt

## Layout

    docs/       schema and rule specifications
    src/        implementation
    tests/      unit tests (rule layer is deterministic, so it is testable)
    data/       synthetic fixtures; real annotations are NOT copied here
    eval/       scored results
    reports/    written benchmark reports
