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

## Setup

    python -m pip install -r requirements.txt

## Layout

    docs/       schema and rule specifications
    src/        implementation
    tests/      unit tests (rule layer is deterministic, so it is testable)
    data/       synthetic fixtures; real annotations are NOT copied here
    eval/       scored results
    reports/    written benchmark reports
