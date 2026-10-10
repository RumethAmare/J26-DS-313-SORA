# C4 Architecture — Offline PII Detection and Redaction

J26-DS-313 · Component 4 · O. S. Jayathilaka (IT23247390)

## Position in the SORA system

```mermaid
flowchart LR
    A[Audio] --> C1[C1<br/>Singlish ASR<br/>transcript]
    C1 --> C3[C3<br/>speakers]
    C1 --> C2[C2<br/>clean_en / clean_si<br/>+ summary]
    C2 --> C4[C4<br/>PII detection<br/>and redaction]
    C1 -. transcript .-> C4
    C4 --> OUT[Shareable redacted<br/>record + summary]
    C4 --> MAP[(Re-identification map<br/>local, access-controlled)]
```

**If C4 is removed,** the record cannot be shared: every name, NIC, phone
number and account in it stays exposed.

## Input and output contract

| | What | Format |
|---|---|---|
| **Input** | C2 single-language record and summary; C1 transcript when present | `<rid>.translation.json`, `<rid>.summary.json`, `<rid>.transcript.json` |
| **Output 1** | Redacted shareable documents, one placeholder per entity everywhere | `out/<rid>.redacted.json` |
| **Output 2** | Every detected span (metadata only, no surfaces) | `out/<rid>.detections.jsonl` |
| **Output 3** | Re-identification map, written only to git-ignored local storage | `reid_map/<rid>.reid.json` |

Run with: `python src/pipeline.py --recording <rid>`

## Inside C4

```mermaid
flowchart TB
    IN[C2 record + summary<br/>C1 transcript] --> LOAD[corpus.py<br/>reads every C2 format]
    LOAD --> RULES[rules.py + numerals.py<br/>NIC · PHONE · ACCOUNT · DOB · EMAIL<br/>spoken digits, number words]
    LOAD --> NER[transformer_ner.py<br/>XLM-RoBERTa, fine-tuned on CPU<br/>PERSON · ADDRESS · ORG]
    RULES --> HYB[Hybrid detector<br/>rules win on overlap]
    NER --> HYB
    HYB --> PROP[Propagation<br/>a known identifier is redacted<br/>wherever it recurs]
    PROP --> NAMES[Cross-script name propagation<br/>a found name is found again in the<br/>other script, case or inflection]
    NAMES --> RES[resolve.py<br/>cross-script entity resolution<br/>Fernando = ප්‍රනාන්දු]
    RES --> ROLE[roles.py<br/>private individual vs<br/>organisation representative]
    ROLE --> RED[redact.py<br/>one placeholder per entity]
    RED --> LEAK{Leak check}
    LEAK --> O1[redacted.json]
    LEAK --> O2[detections.jsonl]
    RED --> O3[(reid map)]
```

| Module | Responsibility | Requirement |
|---|---|---|
| `numerals.py` | decode / encode spoken Sinhala, romanised and English numbers | FR3, FR4 |
| `rules.py` | structured identifiers by format and keyword context | SO3, FR4, NFR7 |
| `transformer_ner.py` | fine-tuned XLM-RoBERTa for PERSON / ADDRESS / ORG (the final model) | SO4, FR2 |
| `ner.py` | spaCy model (comparison and fallback); hybrid detector combining model and rules | SO4, FR2 |
| `resolve.py` | cross-script entity resolution | SO5, FR5 (Contribution 2) |
| `roles.py` | person-role classification | FR7 |
| `redact.py` | consistent placeholders, propagation (exact and cross-script names), leak check, re-id map | FR6, FR8, NFR2, NFR6 |
| `pipeline.py` | end-to-end command; offline guard; latency and memory | FR1, FR9, NFR1, NFR4, NFR5 |
| `demo.py` | local web demo for the panel | — |

## Supporting tooling

| Module | Purpose |
|---|---|
| `synthetic.py` | synthetic train and held-out test data from Sri Lankan identifier formats |
| `validate.py` | schema and integrity checks (offsets, overlaps, cross-script links) |
| `evaluate.py` | strict-span evaluation per label, script and document (FR10) |
| `presidio_baseline.py` | Microsoft Presidio comparison (SO2) |
| `SORA_Dataset/scripts/c4/fix_c4_annotations.py` | deterministic repair of the C4 annotation layer |

## Technology choices

| Choice | Reason |
|---|---|
| **Rules for structured identifiers** | Fixed Sri Lankan formats (NIC, phone) are learnable without data and must be editable without retraining (NFR7). |
| **XLM-RoBERTa, fine-tuned on CPU** | Pre-trained on Sinhala; finds Sinhala names the spaCy CNN misses. Freezing the word-embedding table cut a training step from 75 s to 4 s, so it trains locally and no un-consented recording leaves the laptop. |
| **spaCy CNN (comparison)** | Trains in about 30 minutes; kept as the fallback and for the synthetic-data ablation. |
| **Transliteration + phonetic similarity** for linking | No parallel name data exists; a romaniser plus phonetic keys works offline with no training. |
| **Logistic regression** for roles | Small labelled set (203 real people); interpretable features. |
| **Python standard library web server** for the demo | No internet and no extra dependencies at PP1. |
