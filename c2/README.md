# Component 2 — Code-Mixed Normalization, Translation and Summarization

J26-DS-313 · Owner: A. I. Vithanage (IT23145948)

Given the speaker-attributed, Sinhala-English code-mixed transcript produced by
Components 1 and 3, produce:

1. a **clean single-language record** — a full word-for-word rendering of every
   utterance in English (`clean_en`) and in Sinhala (`clean_si`);
2. a **summary** of the conversation in either language;
3. **structured action items** (`intent`, `owner`, `receiver`, `deadline`),
   including indirect Sinhala commitments such as `"mama call karannam"` and
   `"balamu"`.

The output feeds Component 4, which redacts personal identifiers from it.
Output contract: [`docs/C2_SCHEMA.md`](docs/C2_SCHEMA.md).

## The contribution

The first conversational, spoken-domain Sinhala-English code-mixed parallel and
summarization corpus — every existing Sinhala-English parallel resource is
formal written or government text, and the CS-Sum benchmark (2025) does not
cover this pair — together with a normalization-plus-generation method that:

- handles intra-word morphological mixing (`meeting-eka`, `bankඑකට`);
- is robust to ASR errors, being trained and tested on noisy transcripts;
- summarises the code-mixed transcript **directly**, so indirect Sinhala
  commitments survive instead of being lost in an intermediate translation;
- has controllable output language (English / Sinhala / Singlish).

## Pipeline

    C1+C3 transcript ──► normalize ──► translate (mT5 / NLLB) ──► clean_en, clean_si
                             │
                             └────► summarize (QLoRA LLM) ──► summary + action items
                                        ▲
                     commitment detector (rules) — candidate action items

## Data lives elsewhere

Transcripts and gold C2 annotations come from the shared `SORA_Dataset`
repository, which this component treats as **read-only**. The corpus is still
being annotated, so every script reads whatever is there at run time; no
recording count is hard-coded.

    SORA_DATASET_ROOT   if unset, looked for next to this repo
                        (../SORA_Dataset, ../GitHub/SORA_Dataset)

## Status

| Phase | What | State |
|---|---|---|
| 0 | Corpus loader (all annotation shapes), locked split, data audit | next |
| 0 | Normalizer, indirect-commitment detector, copy baselines, chrF/BLEU | next |
| 1 | NLLB baseline translator; mT5-small en→si fine-tune and data-size curve | planned |
| 2 | QLoRA summarizer, action-item extraction, output-language control | planned |
| 3 | Noisy-vs-clean evaluation, COMET, LLM-judge + human faithfulness | planned |

Earlier mT5-small result (en→si, 22 train / 5 test recordings, 50 epochs):
BLEU 0.06 → 0.70 → 6.99 → 9.59 at 25/50/75/100 % of training data, still
rising — more recordings are the main lever.

## Metrics

| Output | Metric |
|---|---|
| clean_en / clean_si | chrF++ (headline), BLEU, COMET (Phase 3) |
| summary | LLM-judge + human faithfulness rating against baselines |
| action items | precision / recall / F1 on intent, owner, deadline |

chrF++ is the headline for translation rather than BLEU: Sinhala is
morphologically rich, and BLEU's word n-grams punish correct inflection
variants that chrF's character n-grams credit.

## Setup

    python -m pip install -r requirements.txt

## Layout

    docs/       output contract and annotation guidance
    src/        implementation
    tests/      unit tests
    eval/       scored results (committed)
    reports/    written reports

## Invariants

1. **Never train on the eval split.** The split will be locked in
   `data/split.csv` before any training; training entry points must refuse
   eval recordings.
2. **Never write into `SORA_Dataset`.** Read from it; write here.
3. **Gold summaries used for evaluation must be human-written** — never score
   against a machine-written summary.
