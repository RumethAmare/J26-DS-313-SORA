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
- compares summarising the code-mixed transcript **directly** against
  **translate-first** (the CS-Sum comparison) — on dev, translate-first wins;
- has controllable output language (English / Sinhala / Singlish).

## Pipeline

    C1+C3 transcript
      └─► prepare: spoken digits → numerals, spelling variants        (normalize.py)
           └─► translate: fine-tuned NLLB-600M ──► clean_en, clean_si (train_translation.py)
                │   postprocess: numbers copied from the source,
                │   split Sinhala conjuncts re-joined                (normalize.py)
                └─► summarise clean_en: QLoRA Qwen2.5-3B ──► summary + action items
                     └─► summary → Sinhala (base NLLB)               (train_summary.py)
                          └─► <rid>.translation.json + <rid>.summary.json → C4

    run_c2.py runs the whole chain on a GPU; run_offline.py runs it on a laptop
    CPU with no internet (CTranslate2 int8 + llama.cpp GGUF Q4).

## Data lives elsewhere

Transcripts and gold C2 annotations come from the shared `SORA_Dataset`
repository, which this component treats as **read-only**. The corpus is still
being annotated, so every script reads whatever is there at run time; no
recording count is hard-coded.

    SORA_DATASET_ROOT   if unset, looked for next to this repo
                        (../SORA_Dataset, ../GitHub/SORA_Dataset)

## Status

| Step | What | State |
|---|---|---|
| 1 | Corpus loader (all annotation shapes), recording-level split | done |
| 2 | Normalizer (clitics, fillers, spoken digits, spelling variants) | done |
| 3 | Task A: NLLB-600M fine-tune, en and si, vs zero-shot baseline | done |
| 4 | Task B: QLoRA Qwen2.5-3B, direct vs translate-first | done |
| 5 | Full pipeline (`run_c2.py`), C4-format output | done |
| 6 | Offline CPU pipeline (`export_offline.py`, `run_offline.py`) | built, not yet run on a laptop |
| 7 | Test-split evaluation (`evaluate.py`) | built, run once at the end |
| — | COMET, LLM-judge + human rating, NLLB-1.3B, context-aware translation | PP2 |

### Dev results (5 recordings, 95 utterances)

| Task | Before training | After training |
|---|---|---|
| Singlish → English, chrF++ / BLEU | 36.6 / 13.8 | **56.3 / 33.6** |
| Singlish → Sinhala, chrF++ / BLEU | 28.1 / 6.6 | **51.1 / 22.3** |
| Summary ROUGE-L — direct | 25.3 | 34.0 |
| Summary ROUGE-L — translate-first | 25.3 | **43.7** |
| Action items F1 (intent) — translate-first | 16.0 | **32.0** |

Romanized-augmentation + normalisation ablation (en): 56.4 chrF++, no
significant gain over 56.3. Full numbers in `eval/*.json`; error analysis in
[`docs/C2_RESULTS.md`](docs/C2_RESULTS.md).

Earlier mT5-small result (en→si, 22 train / 5 test recordings, 50 epochs):
BLEU 9.59 at 100 % of training data — the reason for moving to NLLB, which is
pretrained on Sinhala.

## How to run

    python src/prepare_data.py                          # data/{train,dev,test}.jsonl
    python src/train_translation.py --target en         # GPU (Colab T4)
    python src/train_summary.py --input cascade         # GPU
    python src/run_c2.py --split dev --rid J26DS313_R0007   # full pipeline, GPU
    python src/run_offline.py --rid J26DS313_R0007      # full pipeline, laptop CPU, offline
    python src/evaluate.py                              # final test scores → eval/c2_report.md

## Metrics

| Output | Metric |
|---|---|
| clean_en / clean_si | chrF++ (headline), BLEU, COMET (Phase 3) |
| summary | ROUGE-L now; LLM-judge + human faithfulness rating (PP2) |
| action items | precision / recall / F1 on intent, and intent + owner |

chrF++ is the headline for translation rather than BLEU: Sinhala is
morphologically rich, and BLEU's word n-grams punish correct inflection
variants that chrF's character n-grams credit.

## Setup

    python -m pip install -r requirements.txt

## Layout

    docs/       output contract and annotation guidance
    src/        implementation
    tests/      unit tests
    eval/       scored results (committed; scores only, no transcript text)
    data/       train/dev/test splits (gitignored: un-redacted PII)
    predictions/, models/   regenerable outputs (gitignored)

## Invariants

1. **Never train on the test split.** The split is fixed by recording in
   `src/prepare_data.py` (test = the same 5 recordings C4 holds out); training
   scripts read only `train` and `dev`, and `evaluate.py` is run once per
   final model set.
2. **Never write into `SORA_Dataset`.** Read from it; write here.
3. **Gold summaries used for evaluation must be human-written** — never score
   against a machine-written summary.
