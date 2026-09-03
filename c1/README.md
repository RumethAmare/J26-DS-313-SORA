# Component 1 — Code-Switched ASR + Per-Token Language ID

J26-DS-313 · Owner: H.J.R.S. Amarasiri (IT23151888)

Given Sinhala-English code-mixed conversational audio, produce a token stream
where each token carries text, timestamps, a confidence score, a language label
(`SI` / `EN` / `NUM` / `OTHER`) and a switch marker. Feeds Component 2
(normalization/translation/summarization) and Component 3 (diarization fusion).

Design document: [`../C1_FINAL_IMPLEMENTATION.md`](../C1_FINAL_IMPLEMENTATION.md).

## Data lives elsewhere

Audio, gold annotations and the recording manifest come from the shared
`SORA_Dataset` repository, which this component treats as **read-only**. Nothing
here writes back to it — relabelling proposals are emitted as reviewable patches
under `data/` instead.

    SORA_DATASET_ROOT   default /mnt/F/SLIIT/Research/SORA_Dataset
    SORA_HF_HOME        default /mnt/F/SLIIT/Research/.hf_cache

The model cache is deliberately off the root filesystem, which has under 3 GB
free — `large-v3` alone is 3.1 GB.

## Setup

    VP=/mnt/F/SLIIT/Research/SORA_Dataset/.venv/bin/python
    $VP -m pip install -r requirements.txt

Phase 0 needs no PyTorch: faster-whisper runs on CTranslate2. The
`torch`/`peft`/`bitsandbytes` block (~5 GB) is listed but only installed when
Phase 2 opens. `ffmpeg` is a system dependency.

## Running Phase 0

    $VP -m src.script_fold --selftest    # Sinhala/Latin pairs must fold together
    $VP -m src.build_manifest            # lock the train/eval split (once)
    $VP -m src.split_guard --selftest    # leakage guard must raise
    $VP -m src.audit_labels              # label census -> proposed relabel patch
    $VP -m src.decode_compare            # {auto,en,si} x {small,medium,large-v3}
    $VP -m src.evaluate_asr              # WER/CER raw + folded -> eval/

`decode_compare` skips configs that already completed, so an interrupted run
resumes. Narrow it with `--sizes small,medium --modes auto,en`.

The full grid is several hours. Watch it from any terminal — progress is read
off the prediction files, not from the running process, so this works whether
the grid runs here, elsewhere, or has crashed:

    $VP -m src.progress --watch      # live bars, per-config ETA
    $VP -m src.progress --oneline    # one line, for a status bar

Models are fetched separately, because Hugging Face calls stall indefinitely on
this network and an unattended hang inside the grid would stall every remaining
config:

    $VP -m src.fetch_models --check    # what is cached
    $VP -m src.fetch_models            # fetch with bounded retries

## Layout

| Path | What |
|---|---|
| `src/sora_paths.py` | The only module that knows where `SORA_Dataset` is |
| `src/env_bootstrap.py` | Sets `HF_HOME`; **import before any HF library** |
| `src/cuda_bootstrap.py` | Preloads the venv's NVIDIA libs for CTranslate2 |
| `src/script_fold.py` | Sinhala↔Latin folding normalizer |
| `src/build_manifest.py` | Writes the locked `data/manifest.csv` split |
| `src/split_guard.py` | Refuses eval-split recordings in training |
| `src/fetch_models.py` | Resilient model prefetch (the Hub stalls here) |
| `src/decode_compare.py` | Phase 0 decode grid |
| `src/progress.py` | Live grid progress, read off disk |
| `src/evaluate_asr.py` | WER/CER, raw and folded, by split and script group |
| `src/audit_labels.py` | Label census and relabel proposal |
| `eval/baseline_decisions.json` | **Machine-readable decisions later phases read** |
| `eval/baseline_report.md` | Human-readable twin of the above |
| `docs/TOKEN_SCHEMA.md` | The `SI`/`EN`/`NUM`/`OTHER` schema and output contract |

## Two invariants

1. **Never train on the eval split.** `src/split_guard.py` raises rather than
   warns, and unknown recording ids are fatal too — an id with no recorded split
   cannot be shown to be safe. Every training entry point must call
   `assert_no_eval_leakage()` before it touches a model.
2. **Never write into `SORA_Dataset`.** It is shared, and teammates have
   in-progress work in it. Read from it; write here.
