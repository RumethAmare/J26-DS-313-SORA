# Component 1 — Final Implementation Document
## Code-Switched ASR + Per-Token Language Identification
### J26-DS-313 | Owner: H.J.R.S. Amarasiri (IT23151888) | Target: last week of October 2026

---

## 1. Objective

Given raw Sinhala-English code-mixed conversational audio, output a
token-level stream where each token has: text, start/end timestamp,
confidence score, language label (SINHALA / ENGLISH / OTHER), and a
switch marker. Feeds Component 3 (diarization fusion) and Component 2
(normalization/translation/summarization). Must run fully offline at
inference time, be evaluated clean vs. noisy, and be quantized for
realistic on-device latency/memory.

---

## 2. Architecture

```
Audio (.wav, 16kHz mono)
        │
        ▼
[STAGE A] Preprocessing — VAD, segmentation, light noise reduction
        │
        ▼
[STAGE B] ASR — fine-tuned Whisper, word-level timestamps + confidence
        │
        ▼
[STAGE C] Per-token Language ID — CRF: SINHALA / ENGLISH / OTHER
        │   + derived switch markers
        ▼
[STAGE D] Assembly — merge B+C into final token JSON schema
        │
        ▼
Token-level output → feeds C2 and C3
```

Two decisions are resolved empirically in Phase 0, not assumed:
- **Decoding mode**: native Sinhala Unicode output vs. forced-`en`
  ("script collapse"/romanized) decoding — whichever wins on real WER/CER
  becomes the fixed foundation for Stage C's design.
- **Model size**: `small` / `medium` / `large-v3` — see §5.

---

## 3. Environment setup

`requirements.txt`:

```
openai-whisper
faster-whisper
transformers
peft
bitsandbytes
accelerate
datasets
jiwer
pandas
scikit-learn
sklearn-crfsuite
webrtcvad
silero-vad
noisereduce
librosa
audiomentations
ctranslate2
jsonlines
yt-dlp
```

System dependencies (not pip-installable): `ffmpeg`.

Force offline mode in the final deployed inference entry point only (not
training scripts):
```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

---

## 4. Tools by pipeline stage

| Stage | Tools | Purpose |
|---|---|---|
| Decoding-mode & size baseline (Phase 0) | `openai-whisper` / `faster-whisper`, `jiwer`, `pandas` | Run auto/forced-en/forced-si × small/medium/large-v3, compute WER/CER, pick winners |
| A — Preprocessing | `ffmpeg`, `silero-vad` / `webrtcvad`, `noisereduce`, `librosa` | Format conversion, VAD/segmentation, noise reduction |
| B — ASR fine-tuning | `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, Google Colab Pro (heavier runs) | LoRA/QLoRA fine-tuning of Whisper |
| C — Language ID | `sklearn-crfsuite`, `scikit-learn` | CRF token tagger, train/test split, precision/recall/F1 |
| D — Assembly | Plain Python (`json`/`jsonlines`) | Merge ASR + LID into corpus schema |
| Noise robustness (Phase 3) | `audiomentations` | Synthetic noise/reverb augmentation |
| Quantization (Phase 4) | `faster-whisper` / `ctranslate2` | INT8 export, latency/memory/size benchmark |
| Data sourcing | `yt-dlp`, `ffmpeg` | Supplementary audio extraction (see §7.3) |
| Data/process governance | Team's existing manifest CSV + `validate.py`, Git | Train/eval split tracking, corpus validation |

Optional but useful: `wandb` or TensorBoard for tracking training runs
across Phase 2 and Phase 3 retrains.

---

## 5. Model size and fine-tuning method decision

### 5.1 Whisper size comparison

| | `small` (~244M) | `medium` (~769M) | `large-v3` (~1.55B) |
|---|---|---|---|
| Zero-shot accuracy | Weakest, especially low-resource languages | Meaningful step up | Best, largest gain specifically on low-resource languages (general Whisper-family pattern; exact Sinhala numbers unverified — confirm via Phase 0) |
| Fine-tuning signal clarity | Largest relative gain from fine-tuning (more room to improve) | Moderate | Smallest relative gain — strong baseline already, harder to show fine-tuning worked |
| Overfitting risk at 15-20h data | Lowest | Moderate | Highest — more capacity than the data volume comfortably supports |
| LoRA fits 8GB GPU comfortably | Yes | Yes | Tight — QLoRA effectively required |
| Training/iteration speed | Fastest | Moderate | Slowest — directly costs retraining cycles in Weeks 4-7 |
| Quantized deployment footprint | Best fit for on-device target | Good fit | Largest, most likely to strain the real-time/on-device requirement |

**Decision:** use `small` or `medium` as the primary development and
deployment model — best balance of iteration speed, fine-tuning signal
clarity, and deployment fit against the TAF's own on-device constraint.
Treat `large-v3` (with QLoRA) as an optional Week 6-7 stretch-goal
comparison only if the primary pipeline is ahead of schedule — useful for
the final report ("bigger model vs. more fine-tuning data") but not the
default path.

**How to actually decide, not guess:** extend the Phase 0 baseline script
to run zero-shot WER/CER for `small`, `medium`, and `large-v3` on the real
held-out eval set before committing. This is the same script, same data,
three extra inference runs — costs almost nothing and replaces
size-comparison guesswork with a real number.

### 5.2 LoRA vs. QLoRA

| | LoRA | QLoRA |
|---|---|---|
| Base model precision | 16-bit, frozen | 4-bit, frozen |
| Memory footprint | Lower than full fine-tuning, still meaningful | Lowest — ~4x smaller base footprint |
| Training speed | Slightly faster | Slightly slower (quantize/dequantize overhead) |
| Accuracy vs. full fine-tuning | Very close | Very close, small additional gap from quantization noise |
| Needed for `small`/`medium` on 8GB | Usually sufficient alone | Optional extra headroom |
| Needed for `large-v3` on 8GB | Tight, may not fit comfortably | Effectively required |

Source: Dettmers et al., QLoRA (2023) — demonstrates 4-bit quantization +
low-rank adapters enabling large-model fine-tuning on a single consumer
GPU. **Decision:** use LoRA for `small`/`medium` (simpler, marginally
faster); switch to QLoRA only if memory pressure appears or if the
`large-v3` stretch goal is pursued.

---

## 6. Data requirements

### 6.1 Format
Utterance-level (audio, text) pairs for ASR fine-tuning: 16kHz mono
16-bit WAV, 1-30 second clips, sliced from full recordings using existing
`transcript.json` timestamps. Token-level (word, language tag, timestamp)
triples for the CRF tagger, drawn from the team's `tokens.jsonl` schema.
Text script (native Sinhala Unicode vs. romanized) must match the Phase 0
decoding-mode decision and be consistent across all training examples.

### 6.2 Volume and split

| Pool | Size | Role |
|---|---|---|
| Held-out evaluation set | ~1 hour currently validated | Reserved exclusively for evaluation — never trained on, at any phase, by any model size |
| Training pool | 15-20 hours minimum, 30+ preferred | ASR fine-tuning + CRF training |
| Noisy test subset | Slice of either pool, real or `audiomentations`-augmented | Clean-vs-noisy robustness comparison (Phase 3) |

### 6.3 Sources
- **Primary**: team's own recorded meetings via the validated 7-step
  workflow (record → Gemini draft → Claude correction → WhisperX
  alignment → Claude merge → validate.py), informed consent obtained.
- **Supplementary**: public Sinhala-English podcasts/panel discussions,
  including YouTube-sourced audio (`yt-dlp` extraction, then canonicalized
  via `ffmpeg` to 16kHz mono WAV and run through the same 7-step
  workflow). Lower consent tier than directly recorded participants —
  usable for ASR fine-tuning; flag separately before folding into any
  PII-annotated (C4) layer of the shared corpus. Note: downloading
  YouTube audio sits against YouTube's ToS despite being common research
  practice — worth a one-line disclosure to supervisors.
- **Synthetic augmentation**: TTS/audio mixing for speaker/overlap
  diversity, synthetic/real split disclosed, real recordings reserved for
  evaluation.

---

## 7. Repository structure

```
c1/
  requirements.txt
  data/
    manifest.csv              # split column: train | eval, locked once assigned
    corpus/
  eval/
    baseline_report.md
  src/
    decode_compare.py         # Phase 0: auto/forced-en/forced-si × model sizes
    preprocess.py              # Stage A
    finetune_whisper.py        # Stage B — re-runnable as train pool grows
    evaluate_asr.py             # WER/CER
    lid_features.py             # Stage C features
    train_lid_crf.py            # Stage C training
    evaluate_lid.py              # Token accuracy/F1
    assemble_tokens.py           # Stage D
    noise_eval.py                 # Phase 3
    quantize_export.py            # Phase 4
    fetch_youtube_audio.py         # yt-dlp wrapper + canonicalization
  models/
    whisper_finetuned/
    lid_crf.pkl
    whisper_quantized/
  reports/
    phase0_baseline.md
    phase2_first_finetune.md
    phase3_noise_robustness.md
    phase4_quantization.md
    final_report.md
```

Hard invariant to enforce in code, not just convention: `finetune_whisper.py`
and `train_lid_crf.py` must refuse to run if any `eval`-split recording
appears in the training file list.

---

## 8. Implementation phases

### Phase 0 (Week 0) — Baseline decisions
Run Whisper across {auto-detect, forced-en, forced-si} × {small, medium,
large-v3} against the held-out eval set. Compute WER/CER for all
combinations. Record the winning decoding mode and the working model size
in `eval/baseline_report.md`. This file is the single source of truth
every later phase reads from.

### Phase 1 (Weeks 1-3) — Scaffolding + data collection (parallel)
Team-wide recording push targeting 15-20+ validated training hours;
track hours/week from day one. In parallel: build `preprocess.py`,
`lid_features.py`, `assemble_tokens.py`; lock the `manifest.csv` split
policy with the Phase 0 eval recording(s) as `eval`, permanently.

### Phase 2 (Weeks 4-5) — First fine-tune
LoRA fine-tune the chosen model size on the `train` split. Evaluate
WER/CER on the untouched `eval` split, compared against Phase 0 zero-shot.
Train the CRF tagger on real token labels (not placeholder dictionary).
Wire Stages A-D end-to-end; verify schema-valid output.

### Phase 3 (Weeks 6-7) — Iterate + robustness
Retrain if more data has accumulated. Build/source a noisy eval variant;
report clean-vs-noisy WER/CER/F1. Optional: `large-v3` + QLoRA comparison
run if ahead of schedule.

### Phase 4 (Week 8) — Quantization
Export to CTranslate2 INT8. Benchmark latency, memory, model size on
target hardware. Confirm no meaningful WER/CER regression post-quantization.

### Phase 5 (Week 9, buffer) — Integration & writeup
Verify token JSON schema against C3's expected input. Enforce offline-mode
environment variables in the deployed entry point. Compile
`reports/final_report.md`: architecture, dataset stats, all comparison
tables, honest limitations.

---

## 9. Evaluation methodology

| Metric | Measures | Computed on |
|---|---|---|
| WER / CER | ASR accuracy | Held-out eval set, clean and noisy, before/after fine-tuning |
| Token accuracy / F1 | Language ID accuracy | Held-out eval set |
| Switch-point accuracy | Derived-marker correctness | Held-out eval set |
| Timestamp accuracy | Alignment quality | Held-out eval set vs. WhisperX-aligned reference |
| Latency, memory, model size | Deployability | Target hardware benchmark |

Every metric reported twice: before vs. after fine-tuning, and clean vs.
noisy — never a single unqualified number.

---

## 10. Citation and claims status (as of this document)

Kept here so the final report doesn't repeat sourcing mistakes made
earlier in this project's drafting process.

| Claim | Status | Use as |
|---|---|---|
| Whisper CER 32-52% on distinct-script code-switching vs. 7-28% same-script | ✅ Verified — CS-FLEURS (Yan et al., Interspeech 2025, arXiv:2509.14161) | General difficulty-class evidence. **Sinhala is confirmed NOT included in CS-FLEURS** — do not present as a Sinhala-specific number |
| Script collapse as a documented Whisper failure mode | ✅ Verified, correctly titled — Rahman, arXiv:2604.08786 (cite the specific version used; v1 and v2 report different collapse rates) | Motivates the Phase 0 decoding-mode test. Independent, non-peer-reviewed preprint — pair with CS-FLEURS rather than citing alone |
| CRF-based Sinhala-English word-level LID paper | ✅ Paper confirmed real — Smith & Thayasivam, IALP 2019, pp. 228-233 | Cite the paper and architecture. ⚠️ Exact F1 figure (previously stated as 90-94%) is UNVERIFIED — do not state a specific number until read from the IEEE Xplore PDF directly |
| Random Forest Sinhala-English word-level LID paper | ✅ Paper confirmed real — Shanmugalingam & Sumathipala, SCSE 2019, pp. 113-118 | Same treatment — cite paper, withhold the ~90.5% figure until verified from source |
| "No Sinhala-English code-mixed speech dataset with per-token tags exists" | ✅ Supported by absence in de Silva survey (arXiv:1906.02358) | Phrase as "consistent with de Silva's survey, no such dataset appears published" — not as an absolute, unattributed claim |
| Sinhala ASR papers (Gamage et al. ICON 2021; Karunathilaka et al. ICTer 2020) | ✅ Fully verified, correct citations | Cite as-is |
| QLoRA enabling large-model fine-tuning on consumer GPU | ✅ General capability verified — Dettmers et al., 2023 | Cite for the general claim; the "8GB" figure is this project's own target hardware, not an external benchmark result |

**Outstanding task before final submission:** read the two IEEE Xplore
PDFs (IALP 2019, SCSE 2019) directly to confirm the exact reported metric
(F1? accuracy? precision/recall?) and value, then upgrade the CRF/RF
citations in §10 and any slide/report content from hedged to fully cited.

---

## 11. Timeline

Start: 30 August 2026. Target: last week of October 2026.

| Week | Dates | Phase | Key deliverable |
|---|---|---|---|
| 0 | Aug 30 – Sep 5 | Baseline decisions | Decoding-mode + model-size WER/CER comparison; decisions recorded |
| 1-3 | Sep 6 – Sep 26 | Scaffolding + data collection | 15-20+ validated training hours; split policy locked; scaffolding code ready |
| 4-5 | Sep 27 – Oct 10 | First fine-tune | Fine-tuned checkpoint; CRF tagger; before/after WER/CER & F1 |
| 6-7 | Oct 11 – Oct 24 | Iterate + robustness | Possible retrain; clean-vs-noisy comparison; optional large-v3 stretch run |
| 8 | Oct 25 – Oct 31 | Quantization | Quantized model; latency/memory/size report |
| 9 | Nov 1 – Nov 7 (buffer) | Integration & writeup | Final report; offline-verified inference entry point |

---

## 12. Risk flags

- Data collection pace below target by end of Week 2 — escalate
  immediately; this is the least late-fixable risk.
- Fewer than ~5 hours of `train`-split data by Phase 2 start — flag before
  fine-tuning; risk of an unstable, overfit, misleading result.
- Phase 0 WER/CER far worse than the 32-52% CS-FLEURS distinct-script
  range — signal to reset final-accuracy expectations.
- Any attempt to train on `eval`-split data — hard stop, not a warning.
- Citing the unverified CRF/RF percentage figures in any submitted
  document before §10's outstanding task is resolved.
