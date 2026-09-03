# Component 1 — Final Implementation Document
## Code-Switched ASR + Per-Token Language Identification
### J26-DS-313 | Owner: H.J.R.S. Amarasiri (IT23151888) | Target: last week of October 2026

---

## Phase 0 status: RESOLVED (Aug 30–Sep 5 2026)

Both open Phase 0 questions are now settled on real data (234 decodes,
26 recordings, 51.2 minutes, `faster-whisper`/CTranslate2 on an RTX 5050).
Full report: `phase0_decode_grid_report.pdf`; machine-readable results:
`eval/baseline_decisions.json`.

- **Decoding mode: forced-`en`.** Wins decisively, not narrowly — all
  three forced-`en` configs cluster under 0.87 WER; every auto-detect and
  forced-`si` config sits at 0.96–0.99, a ~10-point gap no model size
  closes. Root cause: under auto/si, 57–79% of words are deletions —
  Whisper's Sinhala decoder falls into repetition/temperature-fallback
  loops rather than transcribing, a capacity-independent collapse.
  Forcing `en` drops deletions to 5–14%; the model actually attempts
  transcription and the errors become fixable substitutions.
- **Model size: `medium`.** Beats `large-v3` (0.841 vs. 0.868 eval WER)
  and beats `small` (0.855) — **not monotonic with parameter count**.
  This contradicts the general "bigger model, better on low-resource
  languages" assumption used to justify a `large-v3` stretch goal earlier
  in this document — that assumption is now superseded by measurement.
  `large-v3`'s deletion rate collapses (5%) but substitutions (70%) and
  insertions (16%) rise — it hears more but mangles more of what it hears.
- **Script-mismatch hypothesis: rejected.** Folding hypothesis and
  reference into a common romanized space changes WER by at most 0.026
  points across all 9 configs — the high WER is genuine recognition
  failure, not a scoring artifact. Which script to standardize training
  data on can be chosen for convenience, not accuracy.
- **Deployment risk: lower than planned for.** Winning config runs at
  RTF 0.056 (~18× real-time); even `large-v3`/`en` hits RTF 0.173. All
  configs fit well under the 8GB target. The size-vs-deployability
  tension anticipated in §5 did not materialize for this decode mode.
- **Data volume: the binding constraint, confirmed in numbers.** 52
  minutes total (35.6 min train / 15.6 min eval) exist against a 15–20
  hour floor — see the updated Risk Flags (§12).
- **Two recordings need re-annotation before training use**: R0008 (gold
  transcript covers 56% of audio) and R0005, both flagged by the
  team's data-quality audit.

Baseline numbers to beat in Phase 2 (medium/forced-en, eval split):
**WER 0.841, CER 0.588, RTF 0.056.**

Sections below are updated to reflect these decisions; §5's model-size
comparison table is retained as the reasoning that motivated the test,
now marked resolved rather than open.

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

Two decisions were resolved empirically in Phase 0 (see banner above,
full results in §5):
- **Decoding mode: forced-`en`** — this is now the fixed foundation for
  Stage C's design (language ID operates on romanized/English-phonetic
  tokens, not native Sinhala Unicode).
- **Model size: `medium`.**

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

## 5. Model size and fine-tuning method decision — RESOLVED (Phase 0)

### 5.1 Whisper size comparison — measured, not hypothesized

Original plan (superseded): pick `small`/`medium` as a balance of
iteration speed and deployment fit, treat `large-v3` as a stretch-goal
comparison, on the general assumption that larger models gain most on
low-resource languages. **That assumption did not hold for this corpus.**
9-configuration decode grid, 26 recordings, 51.2 minutes, 7-recording
held-out eval split, `faster-whisper`/CTranslate2 float16 on an RTX 5050:

| Model (forced-`en`, the winning decode mode) | Eval WER | Fold CER | RTF | Resident VRAM |
|---|---|---|---|---|
| `small` (244M) | 0.855 | 0.632 | 0.056 | well under 8GB |
| **`medium` (769M) — winner** | **0.841** | **0.588** | **0.056** | well under 8GB |
| `large-v3` (1.55B) | 0.868 | 0.584 | 0.173 | ~5.7GB |

**`medium` wins outright and the relationship is not monotonic** — going
from `medium` to `large-v3` makes WER *worse* (0.841 → 0.868), not
better, despite doubling parameter count again. Error-composition
breakdown explains why: `large-v3`'s deletion rate collapses to 5% (best
of the three — it stops staying silent) but its substitution rate rises
to 70% and insertion rate to 16% (worst of the three) — it attempts to
transcribe more of the audio and gets more of that attempt wrong. WER
penalizes every one of those extra wrong/inserted tokens, and since
downstream components (C2, C3) consume individual word tokens, a
correct token — not a phonetically-close one — is what matters. This is
why WER, not CER, was used as the decision metric (`large-v3` actually
has the best CER of the three, which would have picked the wrong model).

Under auto-detect and forced-`si`, size barely moves the number at all
(all three sizes land between 0.96–0.99) — confirming the failure there
is a decode-mode collapse (§ Phase 0 banner), not a capacity problem no
amount of scaling fixes.

**Decision, now fixed: `medium` is the primary development and
deployment model.** The anticipated size-vs-deployability tension in the
original table below did not materialize for the winning decode mode —
RTF and VRAM headroom are comfortable at all three sizes, so the choice
was decided on accuracy alone.

**`large-v3` stretch-goal status, revised:** still worth an optional
Weeks 6-7 comparison run if ahead of schedule, but reframed — not as "is
bigger better" (Phase 0 already answered no, for this task and this
decode mode) but as "does fine-tuning change this ranking," since all
three Phase 0 numbers are zero-shot.

*Original hypothesis table, retained for the reasoning trail:*

| | `small` (~244M) | `medium` (~769M) | `large-v3` (~1.55B) |
|---|---|---|---|
| Zero-shot accuracy (assumed) | Weakest | Meaningful step up | Best (assumption rejected by Phase 0) |
| Fine-tuning signal clarity | Largest relative gain | Moderate | Smallest relative gain |
| Overfitting risk at 15-20h data | Lowest | Moderate | Highest |
| LoRA fits 8GB GPU comfortably | Yes | Yes | Tight — QLoRA effectively required |
| Training/iteration speed | Fastest | Moderate | Slowest |
| Quantized deployment footprint (assumed) | Best fit | Good fit | Largest (not confirmed as a real constraint by Phase 0) |

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

### 6.2 Volume and split — actual vs. target (updated post-Phase 0)

| Pool | Target | **Actual, as of Phase 0 (Sep 5 2026)** | Role |
|---|---|---|---|
| Held-out evaluation set | ~1 hour | **15.6 min, 7 recordings** — locked, seeded, stratified by script group | Reserved exclusively for evaluation — never trained on, at any phase, by any model size or decode mode |
| Training pool | 15-20 hours minimum, 30+ preferred | **35.6 min, 19 recordings** | ASR fine-tuning + CRF training |
| **Total scorable corpus** | — | **51.2 min, 26 of 27 gold-transcribed recordings** (R0007 excluded — no matching audio file) | — |
| Noisy test subset | Slice of either pool | Not yet built | Clean-vs-noisy robustness comparison (Phase 3) |

**The training pool is at roughly 3% of its 15-20 hour floor.** This is
now the project's single largest risk (see §12) — every WER/CER number
in the Phase 0 report is a zero-shot baseline on a tiny corpus, not yet
informative about fine-tuned performance, and Phase 2 cannot proceed
meaningfully until this gap closes substantially.

**Two recordings need re-annotation before any training use**:
`J26DS313_R0008` (gold transcript covers only 56% of its audio timeline —
its WER score of 4.227 measures the incomplete annotation, not the
model) and `J26DS313_R0005`. Both are excluded from the Phase 0
corpus-level averages by the team's data-quality audit but must be fixed,
not just excluded, before they can contribute to the training pool.

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

### Phase 0 (Week 0) — ✅ COMPLETE (Aug 30–Sep 5 2026)
Ran Whisper across {auto-detect, forced-en, forced-si} × {small, medium,
large-v3} against 26 recordings (51.2 min), scored on a locked 7-recording
eval split (234 decodes total). **Result: forced-`en` decoding, `medium`
model, baseline eval WER 0.841 / CER 0.588 / RTF 0.056** — see the Phase 0
status banner at the top of this document and full report
`phase0_decode_grid_report.pdf` / `eval/baseline_decisions.json`.

### Phase 1 (Weeks 1-3) — Scaffolding + data collection (parallel)
Team-wide recording push targeting 15-20+ validated training hours;
track hours/week from day one. **Starting point confirmed by Phase 0:
35.6 min (19 recordings) currently in the training pool — roughly 3% of
the floor.** Re-annotate R0008 and R0005 before they can count toward
this pool. In parallel: build `preprocess.py`, `lid_features.py`,
`assemble_tokens.py`; the `manifest.csv` split policy is already locked
from Phase 0 (7 recordings as `eval`, seeded and stratified by script
group — do not re-shuffle this split later).

**Gate before Phase 2 begins:** do not start fine-tuning on a training
pool under roughly 5 hours (per the risk flag already in §12) — with the
current ~36 minutes, Phase 2's start date is contingent on Phase 1's
collection pace, not fixed to the calendar week.

### Phase 2 (Weeks 4-5) — First fine-tune
LoRA fine-tune **`medium`, forced-`en`** (both now fixed by Phase 0,
not open choices) on the `train` split. Evaluate WER/CER on the
untouched `eval` split against the **Phase 0 baseline of WER 0.841 / CER
0.588** — this is the number Phase 2 must beat for fine-tuning to be
demonstrated as worthwhile, not just "any WER number." Train the CRF
tagger on real token labels (not placeholder dictionary). Wire Stages A-D
end-to-end; verify schema-valid output.

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
| 0 | Aug 30 – Sep 5 | Baseline decisions | ✅ Done — forced-en / medium confirmed, WER 0.841 baseline recorded |
| 1-3 | Sep 6 – Sep 26 | Scaffolding + data collection | 15-20+ validated training hours; split policy locked; scaffolding code ready |
| 4-5 | Sep 27 – Oct 10 | First fine-tune | Fine-tuned checkpoint; CRF tagger; before/after WER/CER & F1 |
| 6-7 | Oct 11 – Oct 24 | Iterate + robustness | Possible retrain; clean-vs-noisy comparison; optional large-v3 stretch run |
| 8 | Oct 25 – Oct 31 | Quantization | Quantized model; latency/memory/size report |
| 9 | Nov 1 – Nov 7 (buffer) | Integration & writeup | Final report; offline-verified inference entry point |

---

## 12. Risk flags

1. **[TOP RISK, confirmed not hypothetical] Training pool is at 35.6
   minutes against a 15-20 hour floor — roughly 3%.** This is no longer
   a scheduling risk to watch for; it is the project's current, measured
   state as of Sep 5 2026. Weeks 1-3 collection pace must be tracked
   weekly starting immediately, and any week that doesn't show
   substantial growth toward the floor should be escalated to
   supervisors that week, not accumulated silently. Fewer than ~5 hours
   of `train`-split data by the Phase 2 start date (per the original
   plan) should delay Phase 2's start rather than force a fine-tune on
   an unstable, overfit-prone sample.
2. **Two recordings (R0008, R0005) need re-annotation before counting
   toward the training pool** — do not let partial/incomplete
   transcripts silently inflate the training-hours count without fixing
   the underlying annotation first.
3. Phase 2's fine-tuned WER must be compared against the **confirmed
   Phase 0 baseline of 0.841 (medium, forced-en)** — a fine-tune result
   worse than or barely better than this number, given how little
   training data is likely available by Phase 2, is a plausible outcome
   to prepare for and report honestly, not a failure to hide.
4. Any attempt to train on `eval`-split data (the locked 7-recording,
   15.6-minute set) — hard stop in code, not a warning.
5. Citing the unverified CRF/RF percentage figures (§10) in any
   submitted document before that section's outstanding task is
   resolved.

**Resolved, no longer risks:** decoding-mode uncertainty and model-size
uncertainty (both settled by Phase 0, §5); deployment/quantization
fitting the 8GB target (Phase 0 showed comfortable headroom at all sizes
tested); script-mismatch inflating WER (tested and rejected, cost ≤0.026
WER points).
