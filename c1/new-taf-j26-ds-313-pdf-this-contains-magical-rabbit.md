# C1 Implementation Plan — Code-Switched ASR + Per-Token Language-ID
J26-DS-313, owner: H.J.R.S. Amarasiri (IT23151888)

## Context

The TAF assigns C1 the sub-objective: from raw Sinhala-English code-mixed audio,
output the spoken words each tagged SINHALA/ENGLISH/OTHER(Tamil) with
timestamps, confidence, and switch-point markers — feeding C2/C3/C4
downstream. This session already produced two baselines against the repo at
`/mnt/F/SLIIT/Research/SORA_Dataset`:

- `scripts/c1/transcribe_baseline.py` + `evaluate.py`: off-the-shelf
  faster-whisper "small" scores ~93.7% micro WER / 83.4% CER against gold
  (26/27 recordings, 7,895 words) — off-the-shelf ASR essentially fails on
  this domain.
- `scripts/c1/lid_baseline.py`: a rule-based (not trained) SI/EN/OTHER tagger
  scored directly against gold tokens (bypassing ASR) gets 88.2% accuracy
  (90.1% excluding one mislabeled recording) but 0% on OTHER entirely.

This plan lays out what's left to build for the remaining 7 TAF task items,
scoped against a ~9-week runway to the proposal presentation (late October).
Proposal-stage bar = baseline experiments + clear methodology + small
proof-of-concept, not a fully trained final model — corpus size (~30-45 min
audio vs the TAF's own 10-15hr target) makes anything beyond that unrealistic
now regardless of effort spent.

Sequencing: fix data-quality issues first (Section D) since the WER/LID
numbers already produced are contaminated by them — every downstream task
inherits whatever's wrong here.

---

## D. Data-quality fixes (do first)

1. **Manifest status vs validate.py truth.** All 27 recordings are marked
   `VALIDATED` in `manifests/recordings_current.csv`, but 5
   (R0019/20/21/25/26) actually fail `validate.py`'s RTTM/UEM check. Re-run
   `validate.py <rid> --audio ...` for all 27, correct `status` to
   `ANNOTATED_DRAFT` for anything that doesn't pass. Deliverable:
   `scripts/c1/rerun_validate_all.py` + `results/validate_status_audit.csv`.
2. **R0017 mislabeling.** Every token in `annotations/c1/J26DS313_R0017.tokens.jsonl`
   is tagged EN, including obvious Sinhala words — a real gold-data bug, not a
   heuristic failure. Exclude R0017 from Task 3's train/eval sets with an
   explicit note in the dataset card (Section 8); full re-annotation is
   future work.
3. **Numeric-token tagging inconsistency.** Gold inconsistently tags numbers
   (e.g. `"8"`→EN, `"075"`→OTHER) — annotation-guideline drift, not something
   a smarter model fixes. Add one paragraph to `docs/DATA_CONTRACT.md`
   specifying a going-forward rule, and a `validate.py` check [7] that warns
   (not fails) on disagreement, so historical data isn't force-invalidated.
4. **No `requirements.txt`.** See Tooling section below.

---

## Task 1 — Off-the-shelf ASR baseline (mostly done, refine before quoting numbers)

- Add a **script-normalization pass** before WER scoring: gold transcripts
  mix romanized-Latin and native-Sinhala-Unicode script per recording, while
  Whisper always decodes in one auto-detected script — this inflates WER as
  a scoring artifact stacked on the genuine ASR failure. Transliterate both
  sides to one representation before scoring and report **raw-script WER
  alongside normalized WER** so the two failure modes aren't conflated.
- Run a small ablation: model size (small vs medium) × language forcing
  (auto-detect vs forced `si`) — cheap given only ~30-45 min of audio, and
  reads better in a proposal than a single number.
- Deliverable: updated `results/c1_report.md` with both WER variants and the
  ablation table, plus a paragraph motivating fine-tuning as future work.

## Task 2 — Token-level JSON schema (text, timestamps, confidence, lang, switch)

- Real gap: `DATA_CONTRACT.md`'s current schema
  (`utt_id,tok_id,token,lang,start,end,switch`) has **no confidence field**,
  even though the TAF explicitly asks for one. Extend to include
  `asr_confidence` (from faster-whisper's per-word `.probability`, via
  `word_timestamps=True`) and `lang_confidence` (from the LID
  method/classifier score) as two separate fields — don't conflate them.
- Build `scripts/c1/build_token_stream.py`: Whisper word-level output → LID
  labeling → switch computation → one
  `predictions/c1/token_stream/<rid>.tokens.jsonl` per recording in the
  extended schema. This is schema + glue code, fully achievable now.

## Task 3 — Train a SI/EN/OTHER classifier

Small-data reality: ~5,800 gold tokens total, ~92 for OTHER. Standard
transformer fine-tuning (XLM-R) would overfit badly and produce meaningless
numbers at this scale.

- **Recommended approach**: keep the current rule-based heuristic
  (`lid_baseline.py`) as "Method A" (already 88.2%/90.1%, zero training data
  needed — a legitimate floor for a genuinely low-resource pair). Add
  **"Method B": a fastText supervised classifier** for the ambiguous
  Latin-script cases only (Unicode rule still handles SI/Tamil-script
  deterministically) — fastText's subword n-grams suit small/short-text data,
  trains in seconds, quantizes trivially, and fits the offline/8GB-GPU
  constraint far better than transformer fine-tuning.
- **Evaluation**: stratified 5-fold cross-validation, folded by recording
  (not token, to avoid leakage), reporting mean±std F1 per class — a single
  train/test split at this scale is statistically unreliable and worth
  avoiding.
- Explicitly scope **XLM-R/mBERT fine-tuning as future work**, contingent on
  reaching the TAF's 10-15hr corpus target — state this directly in the
  proposal rather than attempting it prematurely.
- Deliverable: `scripts/c1/train_lid_fasttext.py` (data prep + k-fold CV),
  `scripts/c1/lid_hybrid.py` (rule-first, fastText-fallback predictor), and a
  heuristic-vs-fastText comparison table in `results/c1_lid_eval.json`.

## Task 4 — Switch-point detection

- Derived, not separately trained: a switch point is any adjacent token pair
  where `lang[i] != lang[i-1]` within an utterance (matches the gold
  `switch` field already in the schema — audit whether it's populated
  correctly as part of Section D).
- Build `scripts/c1/switch_detection_eval.py`: switch-point F1 (on
  *positions*, not raw token accuracy) comparing gold vs Task 3's predicted
  sequence, plus **CMI (code-mixing index)** per utterance as a
  corpus-characterization statistic worth reporting standalone.
- Fully achievable now — evaluation code only, no new training.

## Task 5 — Tamil/OTHER flagging

- Unicode rule (already in `lid_baseline.py`) handles native-script Tamil at
  ~100% precision — keep as-is.
- Real gap is **romanized** Tamil, which needs no labeled OTHER data if
  approached as anomaly detection instead of 3-way classification: build
  tiny character n-gram models (order 3-4, just counting, no training) for SI
  and EN separately, flag a Latin-script token as candidate-OTHER if its
  likelihood under *both* is below a percentile threshold. Combine with a
  small hand-curated gazetteer of common romanized-Tamil function words.
- Explicitly scope a trained supervised OTHER classifier as future work,
  contingent on synthetic/TTS augmentation (Task 6) growing OTHER examples
  into the hundreds.
- Deliverable: `scripts/c1/otherlang_flag.py` + honest precision/recall
  report against gold OTHER tokens (likely still low — that's fine to state
  as a documented limitation with a clear next step).

## Task 6 — Noise robustness testing

- Synthetically degrade a subset of `processed/audio/*.wav` at controlled
  SNR tiers (clean/15dB/5dB/0dB) using `audiomentations`, mixing in
  background noise (MUSAN or self-recorded ambience) and reverberation.
- Re-run Tasks 1/3/4's existing pipelines on each tier; plot WER, LID
  accuracy, switch-F1, and timestamp error vs SNR — a clean, presentable
  "robustness curve" needing no new labeled data at all.
- Deliverable: `scripts/c1/make_noisy_variants.py`,
  `scripts/c1/noise_robustness_eval.py`, `results/c1_noise_robustness.md`.
  **Best "cheap but real" experiment to prioritize if time gets tight.**

## Task 7 — Quantization and real-time offline measurement

- faster-whisper already runs on CTranslate2 with built-in quantization —
  sweep `compute_type` (int8 / int8_float16 / float16) × model size, measure
  real-time factor (wall-clock / audio duration), peak VRAM/RAM, and
  on-disk model size. One-line parameter changes to the existing baseline
  script, not new engineering.
- fastText's own `quantize()` gives near-free before/after size and accuracy
  numbers for Task 3's classifier.
- Do not attempt LoRA/QLoRA fine-tuning pre-proposal (needs the larger
  corpus) — report quantization/latency on the off-the-shelf models now,
  framed as "fine-tuning layers on top without changing the deployment
  budget."
- Deliverable: `scripts/c1/benchmark_latency.py`, `results/c1_latency_report.md`.

## Task 8 — Shared dataset layer (sequence last)

- `scripts/c1/build_dataset_release.py`: merges Task 2's token stream +
  Task 3's LID labels + Task 4's switch markers + Task 5's OTHER flags into
  one canonical `processed/c1_dataset/<rid>.jsonl` per recording, plus a
  corpus-level `MANIFEST.json` (token counts per language, CMI distribution,
  class-imbalance figures).
- Write `docs/C1_DATASET_CARD.md` documenting corpus size and **every known
  limitation from Section D explicitly** (R0017 exclusion, numeric-tagging
  inconsistency, OTHER scarcity, script inconsistency) — the expected, honest
  thing to do at proposal stage.

---

## Proposal-stage scope summary

| Task | Ship before proposal | Defer as future work |
|---|---|---|
| 1 | Ablation table (size × forced-lang × normalized WER) | Fine-tuning Whisper on domain data |
| 2 | Extended schema + `build_token_stream.py` | — |
| 3 | fastText hybrid, 5-fold CV, vs heuristic | XLM-R/mBERT fine-tuning (needs 10-15hr corpus) |
| 4 | Switch-F1 + CMI on existing predictions | — |
| 5 | Unicode + gazetteer + char-LM outlier detector | Trained OTHER classifier (needs augmented data) |
| 6 | SNR robustness curves | Noise-robust fine-tuning |
| 7 | RTF/VRAM/size table, off-the-shelf models | LoRA/QLoRA model's own latency numbers |
| 8 | Merged JSONL + dataset card with limitations documented | Full 10-15hr corpus release |

---

## Tooling — proposed `requirements.txt`

No dependency manifest exists yet (`.venv` has ad hoc installs). Propose:

```
# ASR
faster-whisper==1.2.1
ctranslate2==4.8.1
nvidia-cublas-cu12
nvidia-cudnn-cu12==9.*

# evaluation
jiwer==4.0.0

# LID
wordfreq==3.1.1
fasttext-wheel        # 'fasttext' has broken wheels on recent Python
langid==1.1.6         # optional cross-check baseline

# noise robustness (Task 6)
audiomentations
soundfile
scipy

# benchmarking (Task 7)
psutil

# shared
pandas
numpy
```

Deliberately omit `torch`/`transformers` given Task 3's fastText-first
recommendation — add only when/if post-proposal XLM-R work actually starts.
Freeze exact versions via `pip freeze > requirements-lock.txt` given the
CUDA-13-driver/CUDA-12-libs setup is version-sensitive.

## Verification

- Each script above should be run standalone and checked against the
  relevant `results/*.json`/`.md` file it produces, the same pattern already
  used for `evaluate.py` and `lid_baseline.py` this session.
- `validate.py` should be re-run for all 27 recordings after Section D fixes
  to confirm `status` in the manifest matches actual pass/fail.
- Cross-validation folds (Task 3) should be checked for no recording
  appearing in both train and test within the same fold.
