# Phase 0 — Baseline decisions and what the corpus actually supports

**Component 1 · J26-DS-313 · Week 0 (Aug 30 – Sep 5)**

Phase 0 exists to replace two guesses with measurements: which Whisper decoding
mode to build on, and which model size. It did that — **forced-English decoding
on Whisper `medium`** — and it also turned up four data problems that would
have quietly invalidated later phases, plus a size result that runs against
§5's stated expectation. The data findings matter as much as the decode
numbers.

Machine-readable results: [`../eval/baseline_decisions.json`](../eval/baseline_decisions.json).
Reproduce with `python -m src.decode_compare` then `python -m src.evaluate_asr`.

---

## 1. Decisions

The full 3×3 grid — {auto, en, si} × {small, medium, large-v3}, 9 configs, 26
recordings each — is complete.

| Decision | Value | Evidence |
|---|---|---|
| Decoding mode | **forced `en`** | eval folded WER 0.841–0.868 vs 0.962–0.994 for every `si`/`auto` config |
| Model size | **`medium`** | eval folded WER 0.841, beating both `small` (0.855) and `large-v3` (0.868) |
| Winning config | **`medium/en`** | eval folded WER **0.841**, raw WER 0.847, RTF 0.056 |

**Criterion:** eval-split micro WER under script-folded normalization. Folding
maps Sinhala and Latin into a common space, so the metric measures recognition
quality rather than which script the corpus happens to use — fine-tuning can
change the output script, but it cannot recover words the model never heard.

All nine configs, ranked by this criterion:

| config | eval folded WER | fold CER | RTF |
|---|---|---|---|
| **`medium/en`** | **0.841** | 0.588 | 0.056 |
| `small/en` | 0.855 | 0.632 | 0.056 |
| `large-v3/en` | 0.868 | 0.584 | 0.173 |
| `large-v3/auto` | 0.962 | 0.740 | 0.972 |
| `large-v3/si` | 0.974 | 0.731 | 0.939 |
| `medium/auto` | 0.993 | 0.804 | 0.597 |
| `medium/si` | 0.993 | 0.790 | 0.576 |
| `small/si` | 0.993 | 0.818 | 0.253 |
| `small/auto` | 0.994 | 0.819 | 0.248 |

Full table with raw/folded WER and CER: [`../eval/baseline_report.md`](../eval/baseline_report.md).

### Harness check

`small`/`auto` scores 0.9525 under the prior baseline's own (buggy) normalizer,
against its reported 0.9372 — reproduced within 0.016. The new harness reads
the same data as `SORA_Dataset/results/c1_report.md`, so the numbers below are
comparable to it.

---

## 2. The script-mismatch hypothesis was wrong

The corpus is script-split — 17 recordings write Sinhala in native Unicode, 8
romanized, 1 mixed — and the working hypothesis was that the prior 0.937 WER
was largely an artifact of scoring native-script hypotheses against romanized
references. `src/script_fold.py` was built to measure exactly that, folding both
sides into a common romanized space (`කොහොමද` and `kohomada` both → `kohomada`).

**Script mismatch costs 0.002–0.010 WER points.** Essentially nothing.

| config | mixed | native | romanized |
|---|---|---|---|
| `small/en` | 0.008 | 0.010 | 0.002 |
| `small/si` | 0.007 | 0.003 | 0.006 |
| `small/auto` | 0.007 | 0.002 | 0.005 |

The hypothesis is falsified, which is a useful result: it removes script
normalization from the list of things that might rescue the WER, and it means
the Phase 1 decision about which script to fine-tune toward can be made on
downstream convenience rather than on accuracy.

---

## 3. What the WER is actually made of

Splitting the errors into substitutions, deletions and insertions separates two
failure modes that both read as "WER ≈ 1.0":

| config | hit | sub | del | ins | RTF | reading |
|---|---|---|---|---|---|---|
| `small/en` | 0.188 | 0.674 | 0.138 | 0.164 | 0.056 | mishearing |
| `small/si` | 0.007 | 0.230 | 0.764 | 0.001 | 0.253 | **silent** |
| `small/auto` | 0.005 | 0.209 | 0.786 | 0.000 | 0.248 | **silent** |

Under `auto`, Whisper detects Sinhala on essentially every recording and then
emits almost nothing — 79% of reference words are never produced at all, and the
hit rate is 0.5%. Forcing `si` reproduces this, which confirms the cause:
`auto` and `si` are the same experiment, because auto-detect already chooses
Sinhala. Whisper-small's Sinhala decoder is the bottleneck, not language
detection.

Forcing `en` changes the failure mode entirely. The model transcribes — hit rate
0.188, deletions down to 0.138 — and it does so **4.4× faster** (RTF 0.056 vs
0.248), because it stops falling into the repetition and temperature-fallback
loops that the Sinhala path triggers.

The output is recognisably an English-phonetic rendering of code-mixed speech:

> **Gold** — Good morning Ceylon Electricity Board Kaduwela office eken, mama
> Chaminda. Oyata connection ekak oneda?
>
> **`small/en`** — Good morning, Ceylon Electricity Board, Cardwell Office. Can
> I come in? Who's the connection?

English spans survive; Sinhala spans become plausible-sounding English. Folded
CER (0.632) is far better than folded WER (0.976), which says the model is
approximating the phonetics while missing the word forms — the profile you would
expect fine-tuning to improve, and the reason `en` is the right foundation even
though its absolute WER is still poor.

---

## 3.5. Model size is not monotonic — and §5's premise doesn't hold here

§5 predicted `large-v3` would show "the largest gain specifically on
low-resource languages," hedged as unverified pending Phase 0. It didn't. On
forced-`en`, the only mode any model actually transcribes in:

| model | eval WER | fold CER | hit | sub | del | ins | RTF |
|---|---|---|---|---|---|---|---|
| small | 0.855 | 0.632 | 0.188 | 0.674 | 0.138 | 0.164 | 0.056 |
| **medium** | **0.841** | 0.588 | 0.237 | 0.638 | 0.125 | 0.131 | 0.056 |
| large-v3 | 0.868 | **0.584** | **0.249** | 0.700 | **0.052** | 0.163 | 0.173 |

`medium` beats `large-v3` by 2.7 WER points on the eval split, at a third of
the compute. large-v3 isn't simply worse, though — its error composition is a
genuine trade, not a regression: deletions collapse to 0.052 (it transcribes
almost everything) and CER is best of the three, but substitutions climb to
0.700 and insertions to 0.163. It hears more and mangles more of what it hears
into extra, wrong tokens. WER charges for every one of those insertions, and
for a per-token pipeline feeding C2/C3, a correct word token is what the
downstream components actually consume — so WER is the right metric to decide
on here, not CER.

Practically convenient: the model that wins on accuracy (`medium`) is also the
one that costs nothing over `small` at RTF 0.056, and both fit comfortably
inside the 8 GB target (large-v3 alone held ~5.7 GB resident at inference).
The size-vs-deployability tension §5 anticipated doesn't arise — `medium` wins
on both axes at once, so there's no trade to make.

`large-v3` + QLoRA remains available as the optional Week 6–7 stretch
comparison per §5, but Phase 0 gives no reason to expect it to beat a
fine-tuned `medium` on this task.

---

## 4. Four data problems found on the way

These were not the assignment, but each would have invalidated a later result.

### 4.1 The prior evaluation shredded its own references

`SORA_Dataset/scripts/c1/evaluate.py` normalizes with `[^\w\s]`. Python's `\w`
excludes Unicode category Mn, so every Sinhala dependent vowel sign and virama
was stripped before scoring — `ඔයාට කොහොමද` became `ඔය ට ක හ මද`. Its
native-script references were destroyed before comparison. `src/script_fold.py`
keeps the buggy version as `normalize_legacy` for reproduction only; real
scoring spares the Sinhala block.

### 4.2 Two recordings are systematically mislabelled

Checking each token's label against how the same word is labelled in every
*other* recording (no external lexicon needed — the corpus is its own
reference), 25 of 27 recordings disagree with the corpus on ≤7% of their
checkable tokens. Two do not:

| Recording | Rate | Direction | Examples |
|---|---|---|---|
| **R0017** | **55%** of 179 | Sinhala labelled `EN` | `mama`, `Oyata`, `ekak`, `oneda?`, `eken,` |
| **R0023** | **20%** of 128 | English labelled `SI` | `complaint`, `sir,`, `file`, `Phone`, `number` |

R0017's manifest agrees with the error — it declares `EN+OTHER+OTHER`, so
nothing cross-checked it — yet its transcript is plainly code-mixed. Its
romanized Sinhala was read as English at both the manifest and token level.
**R0023 is in the eval split**, so this directly affects the headline LID
number. Both must be corrected before any CRF training or LID evaluation:
a recording annotated on a different reading of the label set does not add
noise, it adds a competing definition.

### 4.3 `OTHER` had no learnable definition

Of 92 `OTHER` tokens, 81 were numerals, phone numbers, NICs or reference codes;
227 further numeric tokens sat in `EN` and 15 in `SI`. Hence the measured
`OTHER` F1 of **0.00** — a label problem, not a modelling one. The proposed
schema adds a deterministic `NUM` class and reserves `OTHER` for Tamil, moving
323 tokens (5.6%) and leaving 11 genuinely language-bearing `OTHER` tokens.

Consequence worth stating plainly: **there are zero Tamil-script tokens in the
corpus**, and 11 Tamil tokens total. `OTHER` cannot be learned from this data.
Until targeted Tamil is collected, its F1 should be reported as *not
measurable* rather than as 0.00.

Details and the output contract: [`../docs/TOKEN_SCHEMA.md`](../docs/TOKEN_SCHEMA.md).

### 4.4 Two references are incomplete

R0008 has 22 gold words for 62.5 s of audio (0.35 words/s, 56% of the timeline
uncovered) while Whisper found 111 — its WER of 5.045 measures the annotation,
not the model. R0005 has a 57% timeline gap. Both should be excluded from ASR
scoring until re-annotated. The rest of the corpus sits at a plausible
1.5–2.7 words/s. Reproduce with `python -m src.audit_references`.

---

## 5. The binding constraint

The corpus is **52.2 minutes** across 27 recordings. §6.2 of the implementation
document assumes ~1 hour held out for evaluation plus 15–20 hours for training.
There is currently less than the eval set alone.

The split is locked anyway — 19 train / 7 eval, stratified by script group,
seeded and reproducible (`data/manifest.csv`) — because Phase 2's "before vs.
after fine-tuning" claim is only meaningful if both sides are scored on the same
untouched recordings. New recordings join the train pool; the eval set grows
only by explicit decision.

Every number in this report should be read as provisional at this corpus size.
The Week 2 escalation trigger in §12.1 is the most important line in the plan.

---

## 6. Recommendations for Phase 1

1. **Fine-tune `medium` with forced-`en` decoding.** It wins on both axes Phase
   0 measured — lowest eval WER (0.841) and lowest compute (RTF 0.056, tied
   with `small`) — so there is no size-vs-deployability trade to make here.
2. **Correct R0017 and R0023 before any LID work.** Nothing downstream of the
   labels is trustworthy until this is done.
3. **Adopt SI/EN/NUM/OTHER**, or state explicitly that `OTHER` is a catch-all
   and stop reporting its F1.
4. **Exclude R0005 and R0008 from ASR scoring** pending re-annotation.
5. **Treat data collection as the critical path.** No modelling decision in this
   report is limited by modelling; they are all limited by 52 minutes of audio.

---

## 7. What Phase 1 inherits

Phase 0 is closed. Everything a later phase needs is machine-readable in
[`../eval/baseline_decisions.json`](../eval/baseline_decisions.json):
`winning_decode_mode: "en"`, `winning_model_size: "medium"`, plus every
config's full metrics for reference. `finetune_whisper.py` (Phase 2) should
read that file rather than hard-code the choice, so a future Phase 0 rerun
(more data, corrected labels) propagates without a code change.

Two operational notes worth carrying forward, unrelated to the modelling
result but likely to recur: this network resolves IPv6 for huggingface.co but
cannot route it, which hangs `huggingface_hub`, `requests`, and any library
built on them for minutes with no error — `src/env_bootstrap.py` forces IPv4
for Hugging Face traffic and should stay imported first in any script that
touches the Hub. And GPU load for `WhisperModel(device="cuda", ...)` fails at
first inference with a missing `libcublas.so.12` unless
`src/cuda_bootstrap.ensure_cuda_libs()` runs first — the pip CUDA wheels are
present in the venv but not on the default loader path.
