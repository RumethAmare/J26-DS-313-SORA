# Task 7 — Quantization, latency and footprint

Produced by `scripts/benchmark_latency.py` over 6 recordings on an RTX 5050 Laptop GPU (8GB), forced-English decode.

## Why accuracy is in this table

The plan specifies real-time factor, peak VRAM/RAM and on-disk size. Reporting only those would make every quantization look free — the table would improve monotonically as precision drops and the obvious read would be "ship int8 tiny". Quantization is a trade, so the cost side is measured too: **words recovered** on the same audio, which Task 6 established is the metric that can actually rank configurations here (WER is pinned near its ceiling and moves non-monotonically).

## Results

**On-disk size depends only on the model, not on `compute_type`.** CTranslate2 stores one set of weights and quantizes them *at load time*, so int8 and float16 rows for the same model show the same disk figure. Quantization buys memory and sometimes speed at runtime; shrinking the artifact on disk requires converting the model with `ct2-transformers-converter --quantization int8`, which is a separate step not performed here.

Every configuration was run **3 times**; RTF and words recovered are reported as mean ± standard deviation.

| model | device | compute type | disk | RTF | vs real-time | VRAM Δ | RAM Δ | words recovered | range |
|---|---|---|---|---|---|---|---|---|---|
| tiny | cuda | `float16` | 78 MB | 0.0397 ± 0.0043 | 25.2× | 259 MB | 768 MB | 134 ± 4 | 130–138 |
| tiny | cuda | `int8_float16` | 78 MB | 0.0394 ± 0.0032 | 25.4× | 106 MB | 53 MB | 119 ± 16 | 108–138 |
| tiny | cuda | `int8` | 78 MB | 0.0402 ± 0.0026 | 24.9× | 106 MB | 34 MB | 117 ± 12 | 107–130 |
| base | cuda | `float16` | 148 MB | 0.0487 ± 0.0034 | 20.5× | 266 MB | 35 MB | 150 ± 11 | 142–163 |
| base | cuda | `int8_float16` | 148 MB | 0.0444 ± 0.0048 | 22.5× | 138 MB | 34 MB | 117 ± 8 | 112–126 |
| base | cuda | `int8` | 148 MB | 0.0454 ± 0.0018 | 22.0× | 138 MB | 38 MB | 111 ± 28 | 93–144 |
| small | cuda | `float16` | 486 MB | 0.0303 ± 0.0031 | 33.0× | 714 MB | 59 MB | 167 ± 13 | 158–182 |
| small | cuda | `int8_float16` | 486 MB | 0.0436 ± 0.0018 | 22.9× | 389 MB | 281 MB | 157 ± 5 | 151–160 |
| small | cuda | `int8` | 486 MB | 0.0420 ± 0.0029 | 23.8× | 381 MB | 281 MB | 146 ± 20 | 124–162 |
| small | cpu | `int8` | 486 MB | 0.2125 ± 0.0205 | 4.7× | 0 MB | 1152 MB | 161 ± 8 | 152–169 |
| small | cpu | `float32` | 486 MB | 0.3637 ± 0.0277 | 2.7× | 0 MB | 1029 MB | 171 ± 11 | 159–179 |

### The accuracy column cannot rank compute types

This is the most important caveat in the table. Whisper decodes this audio with beam search over acoustics it barely models, so tiny numerical differences cascade into different segmentations and different hallucinations. Two independent single runs of the *same* tiny configurations gave 113 / 117 / 108 and 110 / 127 / 134 words recovered — **up to 24% apart with nothing changed**.

That is why repeats and ranges are shown. Where a configuration's range overlaps another's, the two are indistinguishable on this evidence, and any ordering between them is noise. The defensible claim from this table is that **quantization does not systematically cost accuracy** — not that any particular compute type is best.

RTF and memory, by contrast, are stable and rank cleanly.

### What the repeats do reveal: int8 costs accuracy

A single run could not separate the compute types. Three runs can, because the same ordering reproduces at **every** model size:

| model | `float16` | `int8_float16` | `int8` |
|---|---|---|---|
| tiny | 134 ± 4 | 119 ± 16 | 117 ± 12 |
| base | 150 ± 11 | 117 ± 8 | 111 ± 28 |
| small | 167 ± 13 | 157 ± 5 | 146 ± 20 |

float16 leads at tiny, base and small alike. Three independent replications of the same direction is much stronger evidence than any single gap, several of which have overlapping ranges on their own. **int8 quantization does cost recognition accuracy here** — roughly 10–25% of recovered words — and an earlier single-run pass that showed no cost was reading noise.

### int8 does not make this GPU faster

The other thing the timings settle: on this hardware int8 is **not** a speed optimisation. `small`/`float16` is the fastest configuration measured (RTF 0.0303), beating both int8 variants of the same model and even `tiny` at any precision. float16 maps onto the GPU's tensor cores, while int8 adds dequantisation work without a matching hardware path, and at `tiny` the run is dominated by VAD and decoding overhead rather than matrix multiplication.

So int8's benefit here is **memory alone** — it roughly halves VRAM (714 → 381 MB at `small`, 259 → 106 MB at `tiny`). That is the trade to reason about: less VRAM, no speed gain, some accuracy lost.

- Fastest GPU configuration: **small / `float16`** at RTF 0.0303 (33× real time).
- Leanest GPU configuration: **tiny / `int8_float16`** at 106 MB additional VRAM.
- **CPU-only deployment is viable.** `small` / `int8` on CPU runs at RTF 0.2125 (4.7× real time) using 1152 MB additional RAM, with word recovery indistinguishable from the GPU rows. On this corpus the pipeline does not require a GPU at all — which matters more for an offline deployment target than any of the VRAM figures above.

## fastText LID classifier

Task 3's quantization result, repeated here because it belongs in the deployment budget. Stock fastText allocates 2,000,000 hash buckets, sized for web-scale corpora; with 2,674 training tokens that produces a 400MB model (50MB quantized) holding a few thousand character n-grams in a nearly empty table.

| buckets | quantized size | CV accuracy (Latin-script) |
|---|---|---|
| 2,000,000 (stock) | 50.09 MB | 0.9135 |
| 200,000 | 5.09 MB | 0.9179 |
| **50,000 (shipped)** | **1.34 MB** | **0.9161** |
| 20,000 | 0.59 MB | 0.9062 |
| 5,000 | 0.22 MB | 0.9093 |

Accuracy varies by less than half a standard deviation (±0.05) across a 400× range of capacity, so the extra buckets were simply unused. The shipped model is **1.3 MB** — a 37× reduction against stock at no measured cost.

## Deployment reading

The whole C1 language stack — ASR plus the LID classifier — fits comfortably inside the offline, 8GB-GPU target. The binding constraint on this sub-objective is not compute, memory or model size; it is recognition accuracy on code-mixed Sinhala-English audio, which no quantization setting improves. These numbers establish that a fine-tuned model has room to grow into the deployment budget rather than needing to fight it.

## Limitations

- VRAM is read from `nvidia-smi`, which reports whole-device usage; figures are process-inclusive, not process-exact. torch is deliberately not a dependency of this project, so `torch.cuda.max_memory_allocated()` is unavailable.
- Peak memory is sampled once per recording rather than continuously, so a brief spike between samples would be missed.
- `medium` was not benchmarked: ~1.5GB at this host's ~500KB/s link. The size axis runs downward (tiny/base/small), which is the more relevant direction for offline deployment.
- RTF differences of a few percent remain within noise even across three repeats; only the larger gaps (GPU vs CPU, float16 vs int8 at `small`) are meaningful.
- Six recordings, ~965 reference words. The accuracy column is a small sample and should be read as direction, not magnitude.
