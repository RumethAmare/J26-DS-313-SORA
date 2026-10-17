# Task 7 — Quantization and latency

Produced by `scripts/benchmark_latency.py`; raw numbers in `c1_latency.json`.

Hardware: NVIDIA GeForce RTX 5050 Laptop GPU, 8151 MiB, AMD64 Family 25 Model 124 Stepping 0, AuthenticAMD (12 threads), Windows-11-10.0.26200-SP0. faster-whisper 1.2.1, CTranslate2 4.8.1.

## ASR (faster-whisper / CTranslate2)

5 recordings, 8.5 minutes of audio, decoded with the token-stream settings (beam 5, VAD, word timestamps, auto language). One untimed 10-second warm-up per configuration. `pipeline` = default temperature fallback (what C1 pays today); `fixed` = temperature 0 (deterministic; the fair basis for comparing compute types). See the script docstring for why both are needed.

| mode | model | compute type | device | load (s) | RTF | × real-time | peak GPU (MiB) | peak RAM (MiB) | disk (MB) | drift vs fp16 |
|---|---|---|---|---|---|---|---|---|---|---|
| pipeline | small | float16 | cuda | 0.9 | 0.336 | 3.0× | 1065 | 912 | 464 | ref |
| pipeline | small | int8_float16 | cuda | 1.0 | 0.453 | 2.2× | 678 | 1166 | 464 | 1.307 |
| pipeline | small | int8 | cuda | 0.9 | 0.425 | 2.4× | 725 | 1200 | 464 | 1.375 |
| pipeline | medium | float16 | cuda | 1.8 | 0.598 | 1.7× | 2764 | 2146 | 1460 | ref |
| pipeline | medium | int8_float16 | cuda | 2.5 | 0.695 | 1.4× | 1901 | 2172 | 1460 | 1.266 |
| pipeline | medium | int8 | cuda | 2.9 | 0.872 | 1.1× | 1867 | 2179 | 1460 | 1.115 |
| fixed | small | float16 | cuda | 0.8 | 0.092 | 10.9× | 840 | 1155 | 464 | ref |
| fixed | small | int8_float16 | cuda | 1.2 | 0.108 | 9.3× | 552 | 1367 | 464 | 0.208 |
| fixed | small | int8 | cuda | 1.2 | 0.108 | 9.2× | 588 | 1332 | 464 | 0.208 |
| fixed | medium | float16 | cuda | 2.1 | 0.131 | 7.6× | 2462 | 2168 | 1460 | ref |
| fixed | medium | int8_float16 | cuda | 4.0 | 0.209 | 4.8× | 1512 | 2193 | 1460 | 0.945 |
| fixed | medium | int8 | cuda | 3.5 | 0.208 | 4.8× | 1529 | 2236 | 1460 | 0.945 |
| fixed | small | int8 | cpu | 1.1 | 0.811 | 1.2× | — | 1296 | 464 | 0.250 |
| fixed | medium | int8 | cpu | 3.7 | 2.011 | 0.5× | — | 2246 | 1460 | 1.731 |

- **RTF** = wall time ÷ audio duration; below 1 is faster than real time.
- **drift** = WER of the transcript against the same model's GPU float16 transcript, after script normalisation. It isolates what quantization changes; clean-audio WER against gold (~0.99) is too saturated to show it.
- **disk** is the downloaded checkpoint (float16 weights). CTranslate2 converts to the requested compute type at load time, so int8 saves memory, not download size; an int8-at-rest copy would need re-conversion with `ct2-transformers-converter --quantization int8`.
- GPU memory is device-wide usage above a pre-load baseline (per-process accounting is unavailable under Windows WDDM).

## fastText LID (Task 3 model)

5-fold cross-validation folded by recording, 66 recordings, Latin-script tokens.

| | accuracy | size | predict time / token |
|---|---|---|---|
| full | 0.9644 ± 0.0173 | 9.95 MB | 4.5 µs |
| quantized | 0.9636 ± 0.0166 | 1.32 MB | 5.8 µs |

Per-fold accuracy change from quantizing: -0.0018, +0.0000, +0.0008, -0.0029, +0.0000.
