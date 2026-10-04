# Task 7 — Quantization and latency

Produced by `scripts/benchmark_latency.py`; raw numbers in `c1_latency.json`.

Hardware: NVIDIA GeForce RTX 5050 Laptop GPU, 8151 MiB, AMD64 Family 25 Model 124 Stepping 0, AuthenticAMD (12 threads), Windows-11-10.0.26200-SP0. faster-whisper 1.2.1, CTranslate2 4.8.1.

## fastText LID (Task 3 model)

5-fold cross-validation folded by recording, 48 recordings, Latin-script tokens.

| | accuracy | size | predict time / token |
|---|---|---|---|
| full | 0.9535 ± 0.0270 | 9.81 MB | 4.2 µs |
| quantized | 0.9537 ± 0.0251 | 1.29 MB | 5.9 µs |

Per-fold accuracy change from quantizing: -0.0008, -0.0008, -0.0020, +0.0010, +0.0037.
