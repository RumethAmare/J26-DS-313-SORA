# C3 demo app (PP1)

Offline Gradio app for Component 3. Double-click `run_app.bat` (Windows): it creates a Python
environment for the computer it runs on, downloads the model once (Hugging Face token needed once),
then runs fully offline at http://127.0.0.1:7860. It uses an NVIDIA GPU if present, otherwise the CPU.

| Tab | Function |
|---|---|
| 1 · Who spoke when | **F1.** Record (microphone) or upload a WAV file. Output: speaker timeline, talk time, overlap, speaker turns, and downloadable RTTM + JSON in the shared contract `{speaker_id, start, end, overlap_flag}` |
| 1b · Live (experimental) | Chunked live mode. Measured at ~39 % DER vs ~21 % for tab 1 (E020/E020b), so it is not part of the PP1 claim |
| 2 · Benchmark & error analysis | **F2.** Any clip of the C3 benchmark: ground truth vs model, DER split into missed / false alarm / wrong person, an error strip over time, per-speaker breakdown, and the whole-benchmark table |

Tab 2 needs the benchmark files from the dataset repository next to this folder
(`../processed/*.wav`, `../rttm_truth/*.rttm|.uem`, `../rttm_draft/*.rttm`).

Tests (no model download needed): `python -m pytest app/tests -q`
