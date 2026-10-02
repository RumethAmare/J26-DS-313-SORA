# C3: Speaker diarization ("who spoke when") for Sinhala-English conversations

Component 3 of the SORA / Kathā pipeline. It turns a conversation recording into speaker turns
(RTTM), which the other components use to attribute speech.

**Model:** `pyannote/speaker-diarization-community-1` (pyannote.audio 4.0.7), running offline.
**Result on unseen public panel audio:** 7.50 % DER (protocol v1). See `EXPERIMENT_LOG.md`.

## Layout

| Path | Contents |
|---|---|
| `docs/` | Evaluation protocol v1, annotation rules |
| `scripts/01–05` | Audio conversion, zero-shot first pass, Audacity ⇄ RTTM conversion, scorer |
| `scripts/06–09`, `e0xx_run.py` | Experiment notebooks and scripts (Colab, T4 GPU) |
| `scripts/simconv.py` | Synthetic multi-speaker conversation generator |
| `demo/` | Gradio proof-of-concept app and its Colab notebook |
| `results/` | Score tables and summaries per experiment, demo screenshots |
| `C3_Console.py`, `RUN_C3.bat`, `C3_Scorer.html` | One-click convert-and-score tools |

Audio, annotations and model checkpoints are not stored here (participant consent and size).
The annotated dataset lives in the team dataset repository.

## Reproduce

1. `pip install -r requirements.txt` (a Hugging Face token with access to the pyannote models is
   read from the `HF_TOKEN` environment variable / Colab secret, never from code).
2. Score a set of system outputs: `python scripts/05_score.py <truth_dir> <hyp_dir> <out_dir>`.
3. Experiments: open the notebook in Colab with a T4 GPU and run the cells in order.
4. Demo: `demo/10_demo_app.ipynb`.
