# C1 Environment / Running Notes

Two environment issues on this host will hit every GPU script in the C1 plan
(Tasks 1, 6 and 7 especially). Both are handled by `c1/scripts/cuda_env.py`
and the `HF_HUB_OFFLINE` default in `transcribe_ablation.py`; this note
records *why*, so the next script written for C1 does not rediscover them.

## 1. No outbound HTTPS — HuggingFace calls hang, they do not fail

This machine has no working outbound HTTPS route (IPv6 connections sit in
`SYN-SENT` indefinitely). faster-whisper contacts HuggingFace to revalidate a
model **even when the weights are already in `~/.cache/huggingface`**, so a
run whose model is fully cached still hangs forever instead of erroring.

Diagnosis, if a GPU script appears to do nothing: `ss -tnp | grep python`
showing `SYN-SENT` on :443 is this problem, not a slow model load.

Fix: set `HF_HUB_OFFLINE=1` **before** `faster_whisper` is imported. Loading
`small` then takes 0.9s instead of hanging. This also matches C1's offline
deployment constraint, so it is the correct default rather than a workaround.

Consequence for the plan: the `medium` model is **not** in the local cache and
cannot be downloaded here, so Task 1's model-size ablation arm cannot run on
this host. Only `small` is available. The size axis needs either a machine
with network access or a manual copy of the medium weights into
`~/.cache/huggingface/hub/models--Systran--faster-whisper-medium`.

## 2. CUDA libraries live inside the venv, off the loader path

CTranslate2 `dlopen()`s `libcublas.so.12` / `libcudnn.so.9` by bare soname at
first GPU use. Those come from pip (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`)
and install to `.venv/lib/python3.12/site-packages/nvidia/*/lib`, which the
dynamic loader does not search.

The failure is misleading: the model **loads successfully**, and only the
first `transcribe()` call dies with

    RuntimeError: Library libcublas.so.12 is not found or cannot be loaded

which reads like a GPU/driver fault rather than a path fault.

`LD_LIBRARY_PATH` is read by the loader at process start, so exporting it from
inside Python does not reliably work. `cuda_env.ensure_cuda_libs()` therefore
re-execs the interpreter once with the variable set, guarded by an env flag so
it cannot loop. Call it at the top of any C1 script that touches the GPU,
before importing `faster_whisper`:

```python
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cuda_env
cuda_env.ensure_cuda_libs()
```

This host pairs a CUDA-13 driver with CUDA-12 userspace libraries, which the
plan already flags as version-sensitive; pinning the loader to the venv's own
copies keeps runs reproducible rather than dependent on system-wide installs.

## Throughput observed

`small` / float16 on the RTX 5050 (8GB): roughly 20s wall per recording over
51 minutes of audio, so about 9 minutes per full-corpus config and ~26 minutes
for the three language-forcing configs. Comfortably within budget for the
noise-robustness sweep in Task 6, which re-runs this pipeline per SNR tier.

## Running on Windows (from October 2026)

The scripts now also run natively on Windows; the WSL distro the earlier
runs used is gone.

- **Dataset path** is resolved by `scripts/corpus.py`: `$SORA_DATASET_ROOT`,
  else `/mnt/F/SLIIT/Research/SORA_Dataset`, else
  `F:/SLIIT/Research/SORA_Dataset`. No script hardcodes it any more.
- **venv:** `uv venv --python 3.12 c1/.venv` then
  `uv pip install --python c1/.venv/Scripts/python.exe -r c1/requirements.txt`.
- **CUDA DLLs:** on Windows the pip wheels put them in `nvidia/*/bin`, and
  `LoadLibrary` reads `PATH` at call time, so `cuda_env.ensure_cuda_libs()`
  adds those directories in-process instead of re-exec'ing.
- **PyAV must be 18.x.** faster-whisper 1.2.1 calls
  `av.open(metadata_errors=...)`, which PyAV 19 removed; the failure appears
  only at decode time. Now pinned in `requirements.txt`.
- **Network:** this host *does* reach HuggingFace (~3 MB/s), so `small` and
  `medium` are now both cached in `~/.cache/huggingface`. Task 1's medium
  ablation arm is no longer blocked.
- **Console encoding:** set `PYTHONIOENCODING=utf-8`, or any script that
  prints Sinhala dies with `UnicodeEncodeError` on the cp1252 console.
- `results/c1_noise_manifest.json` stores `/mnt/...` paths, so re-running
  `noise_robustness_eval.py` on Windows needs `make_noisy_variants.py` re-run
  first (the noisy WAVs are no longer in git; regeneration is seeded).

## 3. `cuda_env` re-exec drops interpreter flags (fixed)

`cuda_env.ensure_cuda_libs()` restarts the interpreter with
`os.execve(sys.executable, [sys.executable] + sys.argv, env)`. `sys.argv` does
**not** carry interpreter flags, so a job launched as `python3 -u script.py`
comes back as `python3 script.py` and becomes block-buffered.

The symptom is nasty: a long-running background job writes **nothing** to its
log until it exits, which is indistinguishable from a hung process. This bit
the Task 1 ablation and again the Task 6 noise sweep, and both times the
instinct was to check whether the process had stalled.

Diagnosis: `ps` shows the process alive and `nvidia-smi` shows GPU utilisation
high while the log file stays 0 bytes.

Fixed by setting `PYTHONUNBUFFERED=1` in the environment handed to `execve`,
which survives the re-exec regardless of how the caller invoked Python.

## Live transcription (2026-10-09)

The fine-tuned adapters run live through faster-whisper:

```powershell
cd c1\scripts
..\.venv\Scripts\python.exe export_ct2.py --run medium_v4_aug      # once per model
..\.venv\Scripts\python.exe export_ct2.py --run large-v3_v4_aug
..\.venv\Scripts\python.exe export_ct2.py --run small_v4_aug
..\.venv\Scripts\python.exe live_transcribe.py --model medium      # or large-v3, small
```

- `export_ct2.py` merges the LoRA adapter into the base model and converts it
  to CTranslate2 float16 in `c1/models/ct2/<run>/` (git-ignored). transformers 5
  does not save `preprocessor_config.json`; the script copies it from the base
  model, because large-v3 needs its 128 mel bands from that file.
- Decoding settings live in `scripts/ct2_decode.py` and are shared by
  `score_ct2.py` (test-set accuracy) and `live_transcribe.py`.
- Text appears after each pause (default 600 ms, `--silence-ms`), not word by
  word; provisional text shows while speaking (`--no-partials` to disable).
- `--input-file x.wav --realtime` streams a file through the same pipeline.
- Sessions save to `c1/predictions/live/`; `--save-audio` keeps the WAV.

### Web interface (2026-10-16)

```powershell
..\.venv\Scripts\python.exe live_web.py               # --model large-v3 to start with another
# then open http://localhost:8000 in Chrome or Edge on this laptop
```

- The browser records the microphone and streams 16 kHz audio to `live_web.py`,
  which runs the same pipeline as the terminal (`live_transcribe.LiveSession`).
  `eval_live.py` numbers therefore apply to the page as well.
- The page shows provisional grey text while you speak, then each utterance with words
  coloured SI/EN/OTHER. Hovering a word shows its time, alignment score and language
  confidence. You can switch the model (one is loaded at a time) and see latency per
  utterance.
- Stop saves `predictions/live/<run>_<time>.json` + `.tokens.jsonl`, and the page offers
  both as downloads.
- **Mic access needs `localhost` or https.** Opening the page from a phone on the same
  Wi-Fi (`--host 0.0.0.0`) shows the transcript but can't use that device's mic without
  a certificate.
- One session at a time; a second tab is told the server is busy.

### Word timestamps for C3 (2026-10-12)

Every token gets `start`/`end` from CTC forced alignment (`scripts/word_align.py`,
torchaudio's MMS aligner). It runs after the text is printed (~0.05–0.1 s per
utterance), so on-screen latency is unchanged. The first run downloads the
aligner weights (~1.2 GB) into the torch hub cache.

```powershell
# whole recording -> token stream in the shared schema (utt_id J26DS313_R0018_u001, ...)
..\.venv\Scripts\python.exe live_transcribe.py --model large-v3 --no-partials `
    --input-file F:\SLIIT\Research\SORA_Dataset\processed\audio\J26DS313_R0018.wav `
    --tokens-jsonl J26DS313_R0018.tokens.jsonl
```

- `--timing ctc` (default) | `both` (also Whisper's per-word `asr_confidence`,
  +0.4–0.7 s) | `whisper` | `none`.
- Accuracy against the gold token times: `python eval_timestamps.py` on clips →
  `results/c1_timestamps_report.md`; `eval_live.py` on whole recordings → `word_timestamps`.
