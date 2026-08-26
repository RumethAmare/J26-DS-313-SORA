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
