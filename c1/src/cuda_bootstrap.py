#!/usr/bin/env python3
"""
Make the venv's NVIDIA wheels visible to CTranslate2.

CTranslate2 links libcublas / libcudnn by soname and leaves finding them to the
dynamic loader. The pip wheels install them under site-packages/nvidia/*/lib,
which is not on the default search path, so faster-whisper fails at first
inference with:

    RuntimeError: Library libcublas.so.12 is not found or cannot be loaded

Exporting LD_LIBRARY_PATH would fix it, but only for whoever remembers to
export it. Preloading the shared objects with ctypes achieves the same thing
from inside the process: once loaded under their soname, the loader satisfies
CTranslate2's later lookups from what is already mapped. That keeps every entry
point runnable as a plain `python -m src.<script>` on any teammate's checkout.

Import this before faster_whisper, then call ensure_cuda_libs().
"""
import ctypes
import glob
import os
import sysconfig

# cublas depends on cublasLt, and cudnn's sub-libraries depend on libcudnn
# itself, so the base libraries load first.
_PRELOAD_ORDER = [
    "nvidia/cublas/lib/libcublasLt.so.*",
    "nvidia/cublas/lib/libcublas.so.*",
    "nvidia/cudnn/lib/libcudnn.so.*",
    "nvidia/cudnn/lib/libcudnn_*.so.*",
    "nvidia/cuda_nvrtc/lib/libnvrtc.so.*",
]

_loaded = False


def _site_packages():
    candidates = [sysconfig.get_paths().get("purelib"),
                  sysconfig.get_paths().get("platlib")]
    return [c for c in dict.fromkeys(candidates) if c and os.path.isdir(c)]


def ensure_cuda_libs(verbose=False):
    """
    Preload the NVIDIA shared objects. Idempotent; safe on CPU-only machines.

    Returns the number of libraries loaded. Zero is not an error — a system-wide
    CUDA install needs no help, and a CPU-only box has nothing to load.
    """
    global _loaded
    if _loaded:
        return 0

    count = 0
    for base in _site_packages():
        for pattern in _PRELOAD_ORDER:
            for path in sorted(glob.glob(os.path.join(base, pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    count += 1
                    if verbose:
                        print(f"  preloaded {os.path.basename(path)}")
                except OSError as exc:
                    if verbose:
                        print(f"  skipped {os.path.basename(path)}: {exc}")

    _loaded = True
    if verbose:
        print(f"ensure_cuda_libs: {count} libraries preloaded")
    return count


if __name__ == "__main__":
    n = ensure_cuda_libs(verbose=True)
    print(f"\n{n} NVIDIA libraries preloaded")
    try:
        import ctranslate2
        print(f"ctranslate2 {ctranslate2.__version__}, "
              f"CUDA devices: {ctranslate2.get_cuda_device_count()}")
    except Exception as exc:
        print(f"ctranslate2 probe failed: {type(exc).__name__}: {exc}")
