#!/usr/bin/env python3
"""
cuda_env.py -- make CTranslate2 find the venv's CUDA libraries.

faster-whisper runs on CTranslate2, which dlopen()s libcublas / libcudnn by
bare soname at first GPU use. The CUDA libraries here come from pip
(nvidia-cublas-cu12, nvidia-cudnn-cu12) and land inside the virtualenv at
site-packages/nvidia/*/lib, which is not a path the dynamic loader searches.
The result is a run that loads the model fine and only then dies with
"Library libcublas.so.12 is not found or cannot be loaded" -- after the model
load has already succeeded, which makes it look like a GPU problem rather
than a path problem.

LD_LIBRARY_PATH is read by the loader at process start, so setting it from
inside a running Python process is not reliable. This module therefore
re-execs the interpreter once, with the variable set, before any CUDA use.

This machine also has a CUDA-13 driver against CUDA-12 userspace libraries,
which the C1 plan already flags as version-sensitive -- pinning the loader to
the venv's own copies keeps the run reproducible instead of depending on
whatever happens to be installed system-wide.
"""
import os
import sys
import glob

_GUARD = "_C1_CUDA_ENV_REEXEC"


def nvidia_lib_dirs():
    """Every site-packages/nvidia/*/lib directory for the running interpreter."""
    dirs = []
    for site_dir in sys.path:
        pattern = os.path.join(site_dir, "nvidia", "*", "lib")
        dirs.extend(d for d in glob.glob(pattern) if os.path.isdir(d))
    return sorted(set(dirs))


def ensure_cuda_libs():
    """Re-exec with LD_LIBRARY_PATH covering the venv's CUDA libs, once.

    No-op if there are no such directories (CPU-only install) or if this
    process is already the re-exec'd child.
    """
    if os.environ.get(_GUARD):
        return
    dirs = nvidia_lib_dirs()
    if not dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    present = set(p for p in current.split(os.pathsep) if p)
    missing = [d for d in dirs if d not in present]
    if not missing:
        return

    new_path = os.pathsep.join(missing + ([current] if current else []))
    env = dict(os.environ, LD_LIBRARY_PATH=new_path, **{_GUARD: "1"})
    os.execve(sys.executable, [sys.executable] + sys.argv, env)
