#!/usr/bin/env python3
"""
Pre-fetch Whisper checkpoints into the local cache, resiliently.

Two reasons this is a script rather than "just let faster-whisper download it":

1. **The Hub stalls here.** Downloads and even cache-revalidation calls hang
   indefinitely on this network — no error, no progress, 0% CPU. Observed
   twice: a `large-v3` fetch that created no files in 14 minutes, and a
   `WhisperModel()` load that blocked for minutes before the cached-first fix.
   An unattended hang inside the decode grid would stall every remaining
   config, so fetching is separated out, bounded by a timeout, and retried.

2. **§3 requires offline inference.** Models have to be in the cache before
   the offline entry point runs, so "make sure these are downloaded" is a real
   step someone has to be able to repeat.

Downloads go to the cache configured by src.env_bootstrap (/mnt/F, not the
root filesystem, which has under 3 GB free against large-v3's 3.1 GB).

Usage:
    python -m src.fetch_models                    # small, medium, large-v3
    python -m src.fetch_models --models large-v3
    python -m src.fetch_models --check            # report cache state, fetch nothing
"""
import argparse
import os
import shutil
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bound the per-request timeout BEFORE huggingface_hub is imported, so a stalled
# connection fails fast enough to retry instead of hanging forever.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")

from src import env_bootstrap  # noqa: F401,E402  — sets HF_HOME first

REPOS = {
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}
APPROX_GB = {"small": 0.5, "medium": 1.5, "large-v3": 3.1}
MAX_ATTEMPTS = 5

# Exactly what faster-whisper asks for. Matching it matters in both directions:
# a bare snapshot_download would try to pull the whole repo (PyTorch weights and
# all) and would also report a perfectly usable cache as MISSING, because the
# cache legitimately holds only these five files.
ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def cached_path(repo):
    """Local snapshot path if the files faster-whisper needs are cached."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo, local_files_only=True,
                                 allow_patterns=ALLOW_PATTERNS)
    except Exception:
        return None


# faster-whisper accepts a plain directory in place of a repo id, which is the
# escape hatch when huggingface_hub itself is the problem.
CT2_FILES = ["config.json", "preprocessor_config.json", "tokenizer.json"]
CT2_EITHER = ["vocabulary.json", "vocabulary.txt"]  # repos carry one or the other
CT2_WEIGHTS = "model.bin"


def direct_dir(name):
    return os.path.join(env_bootstrap.HF_HOME, "direct", f"faster-whisper-{name}")


def direct_complete(name):
    """A direct-download dir counts as usable once weights and configs are there."""
    d = direct_dir(name)
    if not os.path.isdir(d):
        return None
    if not os.path.exists(os.path.join(d, CT2_WEIGHTS)):
        return None
    if any(not os.path.exists(os.path.join(d, f)) for f in CT2_FILES):
        return None
    if not any(os.path.exists(os.path.join(d, f)) for f in CT2_EITHER):
        return None
    return d


def _stream_file(repo, filename, dest, required=True):
    """
    Download one file over plain HTTPS, resuming a partial if present.

    Deliberately not using huggingface_hub: snapshot_download hangs here with no
    error and nothing on disk, both with and without the Xet backend, while
    plain HTTPS to the identical URLs sustains ~640 KB/s.
    """
    import requests

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}

    with requests.get(url, headers=headers, stream=True, timeout=(20, 60),
                      allow_redirects=True) as resp:
        if resp.status_code == 404 and not required:
            return False
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {resp.status_code} for {filename}")
        if resp.status_code == 200:
            have = 0  # server ignored the range; start over
        total = int(resp.headers.get("Content-Length") or 0) + have

        mode = "ab" if have and resp.status_code == 206 else "wb"
        last = time.time()
        with open(tmp, mode) as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                have += len(chunk)
                if time.time() - last > 10:
                    pct = f"{have / total:.0%}" if total else "?"
                    print(f"      {filename}: {have / 1e6:.0f} MB {pct}", flush=True)
                    last = time.time()

    os.replace(tmp, dest)
    print(f"      {filename}: {os.path.getsize(dest) / 1e6:.0f} MB done", flush=True)
    return True


def fetch_direct(name, repo):
    """Full model download over plain HTTPS into a local directory."""
    dest_dir = direct_dir(name)
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[{name}] direct HTTPS download -> {dest_dir}", flush=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            for filename in CT2_FILES + [CT2_WEIGHTS]:
                target = os.path.join(dest_dir, filename)
                if os.path.exists(target):
                    continue
                _stream_file(repo, filename, target)
            if not any(os.path.exists(os.path.join(dest_dir, f)) for f in CT2_EITHER):
                for filename in CT2_EITHER:
                    if _stream_file(repo, filename,
                                    os.path.join(dest_dir, filename), required=False):
                        break
            if direct_complete(name):
                print(f"[{name}] direct download complete -> {dest_dir}")
                return True
            raise RuntimeError("files missing after download")
        except Exception as exc:
            print(f"[{name}] direct attempt {attempt} failed: "
                  f"{type(exc).__name__}: {str(exc)[:120]}", flush=True)
            if attempt < MAX_ATTEMPTS:
                backoff = min(60, 5 * 2 ** (attempt - 1))
                print(f"[{name}] retrying in {backoff}s (partial files resume)",
                      flush=True)
                time.sleep(backoff)
    return False


def fetch(name, repo, direct=False):
    from huggingface_hub import snapshot_download

    local = direct_complete(name)
    if local:
        print(f"[{name}] already downloaded -> {local}")
        return True

    existing = None if direct else cached_path(repo)
    if existing:
        print(f"[{name}] already cached -> {existing}")
        return True

    if direct:
        return fetch_direct(name, repo)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        free = shutil.disk_usage(env_bootstrap.HF_HOME).free / 1e9
        need = APPROX_GB.get(name, 1.0)
        if free < need + 1:
            print(f"[{name}] ABORT: {free:.1f} GB free, need ~{need:.1f} GB")
            return False

        print(f"[{name}] attempt {attempt}/{MAX_ATTEMPTS} "
              f"(~{need:.1f} GB, {free:.1f} GB free)...", flush=True)
        started = time.time()
        try:
            # max_workers=1: parallel workers made the stalls harder to see and
            # did not help throughput on this connection.
            path = snapshot_download(repo, max_workers=1,
                                     allow_patterns=ALLOW_PATTERNS)
            print(f"[{name}] done in {time.time() - started:.0f}s -> {path}", flush=True)
            return True
        except Exception as exc:
            print(f"[{name}] attempt {attempt} failed after "
                  f"{time.time() - started:.0f}s: {type(exc).__name__}: "
                  f"{str(exc)[:120]}", flush=True)
            if attempt < 2:
                print(f"[{name}] retrying in 5s", flush=True)
                time.sleep(5)
            else:
                # snapshot_download does not fail cleanly here — it hangs — so
                # stop retrying it and take the path that is known to work.
                print(f"[{name}] falling back to direct HTTPS download", flush=True)
                return fetch_direct(name, repo)

    print(f"[{name}] gave up after {MAX_ATTEMPTS} attempts.")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(REPOS),
                    help=f"comma-separated, from {list(REPOS)}")
    ap.add_argument("--check", action="store_true",
                    help="report what is cached and exit without downloading")
    ap.add_argument("--direct", action="store_true",
                    help="skip huggingface_hub entirely and download over plain "
                         "HTTPS into a local directory")
    args = ap.parse_args()

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    for n in names:
        if n not in REPOS:
            ap.error(f"unknown model {n!r}; choose from {list(REPOS)}")

    print(f"HF_HOME: {env_bootstrap.HF_HOME} "
          f"({shutil.disk_usage(env_bootstrap.HF_HOME).free / 1e9:.1f} GB free)\n")

    if args.check:
        for n in names:
            local = direct_complete(n)
            path = local or cached_path(REPOS[n])
            where = "direct" if local else ("cached" if path else "MISSING")
            print(f"  {n:10} {where}")
        return 0

    results = {n: fetch(n, REPOS[n], direct=args.direct) for n in names}
    missing = [n for n, ok in results.items() if not ok]
    print(f"\n{sum(results.values())}/{len(results)} models cached.")
    if missing:
        print(f"Still missing: {missing}. Re-run to resume — partial downloads "
              f"are kept, so a retry continues rather than starting over.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
