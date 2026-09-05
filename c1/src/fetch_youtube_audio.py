#!/usr/bin/env python3
"""
Supplementary audio sourcing from YouTube, per C1_FINAL_IMPLEMENTATION.md §6.3.

WHY THIS EXISTS
---------------
§6.3 lists public Sinhala-English podcasts/panel discussions as a supplementary
source once the team's own recordings plateau, explicitly flagged as a lower
consent tier than directly recorded participants: usable for ASR fine-tuning,
but kept out of any PII-annotated (C4) layer until the team decides otherwise.
It also notes downloading YouTube audio sits against YouTube's ToS despite
being common research practice — worth a one-line disclosure to supervisors,
not something to fetch silently.

This script therefore never writes into the shared corpus directly. It:
  1. Downloads best-audio via yt-dlp,
  2. Canonicalizes to 16 kHz mono 16-bit PCM WAV via ffmpeg (matching
     SORA_Dataset/docs/DATA_CONTRACT.md's contract for processed/audio/),
  3. Writes both the WAV and a metadata JSON (source URL, fetch time, and the
     consent-tier flag) into a LOCAL STAGING directory.

Admitting a staged file into SORA_Dataset — assigning it a real J26DS313_R####
id and running it through the team's 7-step workflow (record -> Gemini draft ->
Claude correction -> WhisperX alignment -> Claude merge -> validate.py) — is a
human decision, not something this script does.

Usage:
    python -m src.fetch_youtube_audio --url "https://youtube.com/watch?v=..."
    python -m src.fetch_youtube_audio --url URL1 --url URL2
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This network advertises IPv6 for most hosts but cannot route it, which hangs
# any Python HTTP client (requests/urllib3, and by extension yt-dlp) for
# minutes with no error -- confirmed against huggingface.co earlier in this
# project; the same fix applies to any external host. Must import before
# yt_dlp opens its first connection.
from src import env_bootstrap  # noqa: F401
from src import sora_paths

STAGING_DIR = os.path.join(sora_paths.data_dir(), "youtube_staging")

DISCLOSURE_NOTE = (
    "Sourced from YouTube via yt-dlp. Downloading YouTube audio sits against "
    "YouTube's ToS despite being common research practice (per "
    "C1_FINAL_IMPLEMENTATION.md §6.3) -- disclose to supervisors before "
    "wider use. Lower consent tier than directly recorded participants: "
    "usable for ASR fine-tuning; keep out of any C4 PII-annotated layer "
    "until the team explicitly clears it."
)


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr[-2000:]}")
    return result


def fetch_one(url, out_dir):
    """Download one URL, canonicalize its audio, write WAV + metadata JSON."""
    import yt_dlp

    with tempfile.TemporaryDirectory() as tmp:
        raw_template = os.path.join(tmp, "%(id)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": raw_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            raw_path = ydl.prepare_filename(info)

        video_id = info.get("id", "unknown")
        stem = f"YT_{video_id}"
        wav_path = os.path.join(out_dir, f"{stem}.wav")
        meta_path = os.path.join(out_dir, f"{stem}.meta.json")

        if os.path.exists(wav_path):
            print(f"  [{video_id}] already staged -> {wav_path}, skipping")
            return wav_path

        # Canonicalize: 16 kHz, mono, 16-bit PCM -- matches DATA_CONTRACT.md
        # exactly, so a staged file needs no further conversion once admitted.
        _run([
            "ffmpeg", "-y", "-i", raw_path,
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            wav_path,
        ])

    meta = {
        "video_id": video_id,
        "source_url": url,
        "webpage_url": info.get("webpage_url", url),
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "duration_s": info.get("duration"),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "consent_tier": "youtube_supplementary",
        "disclosure": DISCLOSURE_NOTE,
        "status": "staged -- not yet admitted to SORA_Dataset",
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    return wav_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", action="append", required=True,
                    help="a YouTube URL; repeat --url for multiple")
    ap.add_argument("--out-dir", default=STAGING_DIR)
    args = ap.parse_args()

    sora_paths.ensure_dir(args.out_dir)
    print(f"Staging to {args.out_dir}\n")
    print(DISCLOSURE_NOTE + "\n")

    staged = []
    for i, url in enumerate(args.url, 1):
        print(f"[{i}/{len(args.url)}] fetching {url}")
        try:
            path = fetch_one(url, args.out_dir)
            staged.append(path)
            print(f"    -> {path}")
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")

    print(f"\nStaged {len(staged)}/{len(args.url)} file(s) in {args.out_dir}")
    print("Not yet part of the corpus -- run the team's 7-step workflow to "
          "admit any of these into SORA_Dataset.")
    return 0 if len(staged) == len(args.url) else 1


if __name__ == "__main__":
    sys.exit(main())
