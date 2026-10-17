"""
STEP 1 - Convert raw recordings to the project's standard audio format.

Standard (from Dataset_Building_Guide, section 3): 16 kHz, mono, 16-bit WAV.
Every speech tool expects this. The originals are never touched.

Usage:
    python 01_convert_audio.py "../../Dataset" "../processed"

Needs ffmpeg installed and on your PATH.
"""

import subprocess
import sys
from pathlib import Path

AUDIO_EXTS = {".mp4", ".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg"}


def clip_id(path: Path) -> str:
    """Segment-1.m4a.mp4 -> Segment-1  (strip every extension)."""
    name = path.name
    while "." in name:
        name = name.rsplit(".", 1)[0]
    return name


def main(src_dir: str, out_dir: str) -> None:
    src = Path(src_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        print(f"No audio found in {src}")
        return

    total = 0.0
    for f in files:
        target = out / f"{clip_id(f)}.wav"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(f),
             "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
             "-c:a", "pcm_s16le", str(target)],
            check=True,
        )
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(target)]
        ))
        total += dur
        print(f"  {f.name}  ->  {target.name}   {dur:7.2f} s")

    print(f"\n{len(files)} files, {total:.2f} s total ({total / 60:.2f} min)")
    print(f"Written to: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
