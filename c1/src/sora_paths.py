#!/usr/bin/env python3
"""
The one place that knows where the shared SORA_Dataset repository lives.

SORA_Dataset is a shared team repo holding the audio, annotations and manifest
this component reads from. It is treated as READ-ONLY: nothing in c1/ ever
writes back into it. Every other module imports its paths from here so that
dependency is one file rather than nine.

Override the location with the SORA_DATASET_ROOT environment variable.
"""
import json
import os

DEFAULT_SORA_ROOT = "/mnt/F/SLIIT/Research/SORA_Dataset"

# c1/src/sora_paths.py -> c1/
C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sora_root():
    root = os.environ.get("SORA_DATASET_ROOT", DEFAULT_SORA_ROOT)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"SORA_Dataset not found at {root!r}. "
            "Set SORA_DATASET_ROOT to point at the shared dataset repository."
        )
    return root


def audio_dir():
    """Canonical 16 kHz mono 16-bit WAVs, per docs/DATA_CONTRACT.md."""
    return os.path.join(sora_root(), "processed", "audio")


def gold_dir():
    """C1 gold annotations: <rid>.transcript.json and <rid>.tokens.jsonl."""
    return os.path.join(sora_root(), "annotations", "c1")


def manifest_path():
    """The team's recording manifest (upstream; c1/data/manifest.csv derives from it)."""
    return os.path.join(sora_root(), "manifests", "recordings_current.csv")


def audio_path(rid):
    return os.path.join(audio_dir(), f"{rid}.wav")


def transcript_path(rid):
    return os.path.join(gold_dir(), f"{rid}.transcript.json")


def tokens_path(rid):
    return os.path.join(gold_dir(), f"{rid}.tokens.jsonl")


def gold_recording_ids():
    """Every recording with a gold transcript, sorted. Includes ids lacking audio."""
    suffix = ".transcript.json"
    return sorted(
        fname[: -len(suffix)]
        for fname in os.listdir(gold_dir())
        if fname.endswith(suffix)
    )


def has_audio(rid):
    return os.path.exists(audio_path(rid))


def load_transcript(rid):
    with open(transcript_path(rid), encoding="utf-8") as fh:
        return json.load(fh)


def load_tokens(rid):
    """Gold tokens for one recording, as a list of dicts."""
    out = []
    with open(tokens_path(rid), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ── c1/ output locations ──────────────────────────────────────
def c1_path(*parts):
    return os.path.join(C1_ROOT, *parts)


def data_dir():
    return c1_path("data")


def predictions_dir():
    return c1_path("predictions")


def eval_dir():
    return c1_path("eval")


def reports_dir():
    return c1_path("reports")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == "__main__":
    ids = gold_recording_ids()
    missing = [r for r in ids if not has_audio(r)]
    print(f"SORA_Dataset root : {sora_root()}")
    print(f"c1 root           : {C1_ROOT}")
    print(f"gold recordings   : {len(ids)}")
    print(f"missing audio     : {missing or 'none'}")
