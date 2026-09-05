#!/usr/bin/env python3
"""
Stage A -- slice ASR fine-tuning clips from train-split recordings.

Three things this script refuses to do, on purpose:

1. Touch eval-split audio. Imports split_guard.train_ids() and nothing else --
   there is no code path here that can see an eval recording id, let alone
   slice a clip from one.
2. Count R0005 / R0008 toward the training pool. C1_FINAL_IMPLEMENTATION.md
   §8 says re-annotate them before they count -- data/reference_audit.json
   (from Phase 0's audit_references.py) already flags both for incomplete
   gold-transcript timeline coverage, so this reads that file and excludes
   them by default rather than waiting on a human to remember the rule.
3. Trust the annotated utterance boundary exactly. R0002 and R0003's token
   timestamps are still marked PROVISIONAL in the upstream manifest (not yet
   through WhisperX forced alignment), so every clip boundary is trimmed with
   webrtcvad rather than cut at the raw annotated edge -- cheap insurance
   against slicing off the first or last word of a provisional utterance.
   (webrtcvad, not silero-vad: silero-vad's PyPI package pulls torch and
   torchaudio unconditionally, which defeats the point of a lightweight
   Phase-1 VAD -- see load_vad_model()'s docstring below.)

Usage:
    python -m src.preprocess                    # all eligible train recordings
    python -m src.preprocess --include-flagged  # also slice R0005 / R0008
    python -m src.preprocess --denoise          # run noisereduce on each clip
"""
import argparse
import csv
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

from src import sora_paths
from src.split_guard import assert_no_eval_leakage, train_ids

MIN_CLIP_S, MAX_CLIP_S = 1.0, 30.0
TARGET_SR = 16000
OUT_DIR = os.path.join(sora_paths.data_dir(), "asr_clips")
MANIFEST_PATH = os.path.join(sora_paths.data_dir(), "asr_clips_manifest.csv")
FIELDNAMES = ["clip_path", "recording", "utt_id", "start", "end", "duration_s", "text"]


def flagged_recordings():
    """R0005/R0008-style recordings: incomplete gold reference, per Phase 0's
    audit_references.py. Read fresh each run so a re-annotation is picked up
    automatically once the audit is re-run, with no code change needed here."""
    path = os.path.join(sora_paths.data_dir(), "reference_audit.json")
    if not os.path.exists(path):
        print(f"  (no {path} -- run src.audit_references first to enable the "
              f"incomplete-reference exclusion; proceeding without it)")
        return set()
    with open(path, encoding="utf-8") as fh:
        audit = json.load(fh)
    return set(audit.get("recommend_exclude_from_asr_scoring", []))


VAD_FRAME_MS = 30  # webrtcvad accepts 10/20/30ms frames only


def load_vad_model():
    """
    webrtcvad, not silero-vad: silero-vad's PyPI package requires torch and
    torchaudio unconditionally (confirmed against its PyPI metadata), which
    contradicts the whole reason Stage A lists a VAD dependency separately
    from Stage B's torch install -- torch is deliberately deferred to Phase 2
    (C1_FINAL_IMPLEMENTATION.md's own hardware note; ~5GB against limited
    disk). webrtcvad is a plain C-extension frame classifier with no ML
    framework behind it, which is what "lightweight VAD for Phase 1" actually
    meant. Aggressiveness 2 (of 0-3): more willing to call something speech
    than the most aggressive setting, which matters more here than shaving a
    few extra ms of silence off each clip edge.
    """
    import webrtcvad
    return webrtcvad.Vad(2)


def vad_trim(audio, sr, vad):
    """
    Trim leading/trailing silence by classifying fixed-length frames as
    speech/non-speech and cutting at the first and last speech frame.

    Falls back to the untrimmed clip if VAD finds no speech at all -- an empty
    trim is a worse failure than an untrimmed clip, since the former loses an
    entire training example rather than a fraction of a second of silence.
    """
    frame_len = int(sr * VAD_FRAME_MS / 1000)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    speech_frames = []
    for i in range(0, len(pcm16) - frame_len + 1, frame_len):
        frame = pcm16[i:i + frame_len].tobytes()
        if vad.is_speech(frame, sr):
            speech_frames.append(i)

    if not speech_frames:
        return audio
    start, end = speech_frames[0], speech_frames[-1] + frame_len
    return audio[start:end]


def slice_recording(rid, vad, denoise, writer):
    audio, sr = sf.read(sora_paths.audio_path(rid))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        raise ValueError(
            f"{rid}: expected {TARGET_SR} Hz per DATA_CONTRACT.md, got {sr} Hz"
        )

    transcript = sora_paths.load_transcript(rid)
    n_clips = 0
    total_s = 0.0

    for utt in transcript.get("utterances", []):
        dur = utt["end"] - utt["start"]
        if dur < MIN_CLIP_S:
            continue  # too short to be a useful training example
        text = utt.get("text", "").strip()
        if not text:
            continue

        s0, s1 = int(utt["start"] * TARGET_SR), int(utt["end"] * TARGET_SR)
        clip = audio[max(0, s0):min(len(audio), s1)]
        if clip.size == 0:
            continue

        if vad is not None:
            clip = vad_trim(clip, TARGET_SR, vad)
        if denoise:
            import noisereduce as nr
            clip = nr.reduce_noise(y=clip, sr=TARGET_SR)

        clip_dur = len(clip) / TARGET_SR
        if clip_dur < MIN_CLIP_S:
            continue
        if clip_dur > MAX_CLIP_S:
            # §6.1 wants 1-30s clips; an over-long utterance is split at its
            # midpoint rather than truncated, so no text is silently dropped.
            mid = len(clip) // 2
            _write_clip(rid, utt, clip[:mid], text, writer, suffix="a")
            _write_clip(rid, utt, clip[mid:], text, writer, suffix="b")
            n_clips += 2
            total_s += clip_dur
            continue

        _write_clip(rid, utt, clip, text, writer)
        n_clips += 1
        total_s += clip_dur

    return n_clips, total_s


def _write_clip(rid, utt, clip, text, writer, suffix=""):
    clip_id = f"{utt['utt_id']}{suffix}"
    clip_path = os.path.join(OUT_DIR, f"{clip_id}.wav")
    sf.write(clip_path, clip.astype(np.float32), TARGET_SR, subtype="PCM_16")
    writer.writerow({
        "clip_path": clip_path,
        "recording": rid,
        "utt_id": clip_id,
        "start": round(utt["start"], 3),
        "end": round(utt["end"], 3),
        "duration_s": round(len(clip) / TARGET_SR, 3),
        "text": text,
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-flagged", action="store_true",
                    help="also slice R0005/R0008-style incomplete-reference recordings")
    ap.add_argument("--denoise", action="store_true", help="run noisereduce per clip")
    ap.add_argument("--no-vad", action="store_true",
                    help="skip silero-vad trimming (debug only)")
    args = ap.parse_args()

    recordings = assert_no_eval_leakage(train_ids(), context="preprocess (Stage A)")

    flagged = flagged_recordings()
    if flagged and not args.include_flagged:
        skipped = sorted(set(recordings) & flagged)
        recordings = sorted(set(recordings) - flagged)
        if skipped:
            print(f"Excluding {len(skipped)} flagged recording(s) (incomplete "
                  f"reference, per data/reference_audit.json): {skipped}")
            print("  Pass --include-flagged to override once re-annotated.\n")

    vad = None
    if not args.no_vad:
        print("Loading webrtcvad...")
        try:
            vad = load_vad_model()
        except Exception as exc:
            print(f"  webrtcvad unavailable ({type(exc).__name__}: {exc}); "
                  f"proceeding without edge-trimming.\n")

    sora_paths.ensure_dir(OUT_DIR)
    total_clips = 0
    total_seconds = 0.0

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for i, rid in enumerate(recordings, 1):
            print(f"[{i}/{len(recordings)}] {rid}", flush=True)
            n_clips, secs = slice_recording(rid, vad, args.denoise, writer)
            total_clips += n_clips
            total_seconds += secs
            print(f"    -> {n_clips} clips, {secs:.1f}s")

    hours = total_seconds / 3600
    print(f"\n{total_clips} clips written to {OUT_DIR}")
    print(f"Total: {total_seconds/60:.1f} min ({hours:.2f} h)")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"\nAgainst C1_FINAL_IMPLEMENTATION.md §6.2's 15-20h floor: "
          f"{hours:.2f}h ({hours/15:.0%} of the 15h floor).")
    print("Run `python -m src.collection_pace` for the pace-vs-deadline view.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
