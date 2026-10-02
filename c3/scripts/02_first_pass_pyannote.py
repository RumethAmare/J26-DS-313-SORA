"""
STEP 2 - Machine first pass. Let pyannote guess who spoke when.

This is your annotation ASSISTANT, not your answer. You correct it by ear in
Audacity next. The same output is also your zero-shot baseline later.

PROTOCOL v1 (C3_Diarization_Roadmap, section 2) - do not improvise:
  * NO speaker-count constraint. The model must decide for itself.
  * Pipeline defaults, nothing tweaked.
  * The token is read from the environment. It is NEVER written in this file.

An "oracle" run with the true speaker count is allowed ONLY as a separate,
clearly-labelled extra condition (--oracle-speakers). Never quote an oracle
number as the headline - that is what invalidated experiment E002.

--------------------------------------------------------------------------
SETUP, once
--------------------------------------------------------------------------
    pip install "pyannote.audio>=4.0.1"

    Version 4 or newer is REQUIRED. Older releases call a torchaudio function
    that no longer exists and crash on import ("no attribute
    set_audio_backend"). Version 4 needs ffmpeg on your PATH.

    Accept the model terms while logged in to huggingface.co:
      huggingface.co/pyannote/speaker-diarization-3.1
      huggingface.co/pyannote/segmentation-3.0
      huggingface.co/pyannote/speaker-diarization-community-1

    Store your token OUTSIDE the code:
      Windows :  setx HF_TOKEN "hf_xxxxxxxx"     (reopen the terminal after)

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python 02_first_pass_pyannote.py ../processed ../rttm_draft_community-1
    python 02_first_pass_pyannote.py ../processed ../rttm_draft_v3.1 --model v3.1
    python 02_first_pass_pyannote.py ../processed ../rttm_draft --oracle-speakers 4

No NVIDIA GPU? It falls back to CPU automatically. Slower, but it works.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Your project's premise is that audio never leaves the device. pyannote 4 sends
# anonymous usage stats by default. Off, before anything is imported.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

MODELS = {
    "community-1": "pyannote/speaker-diarization-community-1",
    "v3.1": "pyannote/speaker-diarization-3.1",
}


def get_token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok:
        try:  # running in Colab?
            from google.colab import userdata  # type: ignore
            tok = userdata.get("HF_TOKEN")
        except Exception:
            tok = None
    if not tok:
        sys.exit(
            "No Hugging Face token found.\n"
            "  Windows:  setx HF_TOKEN \"hf_xxxx\"   then reopen the terminal.\n"
            "Do NOT paste the token into this file."
        )
    if not tok.startswith("hf_"):
        print("  ! that does not look like a Hugging Face token "
              "(they start with 'hf_')")
    return tok


def load_pipeline(model_id: str, token: str):
    import torch
    from pyannote.audio import Pipeline

    try:
        pipe = Pipeline.from_pretrained(model_id, token=token)
    except TypeError:          # pyannote 3.x used a different argument name
        pipe = Pipeline.from_pretrained(model_id, use_auth_token=token)

    if pipe is None:
        sys.exit(
            f"{model_id} came back empty.\n"
            "Almost always means the model terms have not been accepted on "
            "huggingface.co while logged in to the account that owns this token."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(torch.device(device))
    print(f"{model_id}\n  loaded on {device}")
    if device == "cpu":
        print("  (CPU works but is slow. A free Colab T4 is much quicker.)")
    return pipe


def as_annotation(output):
    """pyannote 4 returns a container; pyannote 3 returned the annotation."""
    return getattr(output, "speaker_diarization", output)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", choices=list(MODELS), default="community-1",
                    help="community-1 (default, current best) or v3.1 (the one "
                         "your TAF and experiment log reference)")
    ap.add_argument("--oracle-speakers", type=int, default=None,
                    help="EXTRA CONDITION ONLY. Forces the speaker count. "
                         "Written to a separate *_oracle folder.")
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir)
    if args.oracle_speakers:
        out_dir = out_dir.parent / f"{out_dir.name}_oracle{args.oracle_speakers}"
        print(f"\n*** ORACLE RUN: speaker count forced to {args.oracle_speakers}.")
        print("*** Labelled extra condition, NOT the headline number.\n")
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        sys.exit(f"No .wav files in {wav_dir}")

    pipe = load_pipeline(MODELS[args.model], get_token())

    print(f"\n{'clip':<16}{'speakers found':>16}{'turns':>8}{'runtime':>10}")
    print("-" * 50)

    counts = []
    for wav in wavs:
        t0 = time.time()
        kwargs = {"num_speakers": args.oracle_speakers} if args.oracle_speakers else {}
        ann = as_annotation(pipe(str(wav), **kwargs))

        with open(out_dir / f"{wav.stem}.rttm", "w", encoding="utf-8") as fh:
            ann.write_rttm(fh)

        n = len(ann.labels())
        counts.append(n)
        print(f"{wav.stem:<16}{n:>16}{len(list(ann.itertracks())):>8}"
              f"{time.time() - t0:>9.1f}s")

    print(f"\nDraft RTTM written to: {out_dir}")
    if not args.oracle_speakers:
        print(f"Speaker counts found: {counts}")
        print("Anything other than the true number is not a bug - it is your "
              "research problem, measured. Write it down.")
    print("\nNext: python 03_rttm_to_audacity.py "
          f"{out_dir.as_posix()} ../draft_labels")


if __name__ == "__main__":
    main()
