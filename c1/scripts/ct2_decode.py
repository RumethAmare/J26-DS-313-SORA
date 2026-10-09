#!/usr/bin/env python3
"""
ct2_decode.py -- the ONE place that defines how the exported C1 models decode.

Used by score_ct2.py (test-set accuracy) and live_transcribe.py (microphone),
so the number measured on the test set is the behaviour you get live.

Safeguards against repetition loops -- the same procedure as
safeguard_decode.py, which scored best on the test set:
  1. greedy decode (fast; fine for most speech)
  2. if the output is more repetitive than any gold transcript in the corpus
     (safeguard_decode.loop_reasons: compression ratio > 3.0, a word repeated
     > 10 times or a phrase > 5 times in a row, > 10 words/s), re-decode
     that audio with 5-beam search
  3. if it still loops, cut it where the repetition starts (truncate_loop)
faster-whisper's own fallback (re-decoding at higher temperature, i.e. random
sampling) was tried first: WER 0.546 vs 0.459 for medium_v4_aug, and ~4x
slower, so it is switched off. no_repeat_ngram_size stays 0: real speech
repeats words (හරි හරි, ඔව් ඔව්).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cuda_env  # noqa: E402

cuda_env.ensure_cuda_libs()

import safeguard_decode as sg  # noqa: E402

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CT2_DIR = os.path.join(C1_ROOT, "models", "ct2")
SR = 16000

# --model short name -> exported run
MODELS = {"medium": "medium_v4_aug", "large-v3": "large-v3_v4_aug"}

DECODE_OPTS = dict(
    language="en",                 # the token the adapters were trained with
    task="transcribe",
    beam_size=1,
    temperature=0.0,               # no sampling fallback; see the docstring
    compression_ratio_threshold=None,
    log_prob_threshold=None,
    no_speech_threshold=None,
    condition_on_previous_text=False,
    without_timestamps=True,       # the adapters were trained with <|notimestamps|>
                                   # (word_timestamps still come from cross-attention)
    no_repeat_ngram_size=0,
    word_timestamps=True,
    vad_filter=False,              # callers hand in speech already
)
BEAM_FALLBACK = 5
# Provisional (while-speaking) text: fastest possible, no checks.
PARTIAL_OPTS = dict(DECODE_OPTS, word_timestamps=False)


def load_model(name, compute_type="float16"):
    from faster_whisper import WhisperModel
    run = MODELS.get(name, name)
    path = os.path.join(CT2_DIR, run)
    if not os.path.isdir(path):
        raise SystemExit(f"{path} missing -- run: python export_ct2.py --run {run}")
    model = WhisperModel(path, device="cuda", compute_type=compute_type)
    # Measured on this CTranslate2 build (4.8.1): Whisper.generate stops after
    # max_length/2 tokens (448 -> 224, 440 -> 222, 400 -> 202), so the default
    # cut dense Sinhala clips (~385 tokens for 25 s of speech) off mid-word.
    # The decoder itself has 448 positions, 4 of them taken by the prompt, so
    # allow 443 new tokens: max_length = 2 * 443. The model still stops at its
    # end-of-text token; runaway loops end at the cap and are then handled by
    # the loop safeguards.
    model.max_length = 2 * (448 - 4 - 1)
    return run, model


def _run(model, audio, opts):
    segments, info = model.transcribe(audio, **opts)
    texts, words = [], []
    for seg in segments:
        texts.append(seg.text.strip())
        for w in seg.words or []:
            if w.word.strip():
                words.append({"word": w.word.strip(), "start": round(w.start, 2),
                              "end": round(w.end, 2), "probability": round(float(w.probability), 4)})
    return " ".join(t for t in texts if t), words, info


def transcribe(model, audio, partial=False, word_times=True):
    """audio: float32 mono 16 kHz numpy array.

    Returns (text, words, info, action): words is [{word, start, end,
    probability}]; action is "greedy", "beam re-decode" or "beam + truncate".
    word_times=False skips the word-alignment pass (0.4-0.7 s faster per
    utterance); words then carry no start/end/probability.
    """
    if partial:
        text, words, info = _run(model, audio, PARTIAL_OPTS)
        return text, words, info, "partial"
    opts = dict(DECODE_OPTS, word_timestamps=word_times)
    text, words, info = _run(model, audio, opts)
    dur = len(audio) / SR
    if not sg.loop_reasons(text, dur):
        return text, words, info, "greedy"
    text, words, info = _run(model, audio, dict(opts, beam_size=BEAM_FALLBACK))
    if not sg.loop_reasons(text, dur):
        return text, words, info, "beam re-decode"
    text = sg.truncate_loop(text, dur)
    return text, words[:len(text.split())], info, "beam + truncate"
