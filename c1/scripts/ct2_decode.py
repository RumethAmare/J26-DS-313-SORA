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
The same retry also catches invented numbers (see too_many_digits).
faster-whisper's own fallback (re-decoding at higher temperature, i.e. random
sampling) was tried first: WER 0.546 vs 0.459 for medium_v4_aug, and ~4x
slower, so it is switched off. no_repeat_ngram_size stays 0: real speech
repeats words (හරි හරි, ඔව් ඔව්).
"""
import os
import re
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
MODELS = {"small": "small_v4_aug", "medium": "medium_v4_aug", "large-v3": "large-v3_v4_aug"}

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


# --- invented numbers ----------------------------------------------------------
# 26% of the long numbers in the training transcripts contain a counting run
# (0771234567, 198512345678: the role-play scripts used placeholder numbers),
# against 0.5% for random digits. Given little audio -- e.g. one digit read on
# its own -- the model falls back on that habit and writes "1 2 3 4 5 ...".
# Read aloud, even a fast phone number is ~5 digits per second; the invented
# runs seen live were 8-15 per second.
MAX_DIGITS_PER_S = 6.0
DIGIT_SLACK = 2
_DIGIT = re.compile(r"\d")


def too_many_digits(text, dur):
    return len(_DIGIT.findall(text)) > MAX_DIGITS_PER_S * dur + DIGIT_SLACK


def trim_digits(text, dur):
    """Drop digits beyond what the audio could hold, from the END (where the
    invented counting run is), keeping everything else."""
    allowed = int(MAX_DIGITS_PER_S * dur + DIGIT_SLACK)
    out, seen = [], 0
    for ch in text:
        if ch.isdigit():
            seen += 1
            if seen > allowed:
                continue
        out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


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


# --- word timestamps -------------------------------------------------------------
# "ctc"     MMS forced alignment of the final text (word_align.py): times from
#           the sound itself, ~50 ms per utterance. The default for token times.
# "whisper" faster-whisper's cross-attention estimate (word_timestamps=True);
#           also the only source of per-word asr_confidence; +0.4-0.7 s.
# "both"    whisper pass for asr_confidence, CTC for the times.
# "none"    text only.
# eval_timestamps.py compares the methods against the gold token times.
TIMINGS = ("ctc", "whisper", "both", "none")
_aligner = None


def get_aligner():
    global _aligner
    if _aligner is None:
        import word_align
        _aligner = word_align.Aligner()
    return _aligner


def plain_words(text):
    return [{"word": w, "start": None, "end": None, "probability": None} for w in text.split()]


def match_words(text, words):
    """Whisper's word records for the FINAL text. The safeguards can cut or
    edit the text after decoding, and Whisper's word list does not always
    split where the spaces are; a word keeps Whisper's times only where the
    two sequences agree exactly, so a time never sits on the wrong word."""
    from difflib import SequenceMatcher
    out = plain_words(text)
    a, b = [o["word"] for o in out], [w["word"] for w in words]
    for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            out[blk.a + k].update(words[blk.b + k])
    return out


def align_words(audio, words, min_score=None, use_whisper=False):
    """Overwrite start/end with CTC forced-alignment times (in place) and add
    align_score. Words keep start/end None if the alignment fails.
    min_score: re-time words aligned with a lower score (word_align.repair),
    from Whisper's own times if use_whisper and the words carry them, else by
    interpolation between confident neighbours."""
    import word_align
    names = [w["word"] for w in words]
    times = get_aligner().align(audio, names)
    if times is None:
        for w in words:
            w["start"] = w["end"] = w["align_score"] = None
        return words
    if min_score is not None:
        fb = [dict(w) for w in words] if use_whisper else None
        word_align.repair(times, names, min_score, fb)
    for w, t in zip(words, word_align.tidy(times, len(audio) / SR)):
        w["start"], w["end"], w["align_score"] = round(t["start"], 2), round(t["end"], 2), t["score"]
        if t.get("retimed"):
            w["retimed"] = True
    return words


def transcribe(model, audio, partial=False, timing="whisper"):
    """audio: float32 mono 16 kHz numpy array.

    Returns (text, words, info, action): words is [{word, start, end,
    probability[, align_score]}] and always matches text word for word;
    action is "greedy", "beam re-decode", "beam + truncate" or
    "beam + digit trim". An output counts as bad if it loops (loop_reasons)
    or holds more digits than the audio could (too_many_digits).
    timing: see TIMINGS. Callers that want the text on screen before the
    alignment runs use timing="none" and then align_words().
    """
    assert timing in TIMINGS, timing
    if partial:
        text, words, info = _run(model, audio, PARTIAL_OPTS)
        return text, words, info, "partial"
    opts = dict(DECODE_OPTS, word_timestamps=timing in ("whisper", "both"))
    text, words, info = _run(model, audio, opts)
    dur = len(audio) / SR

    def bad(t):
        return sg.loop_reasons(t, dur) or too_many_digits(t, dur)

    action = "greedy"
    if bad(text):
        text, words, info = _run(model, audio, dict(opts, beam_size=BEAM_FALLBACK))
        action = "beam re-decode"
        if bad(text):
            action = "beam + truncate"
            if sg.loop_reasons(text, dur):
                text = sg.truncate_loop(text, dur)
            if too_many_digits(text, dur):
                text, action = trim_digits(text, dur), "beam + digit trim"
    words = match_words(text, words)
    if timing in ("ctc", "both"):
        align_words(audio, words)
    return text, words, info, action
