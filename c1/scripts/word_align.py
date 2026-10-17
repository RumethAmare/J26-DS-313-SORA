#!/usr/bin/env python3
"""
word_align.py -- when was each word said? (word timestamps for C3 and C4)

Whisper decides WHAT was said; this decides WHEN. A small CTC acoustic model
(MMS forced aligner, torchaudio.pipelines.MMS_FA, ~300M parameters, trained on
1,100+ languages) scores every 20 ms frame of the audio for each letter, and
Viterbi forced alignment finds the best path through the known text. Each
word's start/end is where its first/last letter sits on that path -- the
method WhisperX uses.

Whisper's own word timestamps (cross-attention, `word_timestamps=True`) are
an estimate from attention patterns, not from the sound; eval_timestamps.py
measures both against the hand-checked gold token times.

Text the aligner can't spell is handled as follows:
  * Sinhala / Tamil script -> Latin with uroman (ආයුබෝවන් -> aayuboovan)
  * digits -> the wildcard `*`: "077" may have been read in Sinhala or English,
    so no spelling is safe; `*` matches whatever sound is there
  * a word with no letters left (e.g. "-") -> `*` as well
"""
import re
import unicodedata

import numpy as np

SR = 16000
_LATIN_OK = re.compile(r"[^a-z']")
_NON_LATIN = re.compile(r"[^\x00-\x7f]")


class Aligner:
    def __init__(self, device="cuda"):
        import torch
        from torchaudio.pipelines import MMS_FA
        import uroman
        self.torch = torch
        self.device = device
        self.model = MMS_FA.get_model(with_star=True).to(device).eval()
        self.tokenizer = MMS_FA.get_tokenizer()
        self.aligner = MMS_FA.get_aligner()
        self.uroman = uroman.Uroman()
        self.vocab = set(MMS_FA.get_dict()) - {"-"}

    def spell(self, word):
        """Word -> the letters the aligner knows ('*' if none)."""
        if any(ch.isdigit() for ch in word):
            return "*"
        if _NON_LATIN.search(word):
            word = self.uroman.romanize_string(word)
        word = unicodedata.normalize("NFKD", word)
        word = "".join(c for c in word if not unicodedata.combining(c)).lower()
        word = "".join(c for c in _LATIN_OK.sub("", word) if c in self.vocab)
        return word or "*"

    def align(self, audio, words):
        """audio: float32 mono 16 kHz; words: list of str.

        Returns [{start, end, score}] per word, seconds from the start of
        `audio`; score is the mean CTC probability of the word's letters
        (0-1: how well the sound matches the text, NOT whether the text is
        right). Returns None if the audio is too short for the text.
        """
        if not words or len(audio) < SR // 50:
            return None
        torch = self.torch
        spelled = [self.spell(w) for w in words]
        with torch.inference_mode():
            wave = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None].to(self.device)
            emission, _ = self.model(wave)
            try:
                spans = self.aligner(emission[0], self.tokenizer(spelled))
            except RuntimeError:          # more letters than frames
                return None
        sec = len(audio) / SR / emission.shape[1]
        out = []
        for s in spans:
            out.append({"start": round(s[0].start * sec, 3), "end": round(s[-1].end * sec, 3),
                        "score": round(float(np.mean([t.score for t in s])), 4)})
        return out


def repair(times, words, min_score, fallback=None):
    """Re-time words the aligner was unsure of (score < min_score).

    Low scores mean the letters did not match the sound -- the word was
    spoken unclearly, romanized differently, or is not in the audio at all
    -- and such words were placed > 0.5 s from the gold time ~40% of the time
    (vs 14% above 0.2). Each run of unsure words is spread over the gap
    between its confident neighbours, in proportion to word length; if
    `fallback` (e.g. Whisper's own word times, same length as words) has a
    time inside that gap, it is used instead. Marks re-timed words with
    "retimed": True.
    """
    n = len(times)
    ok = [t["score"] >= min_score for t in times]
    i = 0
    while i < n:
        if ok[i]:
            i += 1
            continue
        j = i
        while j < n and not ok[j]:
            j += 1
        lo = times[i - 1]["end"] if i > 0 else times[i]["start"]
        hi = times[j]["start"] if j < n else times[j - 1]["end"]
        lo, hi = min(lo, hi), max(lo, hi)
        lens = [max(len(words[k]), 1) for k in range(i, j)]
        cur = lo
        for k, L in zip(range(i, j), lens):
            span = (hi - lo) * L / sum(lens)
            fb = fallback[k] if fallback else None
            if fb and fb.get("start") is not None and lo <= fb["start"] and fb["end"] <= hi:
                times[k].update(start=fb["start"], end=fb["end"])
            else:
                times[k].update(start=round(cur, 3), end=round(cur + span, 3))
            times[k]["retimed"] = True
            cur += span
        i = j
    return times


def tidy(times, dur):
    """Clamp to [0, dur], keep starts monotonic and words from overlapping."""
    prev_end = 0.0
    for t in times:
        t["start"] = min(max(t["start"], prev_end), dur)
        t["end"] = min(max(t["end"], t["start"]), dur)
        prev_end = t["end"]
    return times
