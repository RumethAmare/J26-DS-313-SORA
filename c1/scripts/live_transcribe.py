#!/usr/bin/env python3
"""
live_transcribe.py -- near real-time Sinhala-English transcription from the
microphone with the fine-tuned C1 Whisper models.

How it works:
  microphone (16 kHz) -> Silero VAD, 32 ms at a time
    -> speech starts: collect audio, show provisional text every ~1 s
    -> pause of SILENCE_MS (or 25 s of speech): the utterance is final;
       transcribe it with the safeguarded decoder (ct2_decode.py), label each
       word SI/EN/OTHER with the Task 3 hybrid LID, print it coloured
    -> after the text is on screen: CTC forced alignment (word_align.py)
       gives each word its start/end in the session (--timing)
  Ctrl+C -> saves predictions/live/<timestamp>.json (and the audio with
  --save-audio; the token stream for C3 with --tokens-jsonl).

Whisper is not a streaming model, so text arrives per utterance: shortly
after each pause, not word by word. Provisional text fills the gap.

Usage:
    python live_transcribe.py --model medium            # faster
    python live_transcribe.py --model large-v3          # more accurate
    python live_transcribe.py --model small             # lightest; works without a big GPU
    python live_transcribe.py --list-devices
    python live_transcribe.py --model medium --device 3 --save-audio
    python live_transcribe.py --input-file x.wav --realtime   # test without a mic
    python live_transcribe.py --model large-v3 --input-file rec.wav --tokens-jsonl rec.tokens.jsonl
                                                              # whole recording -> C3 token stream
"""
import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ct2_decode  # noqa: E402  (CUDA libs first)
import numpy as np  # noqa: E402

SR = ct2_decode.SR
BLOCK = 512                    # 32 ms; the Silero VAD frame size at 16 kHz
SPEECH_ON = 0.5                # VAD probability to call a frame speech
SPEECH_OFF = 0.35              # ... and to call it silence (hysteresis)
PREROLL_S = 0.3                # audio kept from just before speech started
MIN_UTT_S = 0.4                # shorter blips are ignored
MAX_UTT_S = 25.0               # default cap (--max-utterance-s); Whisper's window is 30 s
PARTIAL_EVERY_S = 1.0
# A number read digit by digit has pauses between the digits. Cutting there
# leaves 1 s fragments with no context, where the model invents counting runs
# ("1 2 3 4 5"). So when a pause comes right after a digit, keep listening.
NUMBER_PAUSE_MS = 1500
_ENDS_IN_NUMBER = re.compile(
    # a digit, or a number word, at the end of the text. Not "eka"/"එක":
    # besides "one" it is the everyday Sinhala "the" ("packet eka").
    r"(\d|\b(zero|oh|two|three|four|five|six|seven|eight|nine|double|triple)"
    r"|බිංදුව|බින්දුව|දෙක|තුන|හතර|පහ|හය|හත|අට|නවය"
    r"|\b(binduwa|bindu|deka|thuna|hathara|paha|haya|hatha|ata|nawaya))\W*$",
    re.IGNORECASE)

C1_ROOT = ct2_decode.C1_ROOT
OUT_DIR = os.path.join(C1_ROOT, "predictions", "live")

COLOR = {"SI": "\x1b[32m", "EN": "\x1b[36m", "OTHER": "\x1b[33m"}
DIM, BOLD, RESET, CLEAR = "\x1b[2m", "\x1b[1m", "\x1b[0m", "\r\x1b[2K"


class StreamingVAD:
    """Silero VAD with its recurrent state carried from frame to frame.

    faster-whisper's SileroVADModel resets the state on every call, which is
    right for whole files but wrong for a live stream; this feeds the same
    ONNX model one 32 ms frame at a time.
    """

    def __init__(self):
        from faster_whisper.vad import get_vad_model
        self.session = get_vad_model().session
        self.h = np.zeros((1, 1, 128), dtype="float32")
        self.c = np.zeros((1, 1, 128), dtype="float32")
        self.context = np.zeros(64, dtype="float32")

    def __call__(self, frame):
        x = np.concatenate([self.context, frame])[None, :].astype("float32")
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = frame[-64:]
        return float(np.asarray(out).reshape(-1)[0])


def label_tokens(words, lid, utt_id, t0):
    """Word dicts -> C1 token-stream records (same fields as build_token_stream)."""
    toks, prev = [], None
    for i, w in enumerate(words):
        lang, conf, method = lid.predict(w["word"])
        toks.append({"utt_id": utt_id, "tok_id": i, "token": w["word"], "lang": lang,
                     "start": None if w["start"] is None else round(t0 + w["start"], 2),
                     "end": None if w["end"] is None else round(t0 + w["end"], 2),
                     "switch": prev is not None and lang != prev,
                     "asr_confidence": w["probability"], "lang_confidence": conf,
                     "lang_method": method, "align_score": w.get("align_score")})
        prev = lang
    return toks


def coloured(toks, text):
    if not toks:
        return text
    return " ".join(f"{COLOR.get(t['lang'], '')}{t['token']}{RESET}" for t in toks)


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def mic_source(device, q, stop):
    import sounddevice as sd

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK,
                        device=device, callback=cb):
        while not stop.is_set():
            time.sleep(0.05)


def file_source(path, q, stop, realtime):
    import soundfile as sf
    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        wave = resample_poly(wave, SR, sr).astype("float32")
    for i in range(0, len(wave) - BLOCK + 1, BLOCK):
        if stop.is_set():
            return
        q.put(wave[i:i + BLOCK])
        if realtime:
            time.sleep(BLOCK / SR)
    q.put(None)                                   # end of file


TIME_METHODS = {"ctc": "MMS forced alignment (word_align.py)",
                "both": "MMS forced alignment (word_align.py); asr_confidence from whisper",
                "whisper": "faster-whisper cross-attention (word_timestamps)",
                "none": None}


class LiveSession:
    """The live pipeline, one 32 ms frame at a time -- shared by the terminal
    (main below) and the web page (live_web.py), so both behave the same.

    feed(frame) / finish() return events (dicts with "type"):
      speech      {"on": bool, "t": session time}    speech started / stopped
      partial     {"text"}                           provisional text
      discard     {}                                 blip too short to transcribe
      utterance   {"utt": utterance dict, "dur"}     final text + tokens (no CTC times yet)
      times       {"utt_id", "tokens", "align_s"}    the same tokens with CTC word times
    The word alignment is returned as a callable after the utterance event:
    the caller shows/sends the text first, then calls it to get the times
    event, so times never delay text.
    """

    def __init__(self, model, run, lid, silence_ms=600, number_pause_ms=NUMBER_PAUSE_MS,
                 max_utterance_s=MAX_UTT_S, partials=True, timing="ctc", utt_prefix="live",
                 keep_audio=False):
        self.model, self.run, self.lid = model, run, lid
        self.silence_ms, self.number_pause_ms = silence_ms, number_pause_ms
        self.max_utterance_s, self.partials, self.timing = max_utterance_s, partials, timing
        self.utt_prefix, self.keep_audio = utt_prefix, keep_audio
        self.vad = StreamingVAD()
        self.ctc = timing in ("ctc", "both")
        self.silence_frames = int(silence_ms / 1000 * SR / BLOCK)
        self.number_frames = int(number_pause_ms / 1000 * SR / BLOCK)
        self.waiting_number = False                # pause came after a digit
        self.preroll = []                          # recent frames before speech
        self.utt, self.in_speech, self.quiet, self.last_partial = [], False, 0, 0.0
        self.t_audio = 0.0                         # session clock, from samples received
        self.utt_start = 0.0
        self.audio, self.utterances = [], []
        self.started = datetime.now().strftime("%Y%m%d-%H%M%S")

    def _finalize(self, reason):
        audio = np.concatenate(self.utt) if self.utt else np.zeros(0, dtype="float32")
        self.utt, self.in_speech, self.quiet = [], False, 0
        dur = len(audio) / SR
        off = [] if reason == "max" else [{"type": "speech", "on": False, "t": round(self.t_audio, 2)}]
        if dur < MIN_UTT_S:
            return [{"type": "discard"}] + off
        utt_start = self.utt_start
        t0 = time.time()
        # CTC times are added after the text is out, so they never delay it
        timing = {"ctc": "none", "both": "whisper"}.get(self.timing, self.timing)
        text, words, _, action = ct2_decode.transcribe(self.model, audio, timing=timing)
        dec = time.time() - t0
        uid = f"{self.utt_prefix}_u{len(self.utterances) + 1:03d}"
        toks = label_tokens(words, self.lid, uid, utt_start)
        lag = dec + (self.silence_ms / 1000 if reason == "pause" else 0)
        u = {"utt_id": uid, "start": round(utt_start, 2), "end": round(utt_start + dur, 2),
             "text": text, "tokens": toks, "end_reason": reason,
             "decode_s": round(dec, 2), "latency_after_speech_s": round(lag, 2),
             "decode_action": action}
        self.utterances.append(u)
        events = [{"type": "utterance", "utt": u, "dur": round(dur, 2)}] + off
        if self.ctc and words:
            events.append(lambda: self._align(u, audio, words))   # run after the text is sent
        return events

    def _align(self, u, audio, words):
        t1 = time.time()
        ct2_decode.align_words(audio, words)
        for t, w in zip(u["tokens"], words):
            t["start"] = None if w["start"] is None else round(u["start"] + w["start"], 2)
            t["end"] = None if w["end"] is None else round(u["start"] + w["end"], 2)
            t["align_score"] = w.get("align_score")
        u["align_s"] = round(time.time() - t1, 3)
        return {"type": "times", "utt_id": u["utt_id"], "tokens": u["tokens"], "align_s": u["align_s"]}

    def feed(self, frame, backlog=0):
        """One BLOCK of float32 16 kHz audio. backlog: frames still waiting
        behind this one (provisional text is skipped when falling behind)."""
        if self.keep_audio:
            self.audio.append(frame)
        p = self.vad(frame)
        self.t_audio += BLOCK / SR
        if not self.in_speech:
            self.preroll.append(frame)
            self.preroll = self.preroll[-int(PREROLL_S * SR / BLOCK):]
            if p >= SPEECH_ON:
                self.in_speech, self.quiet = True, 0
                self.utt = list(self.preroll)
                self.utt_start = max(0.0, self.t_audio - len(self.utt) * BLOCK / SR)
                self.last_partial = time.time()
                return [{"type": "speech", "on": True, "t": round(self.utt_start, 2)}]
            return []
        self.utt.append(frame)
        self.quiet = self.quiet + 1 if p < SPEECH_OFF else 0
        if self.quiet == 0:
            self.waiting_number = False
        if (self.quiet == self.silence_frames and not self.waiting_number
                and self.number_frames > self.silence_frames):
            # Normal pause reached: if the speech so far ends on a number,
            # keep listening -- the next digit is probably coming.
            tail = np.concatenate(self.utt[-int(4 * SR / BLOCK):])
            text, _, _, _ = ct2_decode.transcribe(self.model, tail, partial=True)
            self.waiting_number = bool(_ENDS_IN_NUMBER.search(text))
        need = self.number_frames if self.waiting_number else self.silence_frames
        if self.quiet >= need:
            self.utt = self.utt[:-max(0, self.quiet - int(0.2 * SR / BLOCK))]   # keep 0.2 s of the pause
            self.waiting_number = False
            return self._finalize("pause")
        if len(self.utt) * BLOCK / SR >= self.max_utterance_s:
            events = self._finalize("max")
            self.in_speech, self.utt = True, []                                 # speech continues
            self.utt_start, self.last_partial = self.t_audio, time.time()
            return events
        if (self.partials and time.time() - self.last_partial >= PARTIAL_EVERY_S
                and backlog < int(0.5 * SR / BLOCK)):                           # only if keeping up
            text, _, _, _ = ct2_decode.transcribe(self.model, np.concatenate(self.utt), partial=True)
            self.last_partial = time.time()
            return [{"type": "partial", "text": text}]
        return []

    def finish(self, reason="stopped"):
        return self._finalize(reason) if self.in_speech else []

    def document(self, source):
        lags = [u["latency_after_speech_s"] for u in self.utterances if u["end_reason"] == "pause"]
        return {
            "n_utterances": len(self.utterances),
            "transcript": " ".join(u["text"] for u in self.utterances if u["text"]),
            "utterances": self.utterances,
            "session": {
                "model": self.run,
                "model_dir": os.path.relpath(os.path.join(ct2_decode.CT2_DIR, self.run), C1_ROOT),
                "source": source, "started": self.started, "audio_s": round(self.t_audio, 1),
                "decode_opts": {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in ct2_decode.DECODE_OPTS.items()},
                "endpointing": {"silence_ms": self.silence_ms, "number_pause_ms": self.number_pause_ms,
                                "max_utterance_s": self.max_utterance_s,
                                "vad": "silero (faster-whisper bundle), streaming"},
                "timing": self.timing,
                "time_method": TIME_METHODS[self.timing],
                "latency_after_speech_s": {
                    "median": round(float(np.median(lags)), 2) if lags else None,
                    "max": round(max(lags), 2) if lags else None},
            },
        }

    def save(self, doc, out, tokens_jsonl=None):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        if tokens_jsonl:
            with open(tokens_jsonl, "w", encoding="utf-8") as fh:
                for u in self.utterances:
                    for t in u["tokens"]:
                        fh.write(json.dumps(t, ensure_ascii=False) + "\n")
        if self.keep_audio and self.audio:
            import soundfile as sf
            sf.write(out[:-5] + ".wav", np.concatenate(self.audio), SR)


def warm_up(model, timing):
    model.transcribe(np.zeros(SR, dtype="float32"), **ct2_decode.PARTIAL_OPTS)
    if timing in ("ctc", "both"):
        ct2_decode.align_words(np.zeros(SR, dtype="float32"), [{"word": "a"}])  # load + warm-up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium", help="small | medium | large-v3")
    ap.add_argument("--device", type=int, default=None, help="microphone index (--list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--silence-ms", type=int, default=600,
                    help="pause that ends an utterance (shorter = faster text, more cuts)")
    ap.add_argument("--number-pause-ms", type=int, default=NUMBER_PAUSE_MS,
                    help="pause needed to end an utterance that ends on a digit "
                         "(0 = treat numbers like any other speech)")
    ap.add_argument("--max-utterance-s", type=float, default=MAX_UTT_S,
                    help="force a cut after this much continuous speech (lower = text "
                         "sooner during long turns; must stay under 30)")
    ap.add_argument("--no-partials", action="store_true",
                    help="skip provisional text (saves GPU; useful for large-v3)")
    ap.add_argument("--timing", choices=ct2_decode.TIMINGS, default="ctc",
                    help="word start/end in the saved tokens: ctc = forced alignment, run "
                         "after the text is shown (default); whisper = cross-attention "
                         "estimate, also gives asr_confidence (+0.4-0.7 s before the text); "
                         "both = whisper confidence + ctc times; none")
    ap.add_argument("--tokens-jsonl", help="also write the token stream (one token per line, "
                    "session times) -- the format C3/C4 read")
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("--input-file", help="stream a WAV instead of the microphone")
    ap.add_argument("--realtime", action="store_true", help="with --input-file: play at real speed")
    ap.add_argument("--out", help="output JSON path (default predictions/live/<time>.json)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.system("")                                  # ANSI colours on Windows

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    print(f"loading {a.model} ...", flush=True)
    run, model = ct2_decode.load_model(a.model)
    from lid_hybrid import HybridLID
    # utt_id like the gold for a recording (J26DS313_R0018_u001), live_u001 for the mic
    prefix = os.path.splitext(os.path.basename(a.input_file))[0] if a.input_file else "live"
    session = LiveSession(model, run, HybridLID(), silence_ms=a.silence_ms,
                          number_pause_ms=a.number_pause_ms, max_utterance_s=a.max_utterance_s,
                          partials=not a.no_partials, timing=a.timing, utt_prefix=prefix,
                          keep_audio=a.save_audio)
    warm_up(model, a.timing)

    q, stop = queue.Queue(), threading.Event()
    src = threading.Thread(target=file_source if a.input_file else mic_source,
                           args=(a.input_file, q, stop, a.realtime) if a.input_file
                           else (a.device, q, stop), daemon=True)
    src.start()
    print(f"{BOLD}{run}{RESET} ready — "
          f"{'streaming ' + a.input_file if a.input_file else 'speak now'}; Ctrl+C to stop.  "
          f"{COLOR['SI']}SI{RESET} {COLOR['EN']}EN{RESET} {COLOR['OTHER']}OTHER{RESET}\n",
          flush=True)

    def show(events):
        for ev in events:
            if callable(ev):                      # deferred word alignment
                ev()
            elif ev["type"] == "discard":
                sys.stdout.write(CLEAR)
            elif ev["type"] == "partial":
                text = ev["text"]
                shown = text if len(text) <= 110 else "…" + text[-110:]
                sys.stdout.write(f"{CLEAR}{DIM}… {shown}{RESET}")
            elif ev["type"] == "utterance":
                u, toks = ev["utt"], ev["utt"]["tokens"]
                lag, action, reason = u["latency_after_speech_s"], u["decode_action"], u["end_reason"]
                sys.stdout.write(f"{CLEAR}{DIM}[{mmss(u['start'])}]{RESET} "
                                 f"{coloured(toks, u['text']) or DIM + '(no text)' + RESET}"
                                 f"  {DIM}({ev['dur']:.1f}s speech, text {lag:.1f}s after it ended"
                                 f"{', cut at max length' if reason == 'max' else ''}"
                                 f"{', fixed: ' + action if action != 'greedy' else ''}){RESET}\n")
            sys.stdout.flush()

    try:
        while True:
            frame = q.get()
            if frame is None:
                show(session.finish("end"))
                break
            show(session.feed(frame, backlog=q.qsize()))
    except KeyboardInterrupt:
        show(session.finish("stopped"))
    finally:
        stop.set()

    if not session.utterances:
        print("\nno speech captured; nothing saved")
        return
    out = a.out or os.path.join(OUT_DIR, f"{run}_{session.started}.json")
    source = a.input_file or f"microphone {a.device if a.device is not None else '(default)'}"
    session.save(session.document(source), out, a.tokens_jsonl)
    print(f"\nsaved {len(session.utterances)} utterances -> {out}")
    if a.tokens_jsonl:
        print(f"tokens -> {a.tokens_jsonl}")
    if a.save_audio and session.audio:
        print(f"audio -> {out[:-5]}.wav")


if __name__ == "__main__":
    main()
