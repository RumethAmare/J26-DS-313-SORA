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
  Ctrl+C -> saves predictions/live/<timestamp>.json (and the audio with
  --save-audio).

Whisper is not a streaming model, so text arrives per utterance: shortly
after each pause, not word by word. Provisional text fills the gap.

Usage:
    python live_transcribe.py --model medium            # faster
    python live_transcribe.py --model large-v3          # more accurate
    python live_transcribe.py --list-devices
    python live_transcribe.py --model medium --device 3 --save-audio
    python live_transcribe.py --input-file x.wav --realtime   # test without a mic
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
                     "lang_method": method})
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium", help="medium | large-v3")
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
    ap.add_argument("--word-times", action="store_true",
                    help="word-level timestamps in the saved tokens (+0.4-0.7 s per utterance)")
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
    lid = HybridLID()
    vad = StreamingVAD()
    model.transcribe(np.zeros(SR, dtype="float32"), **ct2_decode.PARTIAL_OPTS)  # warm-up

    q, stop = queue.Queue(), threading.Event()
    src = threading.Thread(target=file_source if a.input_file else mic_source,
                           args=(a.input_file, q, stop, a.realtime) if a.input_file
                           else (a.device, q, stop), daemon=True)
    src.start()
    print(f"{BOLD}{run}{RESET} ready — "
          f"{'streaming ' + a.input_file if a.input_file else 'speak now'}; Ctrl+C to stop.  "
          f"{COLOR['SI']}SI{RESET} {COLOR['EN']}EN{RESET} {COLOR['OTHER']}OTHER{RESET}\n",
          flush=True)

    silence_frames = int(a.silence_ms / 1000 * SR / BLOCK)
    number_frames = int(a.number_pause_ms / 1000 * SR / BLOCK)
    waiting_number = False                         # pause came after a digit
    preroll = []                                  # recent frames before speech
    utt, in_speech, quiet, last_partial = [], False, 0, 0.0
    t_audio = 0.0                                 # session clock, from samples received
    utt_start = 0.0
    session_audio, utterances = [], []

    def finalize(reason):
        nonlocal utt, in_speech, quiet
        audio = np.concatenate(utt) if utt else np.zeros(0, dtype="float32")
        utt, in_speech, quiet = [], False, 0
        dur = len(audio) / SR
        if dur < MIN_UTT_S:
            sys.stdout.write(CLEAR)
            return
        t_end = utt_start + dur
        t0 = time.time()
        text, words, _, action = ct2_decode.transcribe(model, audio, word_times=a.word_times)
        if not words and text:          # no alignment pass: words without times
            words = [{"word": w, "start": None, "end": None, "probability": None}
                     for w in text.split()]
        cut = action == "beam + truncate"
        dec = time.time() - t0
        uid = f"live_u{len(utterances) + 1:03d}"
        toks = label_tokens(words, lid, uid, utt_start)
        lag = dec + (a.silence_ms / 1000 if reason == "pause" else 0)
        utterances.append({"utt_id": uid, "start": round(utt_start, 2), "end": round(t_end, 2),
                           "text": text, "tokens": toks, "end_reason": reason,
                           "decode_s": round(dec, 2), "latency_after_speech_s": round(lag, 2),
                           "decode_action": action})
        sys.stdout.write(f"{CLEAR}{DIM}[{mmss(utt_start)}]{RESET} {coloured(toks, text) or DIM + '(no text)' + RESET}"
                         f"  {DIM}({dur:.1f}s speech, text {lag:.1f}s after it ended"
                         f"{', cut at max length' if reason == 'max' else ''}"
                         f"{', fixed: ' + action if action != 'greedy' else ''}){RESET}\n")
        sys.stdout.flush()

    try:
        while True:
            frame = q.get()
            if frame is None:
                if in_speech:
                    finalize("end")
                break
            if a.save_audio:
                session_audio.append(frame)
            p = vad(frame)
            t_audio += BLOCK / SR
            if not in_speech:
                preroll.append(frame)
                preroll = preroll[-int(PREROLL_S * SR / BLOCK):]
                if p >= SPEECH_ON:
                    in_speech, quiet = True, 0
                    utt = list(preroll)
                    utt_start = max(0.0, t_audio - len(utt) * BLOCK / SR)
                    last_partial = time.time()
                continue
            utt.append(frame)
            quiet = quiet + 1 if p < SPEECH_OFF else 0
            if quiet == 0:
                waiting_number = False
            if (quiet == silence_frames and not waiting_number
                    and number_frames > silence_frames):
                # Normal pause reached: if the speech so far ends on a number,
                # keep listening -- the next digit is probably coming.
                tail = np.concatenate(utt[-int(4 * SR / BLOCK):])
                text, _, _, _ = ct2_decode.transcribe(model, tail, partial=True)
                waiting_number = bool(_ENDS_IN_NUMBER.search(text))
            need = number_frames if waiting_number else silence_frames
            if quiet >= need:
                utt = utt[:-max(0, quiet - int(0.2 * SR / BLOCK))]   # keep 0.2 s of the pause
                waiting_number = False
                finalize("pause")
            elif len(utt) * BLOCK / SR >= a.max_utterance_s:
                finalize("max")
                in_speech, utt = True, []                            # speech continues
                utt_start, last_partial = t_audio, time.time()
            elif (not a.no_partials and time.time() - last_partial >= PARTIAL_EVERY_S
                  and q.qsize() < int(0.5 * SR / BLOCK)):           # only if keeping up
                text, _, _, _ = ct2_decode.transcribe(model, np.concatenate(utt), partial=True)
                width = 110
                shown = text if len(text) <= width else "…" + text[-width:]
                sys.stdout.write(f"{CLEAR}{DIM}… {shown}{RESET}")
                sys.stdout.flush()
                last_partial = time.time()
    except KeyboardInterrupt:
        if in_speech:
            finalize("stopped")
    finally:
        stop.set()

    if not utterances:
        print("\nno speech captured; nothing saved")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = a.out or os.path.join(OUT_DIR, f"{run}_{stamp}.json")
    lags = [u["latency_after_speech_s"] for u in utterances if u["end_reason"] == "pause"]
    doc = {
        "n_utterances": len(utterances),
        "transcript": " ".join(u["text"] for u in utterances if u["text"]),
        "utterances": utterances,
        "session": {
            "model": run, "model_dir": os.path.relpath(os.path.join(ct2_decode.CT2_DIR, run), C1_ROOT),
            "source": a.input_file or f"microphone {a.device if a.device is not None else '(default)'}",
            "started": stamp, "audio_s": round(t_audio, 1),
            "decode_opts": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in ct2_decode.DECODE_OPTS.items()},
            "endpointing": {"silence_ms": a.silence_ms, "number_pause_ms": a.number_pause_ms,
                            "max_utterance_s": a.max_utterance_s,
                            "vad": "silero (faster-whisper bundle), streaming"},
            "word_times": a.word_times,
            "latency_after_speech_s": {
                "median": round(float(np.median(lags)), 2) if lags else None,
                "max": round(max(lags), 2) if lags else None},
        },
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\nsaved {len(utterances)} utterances -> {out}")
    if a.save_audio and session_audio:
        import soundfile as sf
        sf.write(out[:-5] + ".wav", np.concatenate(session_audio), SR)
        print(f"audio -> {out[:-5]}.wav")


if __name__ == "__main__":
    main()
