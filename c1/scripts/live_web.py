#!/usr/bin/env python3
"""
live_web.py -- the live C1 transcriber as a web page.

The browser records the microphone and streams it here over a WebSocket;
this server runs exactly the terminal pipeline (live_transcribe.LiveSession:
Silero VAD -> fine-tuned Whisper with loop/number safeguards -> SI/EN/OTHER
labels -> CTC word times) on the GPU and streams the text back.

Usage:
    python live_web.py                      # then open http://127.0.0.1:8000
    python live_web.py --model large-v3 --port 8000

Browsers only allow microphone access on https:// or http://localhost, so
open the page on this laptop. One session at a time (one GPU). Sessions are
saved like the terminal's: predictions/live/<run>_<time>.json + .tokens.jsonl.

WebSocket protocol (/ws):
  client -> {"type": "start", "model": "medium", "partials": true}
            binary: float32 mono 16 kHz samples (any length)
            {"type": "stop"}
  server -> {"type": "status", "state": "loading"|"ready"|"busy", ...}
            LiveSession events (speech, partial, discard, utterance, times)
            {"type": "saved", "doc": session document, "json": path, "tokens": path}
            {"type": "error", "message"}
"""
import argparse
import asyncio
import gc
import json
import os
import queue
import sys
import threading
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ct2_decode  # noqa: E402  (CUDA libs first)
import live_transcribe as lt  # noqa: E402
import numpy as np  # noqa: E402

WEB_DIR = os.path.join(lt.C1_ROOT, "web")
TIMING = "ctc"


class Engine:
    """The loaded model (one at a time: the GPU has 8 GB) and the LID."""

    def __init__(self):
        from lid_hybrid import HybridLID
        self.lid = HybridLID()
        self.name = self.run = self.model = None
        self.lock = threading.Lock()                # model loading
        self.busy = False                           # one live session at a time

    def available(self):
        return [{"name": n, "run": r, "available": os.path.isdir(os.path.join(ct2_decode.CT2_DIR, r))}
                for n, r in ct2_decode.MODELS.items()]

    def ensure(self, name):
        with self.lock:
            if name == self.name:
                return
            self.model = None                       # free the old model before loading
            gc.collect()
            self.run, self.model = ct2_decode.load_model(name)
            self.name = name
            lt.warm_up(self.model, TIMING)


def build_app(engine):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="SORA C1 live transcription")

    @app.get("/api/models")
    def models():
        return {"models": engine.available(), "loaded": engine.name}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        if engine.busy:
            await sock.send_text(json.dumps({"type": "status", "state": "busy"}))
            await sock.close()
            return
        engine.busy = True
        loop = asyncio.get_running_loop()
        outbox = asyncio.Queue()

        def emit(ev):                               # from the worker thread
            loop.call_soon_threadsafe(outbox.put_nowait, ev)

        async def sender():
            while True:
                ev = await outbox.get()
                if ev is None:
                    return
                await sock.send_text(json.dumps(ev, ensure_ascii=False))

        send_task = asyncio.create_task(sender())
        frames, worker = None, None

        def work(session, frames):
            """Feeds 512-sample blocks to the session; runs the GPU work."""
            buf = np.zeros(0, dtype="float32")
            try:
                while True:
                    chunk = frames.get()
                    if chunk is None:
                        break
                    buf = np.concatenate([buf, chunk])
                    while len(buf) >= lt.BLOCK:
                        block, buf = buf[:lt.BLOCK], buf[lt.BLOCK:]
                        for ev in session.feed(block, backlog=frames.qsize()):
                            emit(ev() if callable(ev) else ev)
                for ev in session.finish("stopped"):
                    emit(ev() if callable(ev) else ev)
                if session.utterances:
                    out = os.path.join(lt.OUT_DIR, f"{session.run}_{session.started}.json")
                    doc = session.document("browser microphone (live_web.py)")
                    session.save(doc, out, out[:-5] + ".tokens.jsonl")
                    emit({"type": "saved", "doc": doc,
                          "json": os.path.relpath(out, lt.C1_ROOT),
                          "tokens": os.path.relpath(out[:-5] + ".tokens.jsonl", lt.C1_ROOT)})
                else:
                    emit({"type": "saved", "doc": None})
            except Exception as e:                  # noqa: BLE001
                traceback.print_exc()
                emit({"type": "error", "message": f"{type(e).__name__}: {e}"})

        try:
            while True:
                msg = await sock.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if frames is not None:
                        frames.put(np.frombuffer(msg["bytes"], dtype="<f4").copy())
                    continue
                cmd = json.loads(msg.get("text") or "{}")
                if cmd.get("type") == "start" and frames is None:
                    name = cmd.get("model") or engine.name
                    if name not in ct2_decode.MODELS:
                        emit({"type": "error", "message": f"unknown model {name}"})
                        continue
                    emit({"type": "status", "state": "loading", "model": name})
                    try:
                        await asyncio.to_thread(engine.ensure, name)
                    except BaseException as e:      # noqa: BLE001  (SystemExit: export missing)
                        emit({"type": "error", "message": f"could not load {name}: {e}"})
                        continue
                    session = lt.LiveSession(engine.model, engine.run, engine.lid,
                                             partials=bool(cmd.get("partials", True)), timing=TIMING)
                    frames = queue.Queue()
                    worker = threading.Thread(target=work, args=(session, frames), daemon=True)
                    worker.start()
                    emit({"type": "status", "state": "ready", "model": name, "run": engine.run})
                elif cmd.get("type") == "stop" and frames is not None:
                    frames.put(None)
                    await asyncio.to_thread(worker.join)
                    frames, worker = None, None
        except WebSocketDisconnect:
            pass
        finally:
            if frames is not None:                  # page closed mid-session: still save it
                frames.put(None)
                await asyncio.to_thread(worker.join)
            engine.busy = False
            outbox.put_nowait(None)
            try:
                await send_task
            except Exception:                       # noqa: BLE001  (socket already gone)
                pass

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium", help="model loaded at start: small | medium | large-v3")
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = this laptop only (the browser allows the mic there without https)")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"loading {a.model} ...", flush=True)
    engine = Engine()
    engine.ensure(a.model)
    import uvicorn
    print(f"ready: open http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}",
          flush=True)
    uvicorn.run(build_app(engine), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
